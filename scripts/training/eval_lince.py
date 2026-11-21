#!/usr/bin/env python
"""Score a trained anticipation checkpoint on LinCE (human-tagged tweets).

This tests out-of-domain transfer. The model was trained on SwitchLingua's generated
dialogues, and here it reads real tweets in two of the same pairs:

    ./.venv/bin/python scripts/training/eval_lince.py --ckpt models/anticipation_<TAG>.pt
    ./.venv/bin/python scripts/training/eval_lince.py --ckpt ... --splits validation --pairs spaeng

The labels come from LinCE's gold word tags (``src/lince.py``), so this measures
the model alone, with no tagger error. Everything else matches the SwitchLingua
evaluation:

* the checkpoint's tokenizer, windowing, decision point (``--decide-at``) and language
  input (``--prev-lang`` reads the gold language of the last finished word);
* the same report (``evaluation.full_report``);
* the checkpoint's own threshold ``tau``, tuned on SwitchLingua val. This is the
  primary, honest number. The "oracle" block re-tunes tau on LinCE validation and
  scores LinCE train, as an upper bound on what recalibration alone could recover.

Duration is scored at true switches whose class is known (``dur_known``), in the
head's own classes. For a 3-class head it is also shown collapsed to Short (1-6) vs
Large (7+).

Writes ``data/results/lince_<TAG>.json``. ``--save-preds`` writes an ``.npz`` in the
``train_anticipation`` layout, with the split as the prefix (``validation_*``, ``train_*``).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src import lince  # noqa: E402
from src.anticipation import collapse_duration, dur_labels  # noqa: E402
from src.evaluation import format_report  # noqa: E402
from src.metrics import duration_scores  # noqa: E402
from src.models import PRETRAINED, load_checkpoint  # noqa: E402
from src.streaming import make_loader  # noqa: E402
from src.training import evaluate, score, tune_threshold  # noqa: E402


def device_of(name: str):
    import torch
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def concat(results: list[dict]) -> dict:
    keys = results[0].keys()
    out = {}
    for k in keys:
        parts = [r[k] for r in results if len(r[k])]
        out[k] = np.concatenate(parts) if parts else results[0][k]
    return out


def two_class(res: dict, n_dur: int, cut: float | None) -> dict | None:
    """Short (1-6) vs Large (7+) at the duration points; a 3-class head is collapsed.

    For a 3-class head the prediction is Large when P(large) > ``cut`` (the
    val-tuned cut from ``duration_binary_<tag>.json`` when given, else 0.5).
    """
    if not len(res["dtrue"]):
        return None
    if n_dur == 2:
        y, p = res["dtrue"].astype(int), res["dprob"].argmax(1)
    else:
        y = collapse_duration(res["dtrue"].astype(np.int64))
        p = (res["dprob"][:, 2] > (0.5 if cut is None else cut)).astype(int)
    s = duration_scores(y, p, 2)
    return dict(cut=cut, n=int(len(y)), accuracy=float(np.mean(y == p)),
                macro_f1=float(s["macro_f1"]), per_class_f1=[float(x) for x in s["per_class_f1"]],
                share_large=float(np.mean(y == 1)))


def block(res: dict, tau: float, n_dur: int, cut) -> dict:
    rep = score(res, tau)
    y = res["y"].astype(int)
    p = float(y.mean()) if len(y) else 0.0
    rep["base_rate"] = p
    rep["all_positive_f1"] = 2 * p / (1 + p) if p else 0.0     # F1 of "always switch"
    rep["n_rows"] = int(len(np.unique(res["row"])))
    tc = two_class(res, n_dur, cut)
    if tc:
        rep["duration_two_class"] = tc
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pairs", default="spaeng,hineng")
    ap.add_argument("--splits", default="validation,train")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default=None, help="default data/results/lince_<TAG>.json")
    ap.add_argument("--save-preds", default=None, help="e.g. data/preds/lince_<TAG>.npz")
    ap.add_argument("--duration-cut", type=float, default=None,
                    help="3-class heads: P(large) cut for Short/Large (default: from "
                         "data/results/duration_binary_<TAG>.json, else 0.5)")
    ap.add_argument("--rebuild", action="store_true", help="re-tokenize the LinCE parquets")
    args = ap.parse_args()

    tag = os.path.splitext(os.path.basename(args.ckpt))[0].removeprefix("anticipation_")
    out = args.out or f"data/results/lince_{tag}.json"
    dev = device_of(args.device)
    model, ck = load_checkpoint(args.ckpt, map_location="cpu")
    model.to(dev)
    a = ck["args"]
    if a.get("model") not in PRETRAINED or a.get("unit") != "subword":
        raise SystemExit("eval_lince scores the pretrained sub-word models (xlmr / roberta / mbert)")
    if a.get("turn_sep"):
        raise SystemExit("--turn-sep runs are not supported: a tweet has one speaker turn")
    tau = float(ck["tau"])
    n_dur = int(a.get("dur_classes") or 3)
    from src.xlmr_model import BACKBONES
    backbone = a.get("backbone") or BACKBONES[a["model"]]
    blank_ids = None
    if a.get("decide_at", "word") == "gap":
        from src.subword import blank_piece_ids
        blank_ids = blank_piece_ids(backbone)
    cut = args.duration_cut
    bin_json = f"data/results/duration_binary_{tag}.json"
    if cut is None and n_dur == 3 and os.path.exists(bin_json):
        cut = float(json.load(open(bin_json))["two_class"]["cut"])
    print(f"{args.ckpt}: {a['model']} ({backbone}) | tau {tau:.3f} | {n_dur} duration classes "
          f"({', '.join(dur_labels(n_dur))}) | decide_at {a.get('decide_at', 'word')} | "
          f"prev_lang {bool(a.get('prev_lang'))} | device {dev}")

    common = dict(window=int(a["window"]), stride=int(a["stride"]), unit="subword",
                  predict_at=a.get("predict_at", "piece"), decide_at=a.get("decide_at", "word"),
                  blank_ids=blank_ids, lang_input="prev_word" if a.get("prev_lang") else "current")
    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    per_split, data = {}, {}
    for split in splits:
        parts = []
        for pair in pairs:
            path, stats = lince.build(pair, split, backbone, rebuild=args.rebuild)
            if stats:
                print(f"  built {path}: {stats}")
            t0 = time.time()
            loader = make_loader(path, None, "all", args.batch_size, 0, **common)
            res = evaluate(model, loader, dev)
            print(f"  {pair}/{split}: {len(res['y']):,} positions, {int(res['y'].sum()):,} switches "
                  f"in {time.time() - t0:.0f}s", flush=True)
            parts.append(res)
        data[split] = concat(parts)
        per_split[split] = block(data[split], tau, n_dur, cut)
        print("\n" + format_report(per_split[split], f"LinCE {split} - tau from SwitchLingua val"))
        tc = per_split[split].get("duration_two_class")
        if tc:
            print(f"  Short/Large: macro-F1 {tc['macro_f1']:.3f} | acc {tc['accuracy']:.3f} | "
                  f"n {tc['n']:,} | share Large {tc['share_large']:.2f}")

    payload = dict(ckpt=args.ckpt, tag=tag, tau=tau, n_dur=n_dur, duration_cut=cut,
                   pairs=pairs, splits=per_split, args=a)
    if len(splits) > 1:
        payload["all"] = block(concat(list(data.values())), tau, n_dur, cut)
        print("\n" + format_report(payload["all"], "LinCE, all splits - tau from SwitchLingua val"))
    if "validation" in data and "train" in data:
        v = data["validation"]
        otau, of1 = tune_threshold(1 / (1 + np.exp(-v["logit"])), v["y"])
        payload["oracle"] = dict(tau=otau, val_f1=of1, train=block(data["train"], otau, n_dur, cut))
        print("\n" + format_report(payload["oracle"]["train"],
                                   f"ORACLE: tau re-tuned on LinCE validation ({otau:.3f}), scored on LinCE train"))

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(payload, open(out, "w"), indent=2, default=float)
    print(f"\nwrote {out}")
    if args.save_preds:
        os.makedirs(os.path.dirname(args.save_preds) or ".", exist_ok=True)
        np.savez_compressed(args.save_preds, **{f"{s}_{k}": np.asarray(v)
                                                for s, r in data.items() for k, v in r.items()})
        print(f"saved predictions -> {args.save_preds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
