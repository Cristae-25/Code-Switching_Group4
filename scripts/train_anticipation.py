#!/usr/bin/env python
"""Train the causal transformer baseline and score it against the naive rules.

Run order:
    python scripts/train_anticipation.py --build-vocab      # once, ~1 min
    python scripts/train_anticipation.py --steps 4000

Everything is streamed; the process never holds more than a shuffle buffer of
windows. Evaluation reuses ``src.metrics`` so the numbers are directly
comparable with ``scripts/baseline_anticipation.py``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.anticipation import DUR_LABELS, N_DUR  # noqa: E402
from src.causal_model import CausalSwitchModel, check_causality  # noqa: E402
from src.metrics import by_group, duration_scores, fmt_table, macro_f1, prf  # noqa: E402
from src.streaming import build_vocab, make_loader  # noqa: E402

VOCAB_PATH = "data/vocab_anticipation.json"
PAIRS = ["Cantonese", "Arabic", "Philippines", "German", "Chinese", "Vietnamese",
         "Malay", "Japanese", "Hindi", "Korean", "Spanish", "French", "Russian",
         "Italian"]
PAIR_INDEX = {p: i for i, p in enumerate(PAIRS)}


def device_of(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(model, loader, dev, max_batches=0, threshold=0.5):
    model.eval()
    logits, truth, pairs, dtrue, dpred = [], [], [], [], []
    for i, b in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        sw, dur = model(b["word_ids"].to(dev), b["lang_ids"].to(dev),
                        b["pair_id"].to(dev))
        m = b["sw_mask"].to(dev)
        logits.append(sw[m].float().cpu().numpy())
        truth.append(b["y_sw"].to(dev)[m].cpu().numpy())
        pairs.append(b["pair_id"].to(dev).unsqueeze(1).expand_as(m)[m].cpu().numpy())
        dm = b["dur_mask"].to(dev)
        if dm.any():
            dtrue.append(b["y_dur"].to(dev)[dm].cpu().numpy())
            dpred.append(dur[dm].argmax(-1).cpu().numpy())
    cat = lambda xs: np.concatenate(xs) if xs else np.empty(0)  # noqa: E731
    return (cat(logits), cat(truth), cat(pairs), cat(dtrue), cat(dpred))


def tune_threshold(prob, truth):
    """Pick the F1-maximising threshold on a held-out split."""
    best = (0.0, 0.5)
    for t in np.quantile(prob, np.linspace(0.50, 0.999, 60)):
        best = max(best, (prf(truth, prob >= t)["f1"], float(t)))
    return best[1], best[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="data/tagged.parquet")
    ap.add_argument("--build-vocab", action="store_true")
    ap.add_argument("--vocab-size", type=int, default=50_000)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--stride", type=int, default=192)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dur-weight", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--eval-batches", type=int, default=0,
                    help="0 = score the whole split")
    ap.add_argument("--train-langs", default="",
                    help="comma-separated first_languages to TRAIN on "
                         "(default: all). Everything else becomes the zero-shot set.")
    ap.add_argument("--no-pair", action="store_true",
                    help="drop the language-pair embedding - required for a "
                         "held-out pair the model has never seen")
    ap.add_argument("--out", default="data/anticipation_model.json")
    ap.add_argument("--ckpt", default="models/anticipation_causal.pt")
    ap.add_argument("--names", choices=["script_words", "transparent", "script", "hybrid", "language"],
                    default="script_words",
                    help="how named entities count (see src/segments.py); needs an "
                         "is_entity column, else names count as tagged")
    args = ap.parse_args()
    import src.anticipation as _ant
    _ant.NAMES = args.names

    if args.build_vocab or not os.path.exists(VOCAB_PATH):
        t0 = time.time()
        print("building vocabulary from the train split ...", flush=True)
        vocab = build_vocab(args.path, args.vocab_size)
        os.makedirs(os.path.dirname(VOCAB_PATH), exist_ok=True)
        json.dump(vocab, open(VOCAB_PATH, "w"))
        print(f"  {len(vocab):,} types in {time.time()-t0:.0f}s -> {VOCAB_PATH}")
    vocab = json.load(open(VOCAB_PATH))
    print(f"vocab {len(vocab):,}")

    train_langs = [x.strip() for x in args.train_langs.split(",") if x.strip()]
    holdout = [p for p in PAIRS if p not in train_langs] if train_langs else []
    if holdout:
        print(f"train languages : {', '.join(train_langs)}")
        print(f"zero-shot held out: {', '.join(holdout)}")

    dev = device_of(args.device)
    model = CausalSwitchModel(len(vocab), d_model=args.d_model,
                              n_layers=args.layers, n_heads=args.heads,
                              d_ff=args.d_model * 2, max_len=args.window,
                              n_pairs=len(PAIRS),
                              use_pair=not (args.no_pair or bool(holdout))).to(dev)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"device {dev} | {n_param/1e6:.2f}M params")

    chk = check_causality(model, len(vocab), seq_len=32, device="cpu")
    print(f"causality check: max logit drift {chk['max_abs_logit_drift']:.2e} "
          f"over {chk['positions_tested']} prefixes -> "
          f"{'PREFIX-ONLY OK' if chk['causal'] else 'LEAK'}")
    if not chk["causal"]:
        raise SystemExit("aborting: model is not causal")
    model.to(dev)

    common = dict(window=args.window, stride=args.stride, pair_index=PAIR_INDEX)
    train_loader = make_loader(args.path, vocab, "train", args.batch_size,
                               args.workers, languages=train_langs or None, **common)
    val_loader = make_loader(args.path, vocab, "val", args.batch_size, 0,
                             languages=train_langs or None, **common)
    test_loader = make_loader(args.path, vocab, "test", args.batch_size, 0,
                              languages=train_langs or None, **common)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    ce = nn.CrossEntropyLoss(reduction="none")

    step, t0, running = 0, time.time(), []
    model.train()
    while step < args.steps:
        for b in train_loader:
            if step >= args.steps:
                break
            sw, dur = model(b["word_ids"].to(dev), b["lang_ids"].to(dev),
                            b["pair_id"].to(dev))
            m = b["sw_mask"].to(dev)
            loss = (bce(sw, b["y_sw"].to(dev).float()) * m).sum() / m.sum().clamp(min=1)
            dm = b["dur_mask"].to(dev)
            if dm.any():
                dl = ce(dur.reshape(-1, N_DUR), b["y_dur"].to(dev).reshape(-1))
                loss = loss + args.dur_weight * (
                    (dl * dm.reshape(-1)).sum() / dm.sum().clamp(min=1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running.append(loss.detach().item())
            step += 1
            if step % 200 == 0:
                print(f"  step {step:>5}/{args.steps}  loss {np.mean(running[-200:]):.4f}"
                      f"  {step/(time.time()-t0):.1f} it/s", flush=True)
            model.train()

    print(f"trained {step} steps in {(time.time()-t0)/60:.1f} min")

    vl, vt, _, _, _ = evaluate(model, val_loader, dev, args.eval_batches)
    vprob = 1 / (1 + np.exp(-vl))
    tau, vf1 = tune_threshold(vprob, vt)
    print(f"threshold tuned on val: tau={tau:.3f} (val F1 {vf1:.3f})")

    tl, tt, tp, dt_, dp_ = evaluate(model, test_loader, dev, args.eval_batches)
    prob = 1 / (1 + np.exp(-tl))
    pred = (prob >= tau).astype(int)
    s = prf(tt, pred)
    groups = by_group(tt, pred, tp)
    named = {PAIRS[int(k)]: v for k, v in groups.items()}

    print(fmt_table([dict(model="causal-transformer",
                          P=f"{s['precision']:.3f}", R=f"{s['recall']:.3f}",
                          F1=f"{s['f1']:.3f}", macroF1=f"{macro_f1(named):.3f}",
                          acc=f"{s['accuracy']:.3f}", n=f"{s['n']:,}")],
                    ["model", "P", "R", "F1", "macroF1", "acc", "n"],
                    f"\nTEST split (base rate {tt.mean():.4f})"))

    print("\nPer language pair:")
    print(fmt_table(
        [dict(pair=k, n=f"{v['n']:,}", base=f"{v['base_rate']:.3f}",
              P=f"{v['precision']:.3f}", R=f"{v['recall']:.3f}", F1=f"{v['f1']:.3f}")
         for k, v in sorted(named.items(), key=lambda kv: -kv[1]["f1"])],
        ["pair", "n", "base", "P", "R", "F1"]))

    if holdout:
        zs_loader = make_loader(args.path, vocab, "test", args.batch_size, 0,
                                languages=holdout, **common)
        zl, zt, zp, _, _ = evaluate(model, zs_loader, dev, args.eval_batches)
        zpred = (1 / (1 + np.exp(-zl)) >= tau).astype(int)
        zs = prf(zt, zpred)
        zgroups = {PAIRS[int(k)]: v
                   for k, v in by_group(zt, zpred, zp).items()}
        print("\nZERO-SHOT - language pairs never seen in training:")
        print(fmt_table(
            [dict(pair=k, n=f"{v['n']:,}", base=f"{v['base_rate']:.3f}",
                  P=f"{v['precision']:.3f}", R=f"{v['recall']:.3f}",
                  F1=f"{v['f1']:.3f}")
             for k, v in sorted(zgroups.items(), key=lambda kv: -kv[1]["f1"])]
            + [dict(pair="ALL held out", n=f"{zs['n']:,}",
                    base=f"{zs['base_rate']:.3f}", P=f"{zs['precision']:.3f}",
                    R=f"{zs['recall']:.3f}", F1=f"{zs['f1']:.3f}")],
            ["pair", "n", "base", "P", "R", "F1"]))

    ds = duration_scores(dt_, dp_)
    print(f"\nDuration head (oracle switch positions, n={ds['n']:,}): "
          f"bucket acc {ds['accuracy']:.3f} | within one {ds['within_one']:.3f} "
          f"| MAE {ds['mae_buckets']:.3f}")

    os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
    torch.save(dict(state=model.state_dict(), args=vars(args), tau=tau,
                    vocab_size=len(vocab)), args.ckpt)
    payload = dict(overall=s, per_pair=named, macro_f1=macro_f1(named),
                   duration=ds, tau=tau, val_f1=vf1, params=n_param,
                   causality=chk, steps=step, args=vars(args))
    if holdout:
        payload["zero_shot"] = dict(overall=zs, per_pair=zgroups,
                                    macro_f1=macro_f1(zgroups), holdout=holdout)
    json.dump(payload,
              open(args.out, "w"), indent=2, default=float)
    print(f"\nwrote {args.out} and {args.ckpt}")


if __name__ == "__main__":
    main()
