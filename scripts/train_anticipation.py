#!/usr/bin/env python
"""Train an anticipation model and score it with the shared evaluation report.

    # the from-scratch causal transformer (words, or XLM-R pieces)
    python scripts/train_anticipation.py --build-vocab --path data/tagged_shuffled.parquet --steps 6000
    python scripts/train_anticipation.py --unit subword --build-vocab --steps 6000
    # XLM-R with a causal mask (needs a GPU: see kaggle/train_xlmr.ipynb)
    python scripts/train_anticipation.py --model xlmr --unit subword --steps 12000 --batch-size 16

Protocol (identical for every model): train on the train split; tune the
switch threshold on val; score test once with ``src.evaluation.full_report``
(switch P/R/F1 per pair, macro-F1, per direction; 3-class duration at true
switch points). ``check_causality`` must pass before training starts.
Everything is streamed; the process never holds more than a shuffle buffer.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.anticipation import N_DUR  # noqa: E402
from src.causal_model import check_causality  # noqa: E402
from src.evaluation import format_report, full_report  # noqa: E402
from src.metrics import prf  # noqa: E402
from src.models import build_model  # noqa: E402
from src.streaming import (DEFAULT_PATHS, VOCAB_PATHS, build_subword_vocab,  # noqa: E402
                           build_vocab, make_loader)
from src.subword import XLMR_VOCAB  # noqa: E402

UNIT_DEFAULTS = {"word": dict(window=256, stride=192, vocab_size=50_000,
                              out="data/anticipation_word.json",
                              ckpt="models/anticipation_word.pt"),
                 "subword": dict(window=384, stride=288, vocab_size=100_000,
                                 out="data/anticipation_model_subword.json",
                                 ckpt="models/anticipation_subword.pt")}
MODEL_DEFAULTS = {"scratch": dict(lr=3e-4),
                  "xlmr": dict(lr=2e-5, out="data/anticipation_xlmr.json",
                               ckpt="models/anticipation_xlmr.pt")}
PAIRS = ["Cantonese", "Arabic", "Philippines", "German", "Chinese", "Vietnamese",
         "Malay", "Japanese", "Hindi", "Korean", "Spanish", "French", "Russian",
         "Italian"]
PAIR_INDEX = {p: i for i, p in enumerate(PAIRS)}


def resolve_defaults(args):
    """Fill model- and unit-dependent defaults; no run overwrites another model's files."""
    model = getattr(args, "model", "scratch")
    if model == "xlmr" and args.unit != "subword":
        raise SystemExit("--model xlmr reads XLM-R pieces: use --unit subword")
    for k, v in {**UNIT_DEFAULTS[args.unit], **MODEL_DEFAULTS[model]}.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)
    if args.path is None:
        args.path = DEFAULT_PATHS[args.unit]
    if args.unit == "subword" and args.names != "script_words":
        # data/subword.parquet was built with the default names policy baked in.
        raise SystemExit("--names only applies to --unit word: the sub-word labels were "
                         "built with script_words (use kaggle/load_labels.ipynb for others)")
    if args.window > 512 and model == "xlmr":
        raise SystemExit("XLM-R reads at most 512 positions: use --window 512 or less")
    return args


def device_of(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def autocast(dev):
    """fp16 on CUDA (a T4 has no bf16); full precision elsewhere."""
    if dev.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


@torch.no_grad()
def evaluate(model, loader, dev, max_batches=0):
    """Logits and labels at every scored position, with pair and direction."""
    model.eval()
    out = {k: [] for k in ("logit", "y", "pair", "lang", "dtrue", "dpred")}
    for i, b in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        with autocast(dev):
            sw, dur = model(b["word_ids"].to(dev), b["lang_ids"].to(dev), b["pair_id"].to(dev))
        m = b["sw_mask"]
        out["logit"].append(sw.float().cpu()[m].numpy())
        out["y"].append(b["y_sw"][m].numpy())
        out["pair"].append(b["report_pair"].unsqueeze(1).expand_as(m)[m].numpy())
        out["lang"].append(b["lang_ids"][m].numpy())
        dm = b["dur_mask"]
        if dm.any():
            out["dtrue"].append(b["y_dur"][dm].numpy())
            out["dpred"].append(dur.float().cpu()[dm].argmax(-1).numpy())
    return {k: np.concatenate(v) if v else np.empty(0) for k, v in out.items()}


def make_scheduler(opt, lrs, steps):
    """One-cycle schedule; the warm-up is at least 2 steps (OneCycleLR divides by it)."""
    if steps < 4:
        return torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    return torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lrs, total_steps=steps,
                                               pct_start=min(0.5, max(0.1, 2.5 / steps)))


def tune_threshold(prob, truth):
    """Pick the F1-maximising threshold on a held-out split."""
    best = (0.0, 0.5)
    for t in np.quantile(prob, np.linspace(0.50, 0.999, 60)):
        best = max(best, (prf(truth, prob >= t)["f1"], float(t)))
    return best[1], best[0]


def score(res, tau):
    pred = (1 / (1 + np.exp(-res["logit"])) >= tau).astype(int)
    return full_report(res["y"], pred, res["pair"], res["lang"], res["dtrue"], res["dpred"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["scratch", "xlmr"], default="scratch")
    ap.add_argument("--xlmr-name", default="xlm-roberta-base")
    ap.add_argument("--path", default=None)
    ap.add_argument("--build-vocab", action="store_true")
    ap.add_argument("--vocab-size", type=int, default=None)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=None, help="default: 3e-4 scratch, 2e-5 xlmr")
    ap.add_argument("--head-lr", type=float, default=1e-3, help="xlmr: learning rate of the two heads")
    ap.add_argument("--dur-weight", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--eval-batches", type=int, default=0, help="0 = score the whole split")
    ap.add_argument("--train-langs", default="",
                    help="comma-separated first_languages to TRAIN on (default: all). "
                         "Everything else becomes the zero-shot set.")
    ap.add_argument("--no-pair", action="store_true",
                    help="drop the language-pair embedding (scratch model)")
    ap.add_argument("--unit", choices=["word", "subword"], default="word")
    ap.add_argument("--no-lang", action="store_true",
                    help="no language input (implied by --unit subword)")
    ap.add_argument("--out", default=None, help="default depends on --model and --unit")
    ap.add_argument("--ckpt", default=None, help="default depends on --model and --unit")
    ap.add_argument("--names", choices=["script_words", "transparent", "script", "hybrid", "language"],
                    default="script_words",
                    help="how named entities count (see src/segments.py); word units only")
    args = ap.parse_args()
    import src.anticipation as _ant
    _ant.NAMES = args.names
    resolve_defaults(args)
    use_lang = not (args.no_lang or args.unit == "subword")

    if args.model == "xlmr":
        vocab, vocab_size = None, XLMR_VOCAB          # raw XLM-R ids, no vocabulary
    else:
        vocab_path = VOCAB_PATHS[args.unit]
        if args.build_vocab or not os.path.exists(vocab_path):
            t0 = time.time()
            print("building vocabulary from the train split ...", flush=True)
            builder = build_vocab if args.unit == "word" else build_subword_vocab
            vocab = builder(args.path, args.vocab_size)
            os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
            json.dump(vocab, open(vocab_path, "w"))
            print(f"  {len(vocab):,} types in {time.time()-t0:.0f}s -> {vocab_path}")
        vocab = json.load(open(vocab_path))
        vocab_size = len(vocab)
    print(f"model {args.model} | unit {args.unit} | vocab {vocab_size:,} | "
          f"language input {'on' if use_lang else 'off'}")

    train_langs = [x.strip() for x in args.train_langs.split(",") if x.strip()]
    holdout = [p for p in PAIRS if p not in train_langs] if train_langs else []
    if holdout:
        print(f"train languages : {', '.join(train_langs)}")
        print(f"zero-shot held out: {', '.join(holdout)}")

    dev = device_of(args.device)
    model = build_model(vars(args), vocab_size, n_pairs=len(PAIRS),
                        use_pair=args.model == "scratch" and not (args.no_pair or bool(holdout)),
                        use_lang=use_lang)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"device {dev} | {n_param/1e6:.1f}M params")

    chk = check_causality(model, vocab_size, seq_len=32, device="cpu")
    print(f"causality check: max logit drift {chk['max_abs_logit_drift']:.2e} "
          f"over {chk['positions_tested']} prefixes -> "
          f"{'PREFIX-ONLY OK' if chk['causal'] else 'LEAK'}")
    if not chk["causal"]:
        raise SystemExit("aborting: model is not causal")
    model.to(dev)

    common = dict(window=args.window, stride=args.stride, pair_index=PAIR_INDEX, unit=args.unit)
    loader = lambda split, langs, workers=0: make_loader(  # noqa: E731
        args.path, vocab, split, args.batch_size, workers, languages=langs, **common)
    train_loader = loader("train", train_langs or None, args.workers)
    val_loader, test_loader = loader("val", train_langs or None), loader("test", train_langs or None)

    if args.model == "xlmr":                           # pretrained body slow, new heads fast
        heads = [p for n, p in model.named_parameters() if n.startswith("head_")]
        body = [p for n, p in model.named_parameters() if not n.startswith("head_")]
        groups, lrs = [dict(params=body, lr=args.lr), dict(params=heads, lr=args.head_lr)], \
            [args.lr, args.head_lr]
    else:
        groups, lrs = [dict(params=list(model.parameters()), lr=args.lr)], [args.lr]
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    sched = make_scheduler(opt, lrs, args.steps)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    bce = nn.BCEWithLogitsLoss(reduction="none")
    ce = nn.CrossEntropyLoss(reduction="none")

    step, t0, running = 0, time.time(), []
    model.train()
    while step < args.steps:
        for b in train_loader:
            if step >= args.steps:
                break
            with autocast(dev):
                sw, dur = model(b["word_ids"].to(dev), b["lang_ids"].to(dev),
                                b["pair_id"].to(dev))
            sw, dur = sw.float(), dur.float()
            m = b["sw_mask"].to(dev)
            loss = (bce(sw, b["y_sw"].to(dev).float()) * m).sum() / m.sum().clamp(min=1)
            dm = b["dur_mask"].to(dev)
            if dm.any():
                dl = ce(dur.reshape(-1, N_DUR), b["y_dur"].to(dev).reshape(-1))
                loss = loss + args.dur_weight * ((dl * dm.reshape(-1)).sum() / dm.sum().clamp(min=1))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running.append(loss.detach().item())
            step += 1
            if step % 200 == 0:
                print(f"  step {step:>5}/{args.steps}  loss {np.mean(running[-200:]):.4f}"
                      f"  {step/(time.time()-t0):.2f} it/s", flush=True)
    print(f"trained {step} steps in {(time.time()-t0)/60:.1f} min")

    val = evaluate(model, val_loader, dev, args.eval_batches)
    tau, vf1 = tune_threshold(1 / (1 + np.exp(-val["logit"])), val["y"])
    print(f"threshold tuned on val: tau={tau:.3f} (val F1 {vf1:.3f})")

    report = score(evaluate(model, test_loader, dev, args.eval_batches), tau)
    print("\n" + format_report(report, f"TEST split - {args.model} ({args.unit})"))
    payload = dict(model=args.model, unit=args.unit, **report, tau=tau, val_f1=vf1,
                   params=n_param, causality=chk, steps=step, args=vars(args))
    if holdout:
        zs = score(evaluate(model, loader("test", holdout), dev, args.eval_batches), tau)
        print("\n" + format_report(zs, "ZERO-SHOT - language pairs never seen in training"))
        payload["zero_shot"] = dict(**zs, holdout=holdout)

    os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)
    torch.save(dict(state=model.state_dict(), args=vars(args), tau=tau,
                    vocab_size=vocab_size, use_lang=use_lang), args.ckpt)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(payload, open(args.out, "w"), indent=2, default=float)
    print(f"\nwrote {args.out} and {args.ckpt}")


if __name__ == "__main__":
    main()
