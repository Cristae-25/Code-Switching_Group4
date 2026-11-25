#!/usr/bin/env python
"""Paired tweet-level bootstrap between two models' LinCE predictions (no GPU).

    ./.venv/bin/python scripts/analysis/bootstrap_lince.py xlmr_prev_gap_36k_rsv2 xlmr_prev_gap_36k_rsv2_lince5

Reads ``data/preds/lince_<TAG>.npz`` (``eval_lince.py --save-preds``) and scores each
model at its own checkpoint tau (from ``data/results/lince_<TAG>.json``). Only the
splits both files contain are used, so a multi-dataset run scored on LinCE validation
only is compared with an older model on that same split. The two models may use
different tokenizers, so the resampling unit is the tweet: TP/FP/FN are summed per tweet,
the tweets common to both are resampled with replacement, and F1 is recomputed.
Prints F1 with a 95% interval for each model and the paired difference (B - A),
pooled and per pair. Writes ``data/results/bootstrap_lince_<A>_vs_<B>.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.lince import PAIRS, describe_row  # noqa: E402


def tweet_counts(tag: str, splits: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """``(row_ids, counts[n_tweets, 3])`` with TP, FP, FN per tweet at the run's tau."""
    z = np.load(f"data/preds/lince_{tag}.npz")
    tau = json.load(open(f"data/results/lince_{tag}.json"))["tau"]
    y = np.concatenate([z[f"{s}_y"] for s in splits]).astype(bool)
    pred = 1 / (1 + np.exp(-np.concatenate([z[f"{s}_logit"] for s in splits]))) >= tau
    rows, inv = np.unique(np.concatenate([z[f"{s}_row"] for s in splits]), return_inverse=True)
    c = np.stack([np.bincount(inv, weights=w, minlength=len(rows)) for w in (y & pred, ~y & pred, y & ~pred)], 1)
    return rows, c


def f1(c: np.ndarray) -> np.ndarray:
    tp, fp, fn = c[..., 0], c[..., 1], c[..., 2]
    d = 2 * tp + fp + fn
    return np.divide(2 * tp, d, out=np.zeros_like(tp, dtype=float), where=d > 0)


def paired(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    fa, fb = f1(a[idx].sum(1)), f1(b[idx].sum(1))
    ci = lambda x: [float(v) for v in np.percentile(x, [2.5, 97.5])]  # noqa: E731
    return dict(n_tweets=int(len(a)), f1_a=float(f1(a.sum(0))), f1_b=float(f1(b.sum(0))),
                ci_a=ci(fa), ci_b=ci(fb), diff=float(f1(b.sum(0)) - f1(a.sum(0))), diff_ci=ci(fb - fa))


def splits_of(tag: str) -> set[str]:
    return {k.split("_", 1)[0] for k in np.load(f"data/preds/lince_{tag}.npz").files}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    splits = sorted(splits_of(args.a) & splits_of(args.b))
    if not splits:
        raise SystemExit("the two prediction files share no LinCE split")
    ra, ca = tweet_counts(args.a, splits)
    rb, cb = tweet_counts(args.b, splits)
    common = np.intersect1d(ra, rb)
    ca, cb = ca[np.searchsorted(ra, common)], cb[np.searchsorted(rb, common)]
    pair = np.array([describe_row(int(r))[0] for r in common])
    out = dict(a=args.a, b=args.b, splits=splits, n_boot=args.n_boot,
               pooled=paired(ca, cb, args.n_boot, args.seed),
               per_pair={PAIRS[p]: paired(ca[pair == p], cb[pair == p], args.n_boot, args.seed)
                         for p in PAIRS if (pair == p).any()})
    print(f"LinCE {'+'.join(splits)}: A = {args.a}, B = {args.b}")
    for name, r in [("pooled", out["pooled"]), *out["per_pair"].items()]:
        print(f"  {name:8s} {r['n_tweets']:6,} tweets | A {r['f1_a']:.3f} [{r['ci_a'][0]:.3f}, {r['ci_a'][1]:.3f}]"
              f" | B {r['f1_b']:.3f} [{r['ci_b'][0]:.3f}, {r['ci_b'][1]:.3f}]"
              f" | B - A {r['diff']:+.3f} [{r['diff_ci'][0]:+.3f}, {r['diff_ci'][1]:+.3f}]")
    dest = f"data/results/bootstrap_lince_{args.a}_vs_{args.b}.json"
    json.dump(out, open(dest, "w"), indent=1)
    print("wrote", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
