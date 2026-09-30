#!/usr/bin/env python
"""Tune switch thresholds offline, on predictions saved with --save-preds (no GPU).

    ./.venv/bin/python scripts/tune_thresholds.py data/preds/ablation_base.npz [more.npz ...]

A model outputs a switch probability at every position; a threshold turns it
into a decision. Training uses one global threshold tuned on val. Switch rates
differ 4x between pairs, so this compares three strategies, each tuned on VAL
and applied unchanged to TEST:

    global       one threshold for everything (what training reports)
    per_pair     one threshold per language pair
    per_dir      one threshold per direction (switching into L2 / back into L1)
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import PAIR_KEYS, full_report  # noqa: E402
from src.metrics import prf  # noqa: E402

STRATEGIES = ("global", "per_pair", "per_dir")


def _best_threshold(prob, y) -> float:
    if len(y) == 0 or y.sum() == 0:
        return 0.5
    best = (0.0, 0.5)
    for t in np.quantile(prob, np.linspace(0.30, 0.995, 120)):
        best = max(best, (prf(y, prob >= t)["f1"], float(t)))
    return best[1]


def evaluate_strategy(val: dict, test: dict, strategy: str) -> dict:
    """Tune on val, apply to test; returns the test report plus the thresholds used."""
    pv, pt = 1 / (1 + np.exp(-val["logit"])), 1 / (1 + np.exp(-test["logit"]))
    key = {"global": None, "per_pair": "pair", "per_dir": "lang"}[strategy]
    if key is None:
        th = {"all": _best_threshold(pv, val["y"])}
        pred = pt >= th["all"]
    else:
        th = {int(g): _best_threshold(pv[val[key] == g], val["y"][val[key] == g])
              for g in np.unique(val[key])}
        pred = np.zeros(len(pt), dtype=bool)
        for g, t in th.items():
            m = test[key] == g
            pred[m] = pt[m] >= t
    rep = full_report(test["y"], pred.astype(int), test["pair"], test["lang"])
    return dict(f1=rep["overall"]["f1"], macro_f1=rep["macro_f1"], thresholds=th,
                per_pair={k: v["f1"] for k, v in rep["per_pair"].items()},
                per_direction={k: v["f1"] for k, v in rep["per_direction"].items()})


def load(path: str):
    z = np.load(path)
    get = lambda split: {k: z[f"{split}_{k}"] for k in ("logit", "y", "pair", "lang")}  # noqa: E731
    return get("val"), get("test")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", nargs="+", help=".npz files written by train_anticipation.py --save-preds")
    args = ap.parse_args()
    for path in args.preds:
        val, test = load(path)
        print(f"\n== {path}  (val {len(val['y']):,} / test {len(test['y']):,} positions)")
        rows = {s: evaluate_strategy(val, test, s) for s in STRATEGIES}
        for s, r in rows.items():
            print(f"  {s:9} switch F1 {r['f1']:.3f} | macro-F1 {r['macro_f1']:.3f} | "
                  + ", ".join(f"{k} {v:.3f}" for k, v in r["per_direction"].items()))
        g, p = rows["global"]["per_pair"], rows["per_pair"]["per_pair"]
        gains = sorted(((p[k] - g[k], k) for k in g), reverse=True)
        print("  per-pair thresholds, biggest changes vs global: "
              + ", ".join(f"{k} {d:+.3f}" for d, k in gains[:3] + gains[-2:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
