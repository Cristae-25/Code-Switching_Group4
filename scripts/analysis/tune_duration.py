#!/usr/bin/env python
"""Tune the duration decision offline, on predictions saved with --save-preds (no GPU).

    ./.venv/bin/python scripts/analysis/tune_duration.py data/preds/ablation_base.npz [more.npz ...]

The duration head's argmax leans toward Large, the majority class. This fits
one log-offset per class on VAL to maximise macro-F1 (``tune_class_offsets``),
applies it unchanged to TEST, and prints test before and after, overall and,
when the file records each point's pair (``test_dpair``), per pair.
``--per-pair`` fits separate offsets for each pair instead of one global set.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.anticipation import DUR_LABELS  # noqa: E402
from src.evaluation import PAIR_KEYS, duration_by_pair  # noqa: E402
from src.metrics import (apply_class_offsets, duration_scores, fmt_table,  # noqa: E402
                         tune_class_offsets)


def tuned_predictions(z, per_pair: bool = False):
    """``(test predictions, offsets)``; offsets are one array, or one per pair."""
    if not per_pair or "val_dpair" not in z.files:
        off, _ = tune_class_offsets(z["val_dprob"], z["val_dtrue"])
        return apply_class_offsets(z["test_dprob"], off), off
    pred = apply_class_offsets(z["test_dprob"], np.zeros(z["test_dprob"].shape[1]))
    offs = {}
    for p in np.unique(z["test_dpair"]):
        v, t = z["val_dpair"] == p, z["test_dpair"] == p
        if v.any():
            offs[PAIR_KEYS[int(p)]], _ = tune_class_offsets(z["val_dprob"][v], z["val_dtrue"][v])
            pred[t] = apply_class_offsets(z["test_dprob"][t], offs[PAIR_KEYS[int(p)]])
    return pred, offs


def _row(name, d):
    return dict(name=name, n=f"{d['n']:,}", acc=f"{d['accuracy']:.3f}",
                macroF1=f"{d['macro_f1']:.3f}",
                **{f"F1_{c}": f"{f:.3f}" for c, f in zip(DUR_LABELS, d["per_class_f1"])})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", nargs="+")
    ap.add_argument("--per-pair", action="store_true", help="separate offsets per pair")
    ap.add_argument("--json", default=None, help="write the before/after scores here")
    args = ap.parse_args()
    cols = ["name", "n", "acc", "macroF1"] + [f"F1_{c}" for c in DUR_LABELS]
    out = {}
    for path in args.preds:
        z = np.load(path)
        name = os.path.splitext(os.path.basename(path))[0]
        before = z["test_dprob"].argmax(-1)
        after, off = tuned_predictions(z, args.per_pair)
        b, a = duration_scores(z["test_dtrue"], before), duration_scores(z["test_dtrue"], after)
        print(f"\n== {name}: offsets (small, medium, large) "
              + (np.round(off, 2).tolist() if isinstance(off, np.ndarray)
                 else {k: np.round(v, 2).tolist() for k, v in off.items()}).__str__())
        rows = [_row("argmax", b), _row("val-tuned", a)]
        out[name] = dict(argmax=b, tuned=a, offsets=off if isinstance(off, dict) else off.tolist())
        if "test_dpair" in z.files:
            pb = duration_by_pair(z["test_dtrue"], before, z["test_dpair"])
            pa = duration_by_pair(z["test_dtrue"], after, z["test_dpair"])
            for k in sorted(pb):
                rows += [_row(f"  {k} argmax", pb[k]), _row(f"  {k} tuned", pa[k])]
            out[name].update(argmax_per_pair=pb, tuned_per_pair=pa)
        print(fmt_table(rows, cols))
    if args.json:
        json.dump(out, open(args.json, "w"), indent=2,
                  default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
