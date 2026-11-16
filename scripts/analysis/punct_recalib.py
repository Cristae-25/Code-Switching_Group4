#!/usr/bin/env python
"""How much would deciding *at* the punctuation help? A training-free lower bound (no GPU).

    ./.venv/bin/python scripts/analysis/punct_recalib.py [xlmr_prev]

A switch label sits on the last language-bearing piece before any punctuation,
so the model decides at ``home`` in ``home, 然后`` without having read the comma.
A "decide at the gap" variant would move the label onto the comma. This script
estimates that variant's gain without retraining: it takes the saved logits and
adds what such a model would also know, the kind of gap that follows
(``scripts/analysis/punct_breakdown.py`` buckets), via a logistic recalibration

    p = sigmoid(w . [1, logit, onehot(bucket), onehot(bucket) * logit])

fitted on the **val** predictions and scored on **test**, with τ re-chosen on
val. The model never learns to read the comma itself, so this is a lower bound
on what a retrained decide-at-gap model can reach. The raw model at the run's
own τ is reported next to it, and must reproduce ``anticipation_<TAG>.json``.

Writes ``data/results/punct_recalib_<TAG>.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from punct_breakdown import BUCKETS, NO_LABEL, load_preds, scan  # noqa: E402
from src.evaluation import PAIR_KEYS  # noqa: E402
from src.metrics import fmt_table, prf  # noqa: E402

SPLIT_MOD = {"val": 8, "test": 9}


def features(logit: np.ndarray, bucket: np.ndarray, interact: bool = True) -> np.ndarray:
    oh = np.eye(len(BUCKETS))[bucket]
    cols = [np.ones_like(logit)[:, None], logit[:, None], oh[:, 1:]]   # ADJ is the reference
    if interact:
        cols.append(oh[:, 1:] * logit[:, None])
    return np.hstack(cols)


def fit_logistic(x: np.ndarray, y: np.ndarray, iters: int = 25, l2: float = 1e-6) -> np.ndarray:
    """Plain Newton-Raphson; the problem is tiny (11 features) and convex."""
    w = np.zeros(x.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(x @ w, -30, 30)))
        g = x.T @ (p - y) + l2 * w
        h = (x * (p * (1 - p))[:, None]).T @ x + l2 * np.eye(len(w))
        step = np.linalg.solve(h, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def best_tau(prob: np.ndarray, y: np.ndarray) -> float:
    grid = np.linspace(0.05, 0.95, 181)
    f1 = [prf(y, prob >= t)["f1"] for t in grid]
    return float(grid[int(np.argmax(f1))])


def scores(y, pred, bucket, pairs) -> dict:
    per_pair = {p: prf(y[pairs == p], pred[pairs == p])["f1"] for p in np.unique(pairs)}
    return dict(f1=prf(y, pred)["f1"],
                macro_f1=float(np.mean(list(per_pair.values()))),
                bucket_f1={b: prf(y[bucket == j], pred[bucket == j])["f1"] for j, b in enumerate(BUCKETS)},
                per_pair_f1=per_pair)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", nargs="?", default="xlmr_prev")
    ap.add_argument("--path", default="data/subword.parquet")
    args = ap.parse_args()

    lookup = {}
    for row_id, _pair, b, _y in scan(args.path):
        if row_id % 10 in SPLIT_MOD.values():
            lookup[row_id] = b

    z = load_preds(args.tag)
    run = json.load(open(f"data/results/anticipation_{args.tag}.json"))
    split = {}
    for s in SPLIT_MOD:
        rows, pos = z[f"{s}_row"], z[f"{s}_pos"]
        bucket = np.array([lookup[r][p] for r, p in zip(rows.tolist(), pos.tolist())], dtype=np.int8)
        if (bucket == NO_LABEL).any():
            raise SystemExit(f"{s}: {int((bucket == NO_LABEL).sum())} scored positions have no bucket")
        split[s] = dict(logit=z[f"{s}_logit"].astype(np.float64), y=z[f"{s}_y"].astype(np.float64),
                        bucket=bucket, pairs=np.array(PAIR_KEYS)[z[f"{s}_pair"]])

    v, t = split["val"], split["test"]
    out = {"tag": args.tag, "n_val": int(len(v["y"])), "n_test": int(len(t["y"]))}

    raw_pred = 1 / (1 + np.exp(-t["logit"])) >= run["tau"]
    out["raw"] = dict(tau=run["tau"], **scores(t["y"], raw_pred, t["bucket"], t["pairs"]))
    if abs(out["raw"]["f1"] - run["overall"]["f1"]) > 1e-3:
        raise SystemExit(f"raw F1 {out['raw']['f1']:.4f} != run json {run['overall']['f1']:.4f}")

    for name, interact in (("bucket_offset", False), ("bucket_x_logit", True)):
        w = fit_logistic(features(v["logit"], v["bucket"], interact), v["y"])
        pv = 1 / (1 + np.exp(-features(v["logit"], v["bucket"], interact) @ w))
        tau = best_tau(pv, v["y"])
        pt = 1 / (1 + np.exp(-features(t["logit"], t["bucket"], interact) @ w))
        out[name] = dict(tau=tau, weights=w.tolist(),
                         **scores(t["y"], pt >= tau, t["bucket"], t["pairs"]))

    cols = ["variant", "tau", "f1", "macro_f1", *BUCKETS]
    rows = [{"variant": k, "tau": f"{out[k]['tau']:.3f}", "f1": f"{out[k]['f1']:.4f}",
             "macro_f1": f"{out[k]['macro_f1']:.4f}",
             **{b: f"{out[k]['bucket_f1'][b]:.3f}" for b in BUCKETS}}
            for k in ("raw", "bucket_offset", "bucket_x_logit")]
    print(fmt_table(rows, cols, f"{args.tag}: test switch F1, raw vs gap-aware recalibration (fit on val)"))
    pp = [{"pair": p, **{k: f"{out[k]['per_pair_f1'][p]:.3f}" for k in ("raw", "bucket_x_logit")}}
          for p in out["raw"]["per_pair_f1"]]
    print(fmt_table(pp, ["pair", "raw", "bucket_x_logit"], "per pair"))
    dest = f"data/results/punct_recalib_{args.tag}.json"
    json.dump(out, open(dest, "w"), indent=1, default=float)
    print("wrote", dest)


if __name__ == "__main__":
    main()
