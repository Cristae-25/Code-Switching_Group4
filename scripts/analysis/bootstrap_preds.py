#!/usr/bin/env python
"""Row-level bootstrap CIs on saved test predictions (no GPU).

    ./.venv/bin/python scripts/analysis/bootstrap_preds.py xlmr_6pairs [xlmr_durw_6pairs] [--n 1000]

Each TAG reads ``data/preds/<TAG>.npz`` (written by ``--save-preds``), or the folder it unzips to, and the
switch threshold from ``data/results/anticipation_<TAG>.json``. Conversations, not
positions, are resampled: positions in one row are correlated, so a
position-level bootstrap would give intervals that are too narrow.

For every pair (and overall) it reports the 95% interval of the switch F1,
the duration macro-F1, the Medium F1 and the end-to-end duration accuracy.
Given two TAGs, the same resampled rows score both, so the interval of
``B - A`` is a paired test: a difference whose interval excludes 0 is real.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.evaluation import PAIR_KEYS  # noqa: E402
from src.metrics import fmt_table  # noqa: E402

N_DUR = 3
METRICS = ("switch_f1", "dur_macro_f1", "dur_medium_f1", "e2e_acc")


def load(tag: str, data_dir: str = "data") -> dict:
    path = os.path.join(data_dir, "preds", tag)
    if os.path.isdir(path):         # an unzipped .npz: a folder of <name>.npy files
        z = {f[:-4]: np.load(os.path.join(path, f)) for f in os.listdir(path) if f.endswith(".npy")}
    else:
        z = np.load(path + ".npz")
    if "test_drow" not in z:
        raise SystemExit(f"preds_{tag}.npz predates per-row duration arrays: re-run it")
    tau = json.load(open(os.path.join(data_dir, "results", f"anticipation_{tag}.json")))["tau"]
    fired = 1 / (1 + np.exp(-z["test_dlogit"])) >= tau
    return dict(row=z["test_row"], pair=z["test_pair"], y=z["test_y"],
                pred=(1 / (1 + np.exp(-z["test_logit"])) >= tau).astype(int),
                drow=z["test_drow"], dtrue=z["test_dtrue"], dpred=z["test_dpred"], fired=fired)


def row_counts(p: dict, rows: np.ndarray) -> np.ndarray:
    """Per-row sufficient statistics, so a resample is one matrix product.

    Columns: switch tp, fp, fn; the 3x3 duration confusion; end-to-end hits
    per class, end-to-end false positives per class.
    """
    idx = {r: i for i, r in enumerate(rows)}
    out = np.zeros((len(rows), 3 + N_DUR * N_DUR + 2 * N_DUR))
    ri = np.array([idx[r] for r in p["row"]])
    y, yp = p["y"], p["pred"]
    for col, m in enumerate(((y == 1) & (yp == 1), (y == 0) & (yp == 1), (y == 1) & (yp == 0))):
        np.add.at(out[:, col], ri[m], 1)
    di = np.array([idx[r] for r in p["drow"]], dtype=np.int64)
    dt, dp, f = p["dtrue"], p["dpred"], p["fired"]
    np.add.at(out, (di, 3 + dt * N_DUR + dp), 1)
    hit = f & (dt == dp)
    np.add.at(out, (di[hit], 3 + N_DUR * N_DUR + dt[hit]), 1)
    fp = f & (dt != dp)
    np.add.at(out, (di[fp], 3 + N_DUR * N_DUR + N_DUR + dp[fp]), 1)
    return out


def metrics(s: np.ndarray) -> np.ndarray:
    """``METRICS`` from summed statistics; ``s`` is [..., columns]."""
    tp, fp, fn = s[..., 0], s[..., 1], s[..., 2]
    conf = s[..., 3:3 + N_DUR * N_DUR].reshape(*s.shape[:-1], N_DUR, N_DUR)
    hits = s[..., 3 + N_DUR * N_DUR:3 + N_DUR * N_DUR + N_DUR]
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.nan_to_num(2 * tp / (2 * tp + fp + fn))
        diag = np.diagonal(conf, axis1=-2, axis2=-1)
        cf1 = np.nan_to_num(2 * diag / (conf.sum(-1) + conf.sum(-2)))
        e2e = np.nan_to_num(hits.sum(-1) / conf.sum((-1, -2)))
    return np.stack([f1, cf1.mean(-1), cf1[..., 1], e2e], axis=-1)


def bootstrap(stats: list[np.ndarray], pair_of_row: np.ndarray, n: int, seed: int = 0):
    """``{scope: [point, lo, hi] per run (and B - A)}`` for overall and each pair."""
    rng = np.random.default_rng(seed)
    scopes = {"overall": np.ones(len(pair_of_row), bool)}
    scopes.update({PAIR_KEYS[int(p)]: pair_of_row == p for p in np.unique(pair_of_row)})
    out = {}
    for name, sel in scopes.items():
        rows = np.flatnonzero(sel)
        counts = np.stack([np.bincount(rng.choice(len(rows), len(rows)), minlength=len(rows))
                           for _ in range(n)])                       # [n, rows]
        point = [metrics(st[rows].sum(0)) for st in stats]
        draws = [metrics(counts @ st[rows]) for st in stats]          # [n, metrics]
        if len(stats) == 2:
            point.append(point[1] - point[0])
            draws.append(draws[1] - draws[0])
        out[name] = [(pt, np.percentile(d, 2.5, axis=0), np.percentile(d, 97.5, axis=0))
                     for pt, d in zip(point, draws)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+", help="one or two run TAGs (B - A is reported for two)")
    ap.add_argument("--n", type=int, default=1000, help="bootstrap resamples")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--json", default=None, help="write the intervals here")
    args = ap.parse_args()
    if len(args.tags) > 2:
        raise SystemExit("give one or two TAGs")
    runs = [load(t, args.data_dir) for t in args.tags]
    rows = np.unique(np.concatenate([r["row"] for r in runs]))
    if len(runs) == 2 and not np.array_equal(np.unique(runs[0]["row"]), np.unique(runs[1]["row"])):
        raise SystemExit("the two runs scored different test rows: a paired test needs the same rows")
    pair_of_row = np.zeros(len(rows), dtype=np.int64)
    pair_of_row[np.searchsorted(rows, runs[0]["row"])] = runs[0]["pair"]
    res = bootstrap([row_counts(r, rows) for r in runs], pair_of_row, args.n)

    labels = list(args.tags) + ([f"{args.tags[1]} - {args.tags[0]}"] if len(runs) == 2 else [])
    table = []
    for scope, per_run in res.items():
        for label, (pt, lo, hi) in zip(labels, per_run):
            table.append(dict(scope=scope, run=label, **{
                m: f"{pt[i]:+.3f} [{lo[i]:+.3f}, {hi[i]:+.3f}]" if " - " in label
                else f"{pt[i]:.3f} [{lo[i]:.3f}, {hi[i]:.3f}]" for i, m in enumerate(METRICS)}))
    print(fmt_table(table, ["scope", "run", *METRICS],
                    f"Row-level bootstrap, {args.n} resamples, 95% intervals"))
    if args.json:
        json.dump({scope: {label: dict(point=pt.tolist(), lo=lo.tolist(), hi=hi.tolist())
                           for label, (pt, lo, hi) in zip(labels, per_run)}
                   for scope, per_run in res.items()} | {"_metrics": list(METRICS)},
                  open(args.json, "w"), indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
