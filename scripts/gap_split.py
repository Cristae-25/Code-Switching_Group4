#!/usr/bin/env python
"""Split a decide_at=gap run's gain into moved and unmoved positions (no GPU).

    ./.venv/bin/python scripts/gap_split.py xlmr_prev xlmr_prev_gap

WORD_TAG is a ``--decide-at word`` run and GAP_TAG the same settings with
``--decide-at gap``. Both read ``data/preds_<TAG>.npz`` (or its unzipped folder)
and the threshold from ``data/anticipation_<TAG>.json``.

A position is *unmoved* when its (row, position) key is scored in both runs:
there the next word follows with no punctuation in between, and both models
are scored on the same label. A *moved* label sits on the word in the word
run and on the punctuation after it in the gap run. Average precision is
threshold-free. The rule "switch iff the label sits on punctuation" is the
simplest baseline at the gap decision points.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def load(tag: str) -> dict:
    """``data/preds_<TAG>.npz``, or the folder it unzips to."""
    path = f"data/preds_{tag}"
    if os.path.isdir(path):
        return {f[:-4]: np.load(os.path.join(path, f)) for f in os.listdir(path) if f.endswith(".npy")}
    return dict(np.load(path + ".npz"))


def prf(y, p):
    tp = int(((p == 1) & (y == 1)).sum())
    P = tp / max(int((p == 1).sum()), 1)
    R = tp / max(int((y == 1).sum()), 1)
    return P, R, 2 * P * R / max(P + R, 1e-12)


def avg_precision(y, s):
    o = np.argsort(-s, kind="stable")
    y = y[o]
    tp = np.cumsum(y)
    return float((tp / np.arange(1, len(y) + 1))[y == 1].mean())


def run(tag: str) -> dict:
    z = load(tag)
    tau = json.load(open(f"data/anticipation_{tag}.json"))["tau"]
    prob = 1 / (1 + np.exp(-z["test_logit"].astype(np.float64)))
    key = z["test_row"].astype(np.int64) * 1_000_000 + z["test_pos"].astype(np.int64)
    return dict(y=z["test_y"].astype(int), prob=prob, pred=(prob > tau).astype(int), key=key, tau=tau)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("word_tag")
    ap.add_argument("gap_tag")
    a = ap.parse_args()
    W, G = run(a.word_tag), run(a.gap_tag)
    assert len(np.unique(W["key"])) == len(W["key"]) and len(np.unique(G["key"])) == len(G["key"])
    common = np.intersect1d(W["key"], G["key"])
    mw, mg = ~np.isin(W["key"], common), ~np.isin(G["key"], common)
    assert mw.sum() == mg.sum(), "moved sources and landings must pair up"
    print(f"tau word {W['tau']:.3f}, gap {G['tau']:.3f}; moved {mg.sum():,} of {len(mg):,} ({mg.mean():.1%})")
    print(f"{'positions':10s} {'n':>10s} {'base':>6s}  {'word P/R/F1':>20s}  {'gap P/R/F1':>20s}  {'AP word':>7s} {'AP gap':>7s}")
    for name, sw, sg in (("unmoved", ~mw, ~mg), ("moved", mw, mg),
                         ("all", np.ones_like(mw), np.ones_like(mg))):
        pw, pg = prf(W["y"][sw], W["pred"][sw]), prf(G["y"][sg], G["pred"][sg])
        print(f"{name:10s} {int(sw.sum()):>10,} {W['y'][sw].mean():>6.3f}  "
              f"{'/'.join(f'{v:.3f}' for v in pw):>20s}  {'/'.join(f'{v:.3f}' for v in pg):>20s}  "
              f"{avg_precision(W['y'][sw], W['prob'][sw]):>7.3f} {avg_precision(G['y'][sg], G['prob'][sg]):>7.3f}")
    P, R, F = prf(G["y"], mg.astype(int))
    share = (G["y"][mg] == 1).sum() / max((G["y"] == 1).sum(), 1)
    print(f"switches behind punctuation: {share:.1%}")
    print(f'rule "switch iff the label sits on punctuation": P {P:.3f} R {R:.3f} F1 {F:.3f}')


if __name__ == "__main__":
    main()
