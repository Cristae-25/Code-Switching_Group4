"""Row-level bootstrap on saved predictions."""
import importlib.util
import os

import numpy as np
import pytest

from src.metrics import duration_scores, end_to_end_duration, prf


def _mod():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "bootstrap_preds", os.path.join(root, "scripts", "analysis", "bootstrap_preds.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _preds(seed=0, n=400, rows=40):
    rng = np.random.default_rng(seed)
    row = rng.integers(0, rows, n)
    y = (rng.random(n) < 0.2).astype(int)
    pred = (rng.random(n) < 0.25).astype(int)
    k = 60
    return dict(row=row, pair=(row % 2) + 1, y=y, pred=pred,
                drow=rng.integers(0, rows, k), dtrue=rng.integers(0, 3, k),
                dpred=rng.integers(0, 3, k), fired=rng.random(k) < 0.7)


def test_point_estimates_match_the_report_metrics():
    mod, p = _mod(), _preds()
    rows = np.unique(np.r_[p["row"], p["drow"]])
    got = mod.metrics(mod.row_counts(p, rows).sum(0))
    d = duration_scores(p["dtrue"], p["dpred"])
    e = end_to_end_duration(p["dtrue"], p["dpred"], p["fired"])
    assert got[0] == pytest.approx(prf(p["y"], p["pred"])["f1"])
    assert got[1] == pytest.approx(d["macro_f1"]) and got[2] == pytest.approx(d["per_class_f1"][1])
    assert got[3] == pytest.approx(e["accuracy"])


def test_a_run_against_itself_differs_by_exactly_zero():
    mod, p = _mod(), _preds()
    rows = np.unique(np.r_[p["row"], p["drow"]])
    st = mod.row_counts(p, rows)
    res = mod.bootstrap([st, st], (rows % 2) + 1, n=50)
    for scope in res.values():
        pt, lo, hi = scope[2]
        assert np.allclose(pt, 0) and np.allclose(lo, 0) and np.allclose(hi, 0)
    pt, lo, hi = res["overall"][0]
    assert np.all(lo <= pt + 1e-12) and np.all(pt <= hi + 1e-12)
