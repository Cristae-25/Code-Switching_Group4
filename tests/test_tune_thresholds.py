"""Offline threshold tuning on saved predictions."""
import importlib.util
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _mod():
    spec = importlib.util.spec_from_file_location(
        "tune_thresholds", os.path.join(ROOT, "scripts", "tune_thresholds.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _split(rng, n=4000):
    # pair 0: switches have logit ~ +1; pair 1: the model is shifted, switches at ~ -1
    pair = rng.integers(0, 2, n)
    y = (rng.random(n) < 0.2).astype(int)
    logit = np.where(y == 1, 1.0, -1.0) - 2.0 * pair + rng.normal(0, 0.7, n)
    return dict(logit=logit, y=y, pair=pair, lang=rng.integers(0, 2, n))


def test_per_pair_thresholds_fix_a_miscalibrated_pair():
    rng = np.random.default_rng(0)
    val, test = _split(rng), _split(rng)
    m = _mod()
    glob = m.evaluate_strategy(val, test, "global")
    per = m.evaluate_strategy(val, test, "per_pair")
    assert per["macro_f1"] > glob["macro_f1"] + 0.05     # the shifted pair recovers
    assert set(per["thresholds"]) == {0, 1}


def test_thresholds_are_tuned_on_val_not_test():
    rng = np.random.default_rng(1)
    val, test = _split(rng), _split(rng)
    m = _mod()
    a = m.evaluate_strategy(val, test, "per_pair")
    test2 = dict(test, y=1 - test["y"])                 # flip test labels: thresholds must not move
    b = m.evaluate_strategy(val, test2, "per_pair")
    assert a["thresholds"] == b["thresholds"]
