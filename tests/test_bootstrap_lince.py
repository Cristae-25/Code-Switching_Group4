import numpy as np

from scripts.analysis.bootstrap_lince import f1, paired


def test_f1_from_summed_counts():
    c = np.array([[2, 1, 1], [0, 0, 0], [1, 0, 1]])        # TP, FP, FN per tweet
    assert abs(f1(c.sum(0)) - 2 * 3 / (2 * 3 + 1 + 2)) < 1e-12
    assert f1(np.array([0, 0, 0])) == 0.0


def test_paired_identical_models_have_zero_difference():
    rng = np.random.default_rng(0)
    c = rng.integers(0, 4, size=(200, 3)).astype(float)
    r = paired(c, c.copy(), n_boot=200, seed=0)
    assert r["diff"] == 0.0 and r["diff_ci"] == [0.0, 0.0]
    assert r["ci_a"][0] <= r["f1_a"] <= r["ci_a"][1]


def test_paired_better_model_has_positive_interval():
    a = np.tile([[1.0, 1.0, 1.0]], (300, 1))
    b = np.tile([[2.0, 0.0, 0.0]], (300, 1))
    r = paired(a, b, n_boot=200, seed=0)
    assert r["diff"] > 0 and r["diff_ci"][0] > 0
