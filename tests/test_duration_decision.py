"""Ordered duration head and the val-tuned duration decision."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.causal_model import (CausalSwitchModel, OrdinalHead, check_causality,  # noqa: E402
                              duration_probs, ordinal_targets)
from src.metrics import apply_class_offsets, duration_scores, tune_class_offsets  # noqa: E402
from src.models import build_model  # noqa: E402


def test_ordinal_cuts_stay_ordered_so_probabilities_are_valid():
    head = OrdinalHead(8)
    with torch.no_grad():
        head.gaps.fill_(-20.0)                      # even a collapsed gap keeps the order
    cum = head(torch.randn(50, 8))
    assert cum.shape == (50, 2) and torch.all(cum[:, 0] >= cum[:, 1])
    p = duration_probs(cum, ordinal=True)
    assert p.shape == (50, 3) and torch.all(p >= 0)
    assert torch.allclose(p.sum(-1), torch.ones(50))


def test_ordinal_targets_are_cumulative():
    assert ordinal_targets(torch.tensor([0, 1, 2])).tolist() == [[0, 0], [1, 0], [1, 1]]


def test_ordinal_probabilities_read_the_cuts():
    # P(>Small) = 0.8, P(>Medium) = 0.3 -> Small 0.2, Medium 0.5, Large 0.3
    cum = torch.logit(torch.tensor([[0.8, 0.3]]))
    assert duration_probs(cum, ordinal=True)[0].tolist() == pytest.approx([0.2, 0.5, 0.3])
    assert duration_probs(torch.zeros(1, 3), ordinal=False)[0].tolist() == pytest.approx([1 / 3] * 3)


def test_ordinal_models_stay_causal_and_rebuild_from_their_args():
    m = CausalSwitchModel(200, d_model=32, n_layers=2, n_heads=2, d_ff=64, max_len=64,
                          use_lang=False, dur_head="ordinal")
    assert m.dur_ordinal
    _, dur = m(torch.randint(3, 200, (2, 10)))
    assert dur.shape == (2, 10, 2)
    assert check_causality(m, 200, seq_len=16)["causal"]
    cfg = dict(model="scratch", d_model=32, layers=2, heads=2, window=64, dur_head="ordinal")
    again = build_model(cfg, 200, use_lang=False)
    again.load_state_dict(m.state_dict())             # same shapes: the checkpoint loads


def test_xlmr_takes_the_ordinal_head():
    transformers = pytest.importorskip("transformers")
    from src.xlmr_model import CausalXLMR
    cfg = transformers.XLMRobertaConfig(vocab_size=300, hidden_size=32, num_hidden_layers=2,
                                        num_attention_heads=2, intermediate_size=64,
                                        max_position_embeddings=80)
    m = CausalXLMR(config=cfg, dur_head="ordinal").eval()
    _, dur = m(torch.randint(5, 300, (1, 12)))
    assert dur.shape == (1, 12, 2) and m.dur_ordinal
    assert check_causality(m, 300, seq_len=16)["max_abs_logit_drift"] == 0.0


def _skewed(n=3000, seed=0):
    """A model that under-calls the middle class: Medium is often scored as Large."""
    rng = np.random.default_rng(seed)
    y = rng.choice(3, n, p=[0.25, 0.25, 0.5])
    prob = np.full((n, 3), 0.1)
    prob[np.arange(n), y] = 0.6
    med = y == 1
    prob[med, 2] = 0.65                               # Medium loses to Large at argmax
    return prob / prob.sum(1, keepdims=True), y


def test_tuned_offsets_recover_the_middle_class():
    prob, y = _skewed()
    before = duration_scores(y, prob.argmax(1))
    off, val_f1 = tune_class_offsets(prob, y)
    after = duration_scores(y, apply_class_offsets(prob, off))
    assert off[0] == 0.0 and off[1] > off[2]
    assert after["per_class_f1"][1] > before["per_class_f1"][1]
    assert after["macro_f1"] == pytest.approx(val_f1) and val_f1 > before["macro_f1"]


def test_zero_offsets_are_the_argmax_and_tuning_never_loses_on_val():
    rng = np.random.default_rng(1)
    prob = rng.dirichlet(np.ones(3), 500)
    y = rng.choice(3, 500)
    assert (apply_class_offsets(prob, np.zeros(3)) == prob.argmax(1)).all()
    _, f = tune_class_offsets(prob, y)
    assert f >= duration_scores(y, prob.argmax(1))["macro_f1"] - 1e-12
