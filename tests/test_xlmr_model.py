"""XLM-R with a causal mask: same interface as the from-scratch model."""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from src.causal_model import CausalSwitchModel, check_causality  # noqa: E402
from src.models import build_model  # noqa: E402
from src.xlmr_model import CausalXLMR  # noqa: E402


def _tiny():
    return transformers.XLMRobertaConfig(vocab_size=300, hidden_size=32, num_hidden_layers=2,
                                         num_attention_heads=2, intermediate_size=64,
                                         max_position_embeddings=80)


def test_outputs_have_the_switch_and_three_class_duration_heads():
    m = CausalXLMR(config=_tiny()).eval()
    sw, dur = m(torch.randint(5, 300, (2, 12)))
    assert sw.shape == (2, 12) and dur.shape == (2, 12, 3)


def test_language_ids_are_ignored():
    m = CausalXLMR(config=_tiny()).eval()
    x = torch.randint(5, 300, (1, 10))
    a, _ = m(x, torch.zeros(1, 10, dtype=torch.long))
    b, _ = m(x, torch.ones(1, 10, dtype=torch.long))
    assert torch.equal(a, b)


def test_xlmr_cannot_see_the_future():
    rep = check_causality(CausalXLMR(config=_tiny()), 300, seq_len=24)
    assert rep["max_abs_logit_drift"] == 0.0


def test_the_check_catches_a_bidirectional_xlmr():
    m = CausalXLMR(config=_tiny())
    m.encoder.config.is_decoder = False
    for layer in m.encoder.encoder.layer:
        layer.attention.self.is_causal = False
    assert not check_causality(m, 300, seq_len=24)["causal"]


def test_build_model_picks_the_architecture():
    scratch = build_model(dict(model="scratch", d_model=32, layers=2, heads=2, window=16), 100)
    assert isinstance(scratch, CausalSwitchModel)
    x = build_model(dict(model="xlmr"), 300, xlmr_config=_tiny())
    assert isinstance(x, CausalXLMR)


# --- the same wrapper around RoBERTa and mBERT --------------------------------

from src.xlmr_model import BACKBONES, CausalPretrained  # noqa: E402


def _tiny_cfg(kind):
    kw = dict(vocab_size=300, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
              intermediate_size=64, max_position_embeddings=80)
    return transformers.BertConfig(**kw) if kind == "bert" else transformers.RobertaConfig(**kw)


def test_backbones_name_the_three_pretrained_models():
    assert BACKBONES == {"xlmr": "xlm-roberta-base", "roberta": "roberta-base",
                         "mbert": "bert-base-multilingual-cased"}


@pytest.mark.parametrize("kind", ["bert", "roberta"])
def test_other_backbones_are_causal(kind):
    m = CausalPretrained(config=_tiny_cfg(kind))
    assert check_causality(m, 300, seq_len=24)["max_abs_logit_drift"] == 0.0
    sw, dur = m.eval()(torch.randint(5, 300, (2, 10)))
    assert sw.shape == (2, 10) and dur.shape == (2, 10, 3)


@pytest.mark.parametrize("kind", ["bert", "roberta"])
def test_the_check_catches_bidirectional_backbones(kind):
    m = CausalPretrained(config=_tiny_cfg(kind))
    m.encoder.config.is_decoder = False
    for layer in m.encoder.encoder.layer:
        layer.attention.self.is_causal = False
    assert not check_causality(m, 300, seq_len=24)["causal"]


def test_build_model_picks_each_backbone():
    for model in ("roberta", "mbert"):
        kind = "bert" if model == "mbert" else "roberta"
        m = build_model(dict(model=model), 300, xlmr_config=_tiny_cfg(kind))
        assert isinstance(m, CausalPretrained)
