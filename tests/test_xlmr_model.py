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


def test_pretrained_models_can_take_a_language_input():
    m = CausalPretrained(config=_tiny_cfg("roberta"), use_lang=True).eval()
    assert m.lang_emb.weight.abs().sum() == 0            # starts as the unchanged pretrained model
    torch.nn.init.normal_(m.lang_emb.weight)             # ... then check the input is wired in
    x = torch.randint(5, 300, (1, 10))
    a, _ = m(x, torch.zeros(1, 10, dtype=torch.long))
    b, _ = m(x, torch.ones(1, 10, dtype=torch.long))
    assert not torch.equal(a, b)                                  # the input is used
    assert check_causality(m, 300, seq_len=24)["max_abs_logit_drift"] == 0.0


# --- current-token only (context=1): the pretrained counterpart of naive_l2 -----

def test_current_token_model_reads_no_other_position():
    """Stronger than check_causality: past *and* future positions are rewritten."""
    m = CausalPretrained(config=_tiny_cfg("roberta"), context=1).eval()
    g = torch.Generator().manual_seed(0)
    x = torch.randint(5, 300, (1, 16), generator=g)
    sw_a, dur_a = m(x)
    for t in range(16):
        y = torch.randint(5, 300, (1, 16), generator=g)
        y[0, t] = x[0, t]
        sw_b, dur_b = m(y)
        assert sw_a[0, t] == sw_b[0, t] and torch.equal(dur_a[0, t], dur_b[0, t])


def test_current_token_model_is_a_function_of_the_piece_id():
    m = CausalPretrained(config=_tiny_cfg("roberta"), context=1).eval()
    x = torch.tensor([[7, 9, 7, 11, 7, 9]])
    sw, _ = m(x)
    assert sw[0, 0] == sw[0, 2] == sw[0, 4] and sw[0, 1] == sw[0, 5]


def test_current_token_model_keeps_the_output_shapes():
    m = CausalPretrained(config=_tiny_cfg("roberta"), context=1).eval()
    sw, dur = m(torch.randint(5, 300, (3, 10)))
    assert sw.shape == (3, 10) and dur.shape == (3, 10, 3)
    assert check_causality(m, 300, seq_len=24)["max_abs_logit_drift"] == 0.0


def test_the_full_prefix_model_does_read_the_past():
    """The context-free check must fail without context=1, or it proves nothing."""
    m = CausalPretrained(config=_tiny_cfg("roberta")).eval()
    a, _ = m(torch.tensor([[7, 9, 11, 13]]))
    b, _ = m(torch.tensor([[8, 9, 11, 13]]))
    assert a[0, 3] != b[0, 3]


def test_current_token_model_rejects_a_language_input():
    with pytest.raises(ValueError):
        CausalPretrained(config=_tiny_cfg("roberta"), context=1, use_lang=True)


def test_build_model_passes_the_context():
    m = build_model(dict(model="xlmr", context=1), 300, xlmr_config=_tiny())
    assert m.context == 1
    assert build_model(dict(model="xlmr"), 300, xlmr_config=_tiny()).context is None


def test_the_context_free_check_passes_ctx1_and_fails_the_prefix_model():
    from src.causal_model import check_context_free
    ok = check_context_free(CausalPretrained(config=_tiny_cfg("roberta"), context=1), 300)
    assert ok["context_free"] and ok["max_abs_logit_drift"] == 0.0
    assert not check_context_free(CausalPretrained(config=_tiny_cfg("roberta")), 300)["context_free"]


# --- --freeze-backbone: train the heads only ---------------------------------

def _train_script():
    import importlib.util, os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "train_anticipation", os.path.join(root, "scripts", "training", "train_anticipation.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _probe_args(**kw):
    import argparse
    return argparse.Namespace(**{**dict(model="xlmr", lr=2e-5, head_lr=1e-3, freeze_backbone=False), **kw})


def test_unfrozen_pretrained_trains_body_and_heads_at_two_rates():
    m = CausalPretrained(config=_tiny_cfg("roberta"))
    groups, lrs = _train_script().optimizer_groups(m, _probe_args())
    assert lrs == [2e-5, 1e-3]
    assert sum(len(g["params"]) for g in groups) == len(list(m.parameters()))


def test_frozen_backbone_optimises_the_heads_only():
    m = CausalPretrained(config=_tiny_cfg("roberta"))
    groups, lrs = _train_script().optimizer_groups(m, _probe_args(freeze_backbone=True))
    heads = {id(p) for n, p in m.named_parameters() if n.startswith("head_")}
    assert lrs == [1e-3] and {id(p) for p in groups[0]["params"]} == heads
    assert all(not p.requires_grad for n, p in m.named_parameters() if not n.startswith("head_"))


def test_a_frozen_step_leaves_the_backbone_unchanged():
    m = CausalPretrained(config=_tiny_cfg("roberta"))
    groups, _ = _train_script().optimizer_groups(m, _probe_args(freeze_backbone=True))
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    opt = torch.optim.AdamW(groups)
    sw, dur = m(torch.randint(0, 300, (2, 16)))
    (sw.sum() + dur.sum()).backward()
    opt.step()
    for n, p in m.named_parameters():
        assert torch.equal(p, before[n]) != n.startswith("head_"), n


def test_freezing_needs_a_pretrained_model():
    with pytest.raises(SystemExit):
        _train_script().optimizer_groups(torch.nn.Linear(2, 2),
                                         _probe_args(model="scratch", freeze_backbone=True))
