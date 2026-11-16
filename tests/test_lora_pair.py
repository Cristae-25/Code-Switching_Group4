"""LoRA adapters and the language-pair embedding on the pretrained encoders."""
import argparse

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from src.causal_model import check_causality  # noqa: E402
from src.evaluation import PAIR_KEYS  # noqa: E402
from src.lora import LoRALinear, is_lora_param  # noqa: E402
from src.models import build_model, pair_ids  # noqa: E402
from src.xlmr_model import CausalXLMR  # noqa: E402


def _tiny():
    return transformers.XLMRobertaConfig(vocab_size=300, hidden_size=32, num_hidden_layers=2,
                                         num_attention_heads=2, intermediate_size=64,
                                         max_position_embeddings=80)


def _same_weights(src, dst):
    """Copy ``src``'s pretrained weights into ``dst`` (whose linears may be wrapped)."""
    sd = {k.replace(".base.", "."): v for k, v in dst.state_dict().items()}
    for k, v in src.state_dict().items():
        assert k in sd, k
    dst.load_state_dict({k: src.state_dict()[k.replace(".base.", ".")] if not is_lora_param(k)
                         else v for k, v in dst.state_dict().items()})


def test_lora_starts_as_the_pretrained_model():
    torch.manual_seed(0)
    plain = CausalXLMR(config=_tiny()).eval()
    adapted = CausalXLMR(config=_tiny(), lora_r=4).eval()
    _same_weights(plain, adapted)
    x = torch.randint(5, 300, (2, 12))
    assert torch.allclose(plain(x)[0], adapted(x)[0], atol=1e-6)


def test_lora_wraps_every_encoder_linear_and_freezes_the_rest():
    m = CausalXLMR(config=_tiny(), lora_r=4, use_pair=True, n_pairs=len(PAIR_KEYS))
    n_lora = sum(isinstance(x, LoRALinear) for x in m.encoder.modules())
    assert n_lora == 2 * 6                                   # q, k, v, attn-out, ffn-in, ffn-out per layer
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert all(is_lora_param(n) or n.startswith(("head_", "pair_emb")) for n in trainable)
    assert any(is_lora_param(n) for n in trainable) and "pair_emb.weight" in trainable
    assert not any(p.requires_grad for n, p in m.encoder.named_parameters() if not is_lora_param(n))


def test_lora_and_pair_models_stay_causal():
    m = CausalXLMR(config=_tiny(), lora_r=4, use_lang=True, use_pair=True, n_pairs=len(PAIR_KEYS))
    for p in m.parameters():                                 # non-zero adapters and embeddings
        torch.nn.init.normal_(p, std=0.02) if p.requires_grad else None
    assert check_causality(m, 300, seq_len=24)["max_abs_logit_drift"] == 0.0


def test_pair_embedding_is_a_no_op_until_trained_then_separates_pairs():
    m = CausalXLMR(config=_tiny(), use_pair=True, n_pairs=len(PAIR_KEYS)).eval()
    x = torch.randint(5, 300, (2, 10))
    a = m(x, None, torch.tensor([4, 5]))[0]
    assert torch.allclose(a, m(x)[0], atol=1e-6)             # zero init: step 0 = no pair input
    with torch.no_grad():
        m.pair_emb.weight[5].normal_()
    b = m(x, None, torch.tensor([4, 5]))[0]
    assert torch.allclose(a[0], b[0], atol=1e-6) and not torch.allclose(a[1], b[1])


def test_pretrained_pair_input_distinguishes_german_english_from_german_french():
    m = build_model(dict(model="xlmr", pair_emb=True), 300, xlmr_config=_tiny())
    assert m.pair_emb.num_embeddings == len(PAIR_KEYS)
    batch = {"pair_id": torch.tensor([3, 3]),
             "report_pair": torch.tensor([PAIR_KEYS.index("German->English"),
                                          PAIR_KEYS.index("German->French")])}
    assert pair_ids(m, batch).tolist() == batch["report_pair"].tolist()
    plain = build_model(dict(model="xlmr"), 300, xlmr_config=_tiny())
    assert pair_ids(plain, batch) is batch["pair_id"]


def test_lora_checkpoint_round_trips_through_build_model():
    cfg = dict(model="xlmr", lora_r=4, pair_emb=True)
    a = build_model(cfg, 300, xlmr_config=_tiny())
    b = build_model(cfg, 300, xlmr_config=_tiny())
    b.load_state_dict(a.state_dict())                        # same keys, same shapes


def test_optimizer_trains_only_adapters_and_heads_under_lora():
    import importlib.util, pathlib
    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "training" / "train_anticipation.py"
    spec = importlib.util.spec_from_file_location("train_anticipation", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    m = build_model(dict(model="xlmr", lora_r=4), 300, xlmr_config=_tiny())
    args = argparse.Namespace(model="xlmr", freeze_backbone=False, lora_r=4, lr=2e-4, head_lr=1e-3)
    groups, lrs = mod.optimizer_groups(m, args)
    assert lrs == [2e-4, 1e-3]
    n_trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert sum(p.numel() for g in groups for p in g["params"]) == n_trainable
    with pytest.raises(SystemExit):
        mod.optimizer_groups(m, argparse.Namespace(**{**vars(args), "freeze_backbone": True}))
