"""Build either anticipation model from a run's settings (training or a checkpoint's args)."""
from __future__ import annotations

from .anticipation import N_DUR
from .causal_model import CausalSwitchModel


def build_model(cfg: dict, vocab_size: int, n_pairs: int = 14, use_pair: bool = False,
                use_lang: bool = True, n_dur: int = N_DUR, pretrained: bool = True,
                xlmr_config=None):
    """``cfg["model"]`` is "scratch" (the from-scratch transformer) or "xlmr"."""
    if cfg.get("model", "scratch") == "xlmr":
        from .xlmr_model import CausalXLMR
        return CausalXLMR(cfg.get("xlmr_name", "xlm-roberta-base"), config=xlmr_config,
                          pretrained=pretrained, n_dur=n_dur)
    return CausalSwitchModel(vocab_size, d_model=cfg["d_model"], n_layers=cfg["layers"],
                             n_heads=cfg["heads"], d_ff=cfg["d_model"] * 2,
                             max_len=cfg["window"], n_pairs=n_pairs, use_pair=use_pair,
                             use_lang=use_lang, n_dur=n_dur)
