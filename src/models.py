"""Build any anticipation model from a run's settings (training or a checkpoint's args)."""
from __future__ import annotations

from .anticipation import N_DUR
from .causal_model import CausalSwitchModel
from .evaluation import PAIR_KEYS

PRETRAINED = ("xlmr", "roberta", "mbert")


def build_model(cfg: dict, vocab_size: int, n_pairs: int = 14, use_pair: bool = False,
                use_lang: bool = True, n_dur: int = N_DUR, pretrained: bool = True,
                xlmr_config=None):
    """``cfg["model"]`` is "scratch" (the from-scratch transformer), or a pretrained
    backbone: "xlmr", "roberta" or "mbert" (see ``xlmr_model.BACKBONES``)."""
    model = cfg.get("model", "scratch")
    if model in PRETRAINED:
        from .xlmr_model import BACKBONES, CausalPretrained, CausalXLMR
        name = cfg.get("backbone") or (cfg.get("xlmr_name") if model == "xlmr" else None) \
            or BACKBONES[model]
        cls = CausalXLMR if model == "xlmr" else CausalPretrained
        return cls(name, config=xlmr_config, pretrained=pretrained, n_dur=n_dur,
                   use_lang=cfg.get("prev_lang", False), context=cfg.get("context"),
                   dur_head=cfg.get("dur_head", "softmax"),
                   use_pair=bool(cfg.get("pair_emb", False)), n_pairs=len(PAIR_KEYS),
                   lora_r=int(cfg.get("lora_r") or 0), lora_alpha=cfg.get("lora_alpha"))
    return CausalSwitchModel(vocab_size, d_model=cfg["d_model"], n_layers=cfg["layers"],
                             n_heads=cfg["heads"], d_ff=cfg["d_model"] * 2,
                             max_len=cfg["window"], n_pairs=n_pairs, use_pair=use_pair,
                             use_lang=use_lang, n_dur=n_dur,
                             dur_head=cfg.get("dur_head", "softmax"))


def pair_ids(model, batch):
    """The pair input a model expects from a loader batch.

    The from-scratch model's ``pair_id`` indexes first languages (14), so
    German->English and German->French share one id. A pretrained model with
    ``--pair-emb`` reads ``report_pair``, the full 15-pair index (``PAIR_KEYS``).
    """
    return batch[getattr(model, "pair_key", "pair_id")]


def load_checkpoint(path: str, map_location="cpu", fallback_args: dict | None = None):
    """``(model in eval mode, checkpoint dict)`` from a ``train_anticipation.py --ckpt`` file.

    The architecture is rebuilt from the checkpoint's own args (a pretrained backbone's
    config only, no weight download), then the trained weights are loaded.
    """
    import torch
    ck = torch.load(path, map_location=map_location, weights_only=False)
    cfg = {**(fallback_args or {}), **ck.get("args", {})}
    model = build_model(cfg, ck.get("vocab_size"), use_lang=ck.get("use_lang", True),
                        n_dur=int(cfg.get("dur_classes") or N_DUR), pretrained=False)
    model.load_state_dict(ck["state"])
    model.eval()
    ck["args"] = cfg
    return model, ck
