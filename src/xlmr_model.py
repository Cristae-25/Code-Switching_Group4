"""XLM-R with a causal mask, for anticipatory switch and duration prediction.

XLM-R was pretrained bidirectionally. ``is_decoder=True`` makes Hugging Face
apply a lower-triangular (causal) mask in every self-attention layer, so the
representation at piece t is built from pieces 1..t only. We measured this
(earlier positions move by 0.00 when later pieces change), and
``check_causality`` re-verifies it on the assembled model before any training.

The interface matches ``CausalSwitchModel``: ``forward(word_ids, lang_ids=None,
pair_id=None) -> (switch logits [b, t], duration logits [b, t, n_dur])``. The
language and pair inputs are accepted and ignored: on sub-word pieces the model
reads the text only.

Inputs are raw XLM-R piece ids, with no ``<s>`` token. Windows are right-padded,
and under a causal mask a real position never attends to padding after it, so
no attention mask is needed.
"""
from __future__ import annotations

import torch.nn as nn

from .anticipation import N_DUR


class CausalXLMR(nn.Module):
    def __init__(self, name: str = "xlm-roberta-base", config=None, pretrained: bool = True,
                 n_dur: int = N_DUR, dropout: float = 0.1):
        super().__init__()
        from transformers import XLMRobertaConfig, XLMRobertaModel

        causal = dict(is_decoder=True, add_cross_attention=False)
        if config is not None:                   # explicit config: random init (tests)
            for k, v in causal.items():
                setattr(config, k, v)
            self.encoder = XLMRobertaModel(config, add_pooling_layer=False)
        elif pretrained:                         # fine-tuning: the pretrained weights
            self.encoder = XLMRobertaModel.from_pretrained(name, add_pooling_layer=False, **causal)
        else:                                    # architecture only, to load a checkpoint into
            cfg = XLMRobertaConfig.from_pretrained(name, **causal)
            self.encoder = XLMRobertaModel(cfg, add_pooling_layer=False)
        assert self.encoder.config.is_decoder, "XLM-R must run with a causal mask"
        d = self.encoder.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.head_sw = nn.Linear(d, 1)
        self.head_dur = nn.Linear(d, n_dur)

    def forward(self, word_ids, lang_ids=None, pair_id=None):
        # use_cache=False: a decoder-mode encoder would otherwise build a key/value
        # cache on every call, which only costs memory here.
        h = self.encoder(input_ids=word_ids, use_cache=False).last_hidden_state
        h = self.drop(h)
        return self.head_sw(h).squeeze(-1), self.head_dur(h)
