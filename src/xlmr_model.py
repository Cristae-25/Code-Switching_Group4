"""Pretrained encoders with a causal mask: XLM-R, RoBERTa and mBERT.

These encoders were pretrained bidirectionally. ``is_decoder=True`` makes
Hugging Face apply a lower-triangular (causal) mask in every self-attention
layer, so the representation at piece t is built from pieces 1..t only. We
measured this (earlier positions move by 0.00 when later pieces change), and
``check_causality`` re-verifies it on the assembled model before any training.

| key       | checkpoint                     | pretrained on             | tokenizer          |
|-----------|--------------------------------|---------------------------|--------------------|
| "xlmr"    | xlm-roberta-base               | 100 languages (CC-100)    | SentencePiece 250K |
| "roberta" | roberta-base                   | English only              | byte-level BPE 50K |
| "mbert"   | bert-base-multilingual-cased   | 104 languages (Wikipedia) | WordPiece 119K     |

The interface matches ``CausalSwitchModel``: ``forward(word_ids, lang_ids=None,
pair_id=None) -> (switch logits [b, t], duration logits [b, t, n_dur])``. The
language and pair inputs are accepted and ignored: on sub-word pieces the model
reads the text only. Inputs are the backbone's raw piece ids, with no special
tokens. Windows are right-padded, and under a causal mask a real position
never attends to padding after it, so no attention mask is needed.

``context=1`` is the current-token-only model, the pretrained counterpart of
the ``naive_l2`` baseline. Every piece is encoded as its own length-1
sequence, so the prediction at t reads piece t and nothing else (not even the
past), and every piece gets the same position. In eval mode the model is then a
function of the piece id alone.
"""
from __future__ import annotations

import torch.nn as nn

from .anticipation import N_DUR
from .causal_model import make_dur_head

BACKBONES = {"xlmr": "xlm-roberta-base", "roberta": "roberta-base",
             "mbert": "bert-base-multilingual-cased"}


class CausalPretrained(nn.Module):
    def __init__(self, name: str = "xlm-roberta-base", config=None, pretrained: bool = True,
                 n_dur: int = N_DUR, dropout: float = 0.1, use_lang: bool = False,
                 context: int | None = None, dur_head: str = "softmax"):
        super().__init__()
        self.dur_ordinal = dur_head == "ordinal"
        if context not in (None, 1):
            raise ValueError("context is None (the whole prefix) or 1 (the current piece only)")
        if context == 1 and use_lang:
            raise ValueError("a current-token model takes no language input: it would be context")
        self.context = context
        from transformers import AutoConfig, AutoModel

        causal = dict(is_decoder=True, add_cross_attention=False)
        if config is not None:                   # explicit config: random init (tests)
            for k, v in causal.items():
                setattr(config, k, v)
            self.encoder = AutoModel.from_config(config, add_pooling_layer=False)
        elif pretrained:                         # fine-tuning: the pretrained weights
            self.encoder = AutoModel.from_pretrained(name, add_pooling_layer=False, **causal)
        else:                                    # architecture only, to load a checkpoint into
            cfg = AutoConfig.from_pretrained(name, **causal)
            self.encoder = AutoModel.from_config(cfg, add_pooling_layer=False)
        assert self.encoder.config.is_decoder, "the encoder must run with a causal mask"
        self.vocab_size = self.encoder.config.vocab_size
        d = self.encoder.config.hidden_size
        # Optional language input (--prev-lang): a learned embedding for the language of
        # the last finished word (slots L1, L2, none), added to the piece embeddings.
        # Initialised at zero, so training starts from the pretrained model unchanged.
        self.lang_emb = nn.Embedding(3, d) if use_lang else None
        if self.lang_emb is not None:
            nn.init.zeros_(self.lang_emb.weight)
        self.drop = nn.Dropout(dropout)
        self.head_sw = nn.Linear(d, 1)
        self.head_dur = make_dur_head(d, n_dur, dur_head)

    def forward(self, word_ids, lang_ids=None, pair_id=None):
        if self.context == 1:                    # [b, t] -> [b*t, 1]: one piece per sequence
            b, t = word_ids.shape
            sw, dur = self._encode(word_ids.reshape(b * t, 1), None)
            return sw.reshape(b, t), dur.reshape(b, t, -1)
        return self._encode(word_ids, lang_ids)

    def _encode(self, word_ids, lang_ids):
        # use_cache=False: a decoder-mode encoder would otherwise build a key/value
        # cache on every call, which only costs memory here.
        if self.lang_emb is None:
            h = self.encoder(input_ids=word_ids, use_cache=False).last_hidden_state
        else:
            emb = self.encoder.get_input_embeddings()(word_ids) + self.lang_emb(lang_ids)
            h = self.encoder(inputs_embeds=emb, use_cache=False).last_hidden_state
        h = self.drop(h)
        return self.head_sw(h).squeeze(-1), self.head_dur(h)


class CausalXLMR(CausalPretrained):
    """The XLM-R instance (kept for existing checkpoints and tests)."""

    def __init__(self, name: str = "xlm-roberta-base", **kw):
        super().__init__(name, **kw)
