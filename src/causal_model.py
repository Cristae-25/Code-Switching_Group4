"""A prefix-only (causal) transformer for anticipatory switch prediction.

The whole task is defined by what the model is NOT allowed to see. At position
``t`` it predicts properties of token ``t+1``, so any path by which information
from ``t+1..n`` reaches position ``t`` silently turns the benchmark into a
tagging task with a near-perfect ceiling. Three such paths exist, and all three
are closed here:

1. **Attention.** ``nn.TransformerEncoderLayer`` is bidirectional by default.
   An additive float mask with ``-inf`` above the diagonal, passed with
   ``is_causal=True``, restricts every query at ``t`` to keys ``<= t``. The
   flag alone is only a hint to the fast path - the explicit mask is what
   enforces it, so both are supplied.
2. **Normalisation.** Any layer that pools across time leaks the future.
   ``LayerNorm`` normalises over the feature axis only, which is safe;
   ``BatchNorm1d`` over time would not be, and is not used.
3. **Labels as inputs.** The input at ``t`` carries the language of token
   ``t``, which is in the prefix and therefore legal. It never carries
   ``lang(t+1)``, which IS the label. The two are built from the same array,
   one shifted against the other, so the shift is the only thing standing
   between a legitimate model and a 100% score - ``check_causality`` verifies
   it empirically rather than by inspection. In sub-word mode there is no
   language input at all (``use_lang=False``), because at a mid-word piece the
   word's label depends on pieces not yet read.

``check_causality`` perturbs every token after position ``t`` and asserts the
logits at ``t`` are bit-identical. That is a stronger statement than "we set
the mask": it tests the assembled model, including the heads.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .anticipation import N_DUR


class CausalSwitchModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, n_layers: int = 4,
                 n_heads: int = 4, d_ff: int = 512, max_len: int = 512,
                 dropout: float = 0.1, n_pairs: int = 16, use_pair: bool = False,
                 use_lang: bool = True, n_dur: int = N_DUR):
        super().__init__()
        self.d_model = d_model
        self.use_pair = use_pair
        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        # 2 real slots + 1 pad slot; the language of the CURRENT token only.
        # Sub-word models take no language input (use_lang=False): a word's
        # label is known only once all of its pieces have been read.
        self.lang_emb = nn.Embedding(3, d_model) if use_lang else None
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.pair_emb = nn.Embedding(n_pairs, d_model) if use_pair else None
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers,
                                             norm=nn.LayerNorm(d_model),
                                             enable_nested_tensor=False)
        self.head_sw = nn.Linear(d_model, 1)
        self.head_dur = nn.Linear(d_model, n_dur)

    def forward(self, word_ids, lang_ids=None, pair_id=None):
        b, t = word_ids.shape
        pos = torch.arange(t, device=word_ids.device).unsqueeze(0)
        x = self.word_emb(word_ids) + self.pos_emb(pos)
        if self.lang_emb is not None:
            # lang slot 2 is reserved for padding so pads never look like L1.
            x = x + self.lang_emb(lang_ids)
        if self.pair_emb is not None and pair_id is not None:
            x = x + self.pair_emb(pair_id).unsqueeze(1)
        x = self.drop(x)

        mask = torch.triu(
            torch.full((t, t), float("-inf"), device=word_ids.device), diagonal=1
        )
        h = self.encoder(x, mask=mask, is_causal=True)
        return self.head_sw(h).squeeze(-1), self.head_dur(h)


@torch.no_grad()
def check_causality(model, vocab_size: int, seq_len: int = 32,
                    device: str = "cpu", tol: float = 0.0) -> dict:
    """Empirically verify prefix-only context.

    Builds one sequence, then rewrites every position after ``t`` with random
    tokens and languages. If any logit at or before ``t`` moves, the future is
    reaching the past.
    """
    model.eval().to(device)
    g = torch.Generator(device="cpu").manual_seed(0)
    words = torch.randint(3, vocab_size, (1, seq_len), generator=g).to(device)
    langs = torch.randint(0, 2, (1, seq_len), generator=g).to(device)
    sw_a, dur_a = model(words, langs)

    worst = 0.0
    for t in range(seq_len - 1):
        w2, l2 = words.clone(), langs.clone()
        w2[0, t + 1:] = torch.randint(3, vocab_size, (seq_len - t - 1,), generator=g)
        l2[0, t + 1:] = torch.randint(0, 2, (seq_len - t - 1,), generator=g)
        sw_b, dur_b = model(w2, l2)
        worst = max(worst,
                    float((sw_a[0, :t + 1] - sw_b[0, :t + 1]).abs().max()),
                    float((dur_a[0, :t + 1] - dur_b[0, :t + 1]).abs().max()))
    return dict(max_abs_logit_drift=worst, causal=worst <= tol,
                positions_tested=seq_len - 1)
