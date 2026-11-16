"""Low-rank adapters (LoRA, Hu et al. 2021) for the pretrained encoders, without ``peft``.

Each targeted ``nn.Linear`` keeps its pretrained weight W frozen and learns a
rank-``r`` update: ``y = W x + (alpha / r) * B A x``. ``B`` starts at zero, so
the adapted encoder is exactly the pretrained one at step 0, like the
zero-initialised language and pair embeddings.

``apply_lora`` freezes the whole encoder first, so only the adapters (plus the
model's own new parameters outside the encoder: heads, language and pair
embeddings) train. LoRA only changes linear layers inside a token's own
computation, so the causal mask is untouched; ``check_causality`` still
re-verifies the assembled model.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

# XLM-R / RoBERTa / BERT names: attention query/key/value, the attention output
# projection and both feed-forward layers are all called one of these.
DEFAULT_TARGETS = ("query", "key", "value", "dense")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.r, self.scale = r, alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + (self.drop(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scale


def apply_lora(encoder: nn.Module, r: int = 16, alpha: float | None = None, dropout: float = 0.05,
               targets=DEFAULT_TARGETS) -> int:
    """Freeze ``encoder`` and wrap its targeted linear layers; returns how many were wrapped."""
    for p in encoder.parameters():
        p.requires_grad_(False)
    alpha = float(2 * r if alpha is None else alpha)
    wrapped = 0
    for parent in list(encoder.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and name in targets:
                setattr(parent, name, LoRALinear(child, r, alpha, dropout))
                wrapped += 1
    if not wrapped:
        raise ValueError(f"no linear layer named any of {targets}")
    return wrapped


def is_lora_param(name: str) -> bool:
    return ".lora_A" in name or ".lora_B" in name
