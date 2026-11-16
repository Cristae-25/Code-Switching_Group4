"""Constrained fastText tagging, for Latin-script language pairs.

V1 of the plan: take lid.176's 176-way distribution and read off only the two
labels the row can actually be, so the model never has to *find* the language -
it only has to rank two known candidates. No training required.

Prediction is memoised per token string. Token vocabulary is far smaller than
token count (a conversation reuses "the", "de", "que" constantly), so the cache
turns roughly one model call per token into one per distinct word.
"""
from __future__ import annotations

import os
from functools import lru_cache

import fasttext

from .routing import lang_code

_DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              "models", "lid.176.bin")
_model = None


def load_lid(path: str | None = None):
    """Load (once) the fastText language-ID model."""
    global _model
    if _model is None:
        target = path or _DEFAULT_MODEL
        if not os.path.exists(target):
            raise FileNotFoundError(
                f"fastText LID model not found at {target}. Download with:\n"
                "  curl -L -o models/lid.176.bin "
                "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
            )
        _model = fasttext.load_model(target)
    return _model


@lru_cache(maxsize=1_000_000)
def _scores(token: str) -> dict[str, float]:
    """Full 176-way label distribution for one token, memoised."""
    # fastText rejects newlines; tokens are single words but be defensive.
    labels, probs = load_lid().predict(token.replace("\n", " "), k=176)
    return {l.replace("__label__", ""): p for l, p in zip(labels, probs)}


def tag_token_constrained(token: str, lang_a: str, lang_b: str) -> str:
    """Constrain fastText's 176-way prediction to the two known candidates.

    Ties and mutual-zero cases fall to ``lang_a`` (the first language), which is
    the majority class in every row of this dataset.
    """
    code_a, code_b = lang_code(lang_a), lang_code(lang_b)
    s = _scores(token)
    return lang_a if s.get(code_a, 0.0) >= s.get(code_b, 0.0) else lang_b


def tag_tokens_model(tokens: list[str], first_language: str,
                     second_language: str) -> list[tuple[str, str]]:
    """Tag tokens as first/second language; punctuation and digits are ``other``."""
    out = []
    for tok in tokens:
        if not any(ch.isalpha() for ch in tok):
            out.append((tok, "other"))
        else:
            out.append((tok, tag_token_constrained(tok, first_language, second_language)))
    return out


def unknown_rate(tokens: list[str], lang_a: str, lang_b: str) -> float:
    """Fraction of word tokens neither candidate language scores at all.

    High values mean the model is out of distribution - the signal that catches
    romanized Hindi, where lid.176's `hi` label only covers Devanagari.
    """
    code_a, code_b = lang_code(lang_a), lang_code(lang_b)
    words = [t for t in tokens if any(ch.isalpha() for ch in t)]
    if not words:
        return 0.0
    unknown = sum(
        1 for t in words
        if _scores(t).get(code_a, 0.0) == 0.0 and _scores(t).get(code_b, 0.0) == 0.0
    )
    return unknown / len(words)
