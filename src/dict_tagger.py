"""Dictionary-lookup tagging - the obvious approach, measured.

Two variants of "look the word up in each language's dictionary":

``tag_naive``      if the word exists in the second language's list, call it the
                   second language. This is the intuitive rule, and it fails on
                   lexical overlap: most *common* words exist in both lists.
``tag_frequency``  compare how common the word is in each language and take the
                   higher. Strictly better, and the reason is instructive - the
                   question is never "is this word in the dictionary" but "which
                   language is it more characteristic of".

Frequencies come from `wordfreq` (Zipf scale: ~7 = very common, 0 = absent).
"""
from __future__ import annotations

from functools import lru_cache

from wordfreq import zipf_frequency

LANG_TO_CODE = {
    "English": "en", "Spanish": "es", "German": "de", "French": "fr",
    "Italian": "it", "Malay": "ms", "Philippines": "fil",
    "Vietnamese": "vi", "Hindi": "hi",
}


@lru_cache(maxsize=500_000)
def zipf(word: str, code: str) -> float:
    return zipf_frequency(word, code)


def _is_word(token: str) -> bool:
    return any(ch.isalpha() for ch in token)


def tag_naive(tokens: list[str], first_language: str,
              second_language: str) -> list[tuple[str, str]]:
    """"If it's in the second language's dictionary, it's the second language.\""""
    l2 = LANG_TO_CODE.get(second_language, "en")
    out = []
    for t in tokens:
        if not _is_word(t):
            out.append((t, "other"))
        elif zipf(t.lower(), l2) > 0:
            out.append((t, second_language))
        else:
            out.append((t, first_language))
    return out


def tag_frequency(tokens: list[str], first_language: str, second_language: str,
                  margin: float = 0.0) -> list[tuple[str, str]]:
    """Whichever language the word is more characteristic of wins."""
    l1 = LANG_TO_CODE.get(first_language)
    l2 = LANG_TO_CODE.get(second_language, "en")
    out = []
    for t in tokens:
        if not _is_word(t) or l1 is None:
            out.append((t, "other" if not _is_word(t) else first_language))
            continue
        w = t.lower()
        f1, f2 = zipf(w, l1), zipf(w, l2)
        if f1 == 0 and f2 == 0:
            out.append((t, "other"))          # unknown to both: proper noun etc.
        else:
            out.append((t, second_language if f2 > f1 + margin else first_language))
    return out


def overlap_stats(tokens: list[str], first_language: str,
                  second_language: str) -> dict:
    """How often is a word present in BOTH dictionaries? That is the ceiling
    on the naive rule: every such word is a coin flip for it."""
    l1 = LANG_TO_CODE.get(first_language)
    l2 = LANG_TO_CODE.get(second_language, "en")
    words = [t.lower() for t in tokens if _is_word(t)]
    if not words or l1 is None:
        return dict(words=0, both=0, neither=0, only_l1=0, only_l2=0)
    both = neither = only1 = only2 = 0
    for w in words:
        a, b = zipf(w, l1) > 0, zipf(w, l2) > 0
        both += a and b
        neither += not a and not b
        only1 += a and not b
        only2 += b and not a
    return dict(words=len(words), both=both, neither=neither,
                only_l1=only1, only_l2=only2)
