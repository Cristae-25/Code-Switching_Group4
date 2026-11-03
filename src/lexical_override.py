"""Overrule AnE-LID where dict-freq's evidence is overwhelming (added 2026-11-03).

AnE-LID was chosen on LinCE spa-eng / hin-eng and Denglisch, the only
human-annotated sets we have. It was never checked on Malay, Vietnamese or
Tagalog, and there it tags common L1 function words English: Malay ``akan``
×1,496, ``dan``, ``yang``; Vietnamese ``và``, ``sẽ``, ``của``, even spans like
"bekerjasama dengan agensi lain untuk". Each one invents two switches.
``scripts/consensus_f1.py`` found 40-45% of the positions around such words
labelled as switches, and the trained model refusing them (recall 0.16-0.18).

Two rules, applied to word tokens that are not names:

1. **Vietnamese letters.** In a Vietnamese row, a word containing ``đ ă ơ ư``
   or a stacked tone mark (U+1EA0-U+1EF9) is Vietnamese. No English word has
   them; French/Spanish accents (é, ê, ñ) are not in the set.
2. **Frequency margin.** The word flips to the other language of the pair when
   it is common there (Zipf >= ``FLOOR``) and at least ``MARGIN`` Zipf more
   common there than in the language AnE gave it. Margin 2.0 = 100x more
   frequent. Shared words (``a``, ``in``, ``me``, ``online``) never clear it.
   All-caps tokens skip this rule: an acronym's lowercase form is an unrelated
   word ("AI" vs French "ai", as in j'ai).

Scope: ``OVERRIDE_L1`` with English as L2. Romanized Hindi is excluded
(wordfreq's Hindi list is Devanagari, so dict-freq can't read it), German->French
is dict-freq already, and script-tagged pairs are exact.
"""
from __future__ import annotations

import unicodedata

from .dict_tagger import LANG_TO_CODE, zipf

OVERRIDE_L1 = frozenset({"Spanish", "Italian", "Malay", "Philippines", "German", "French",
                         "Vietnamese"})
MARGIN = 2.0
FLOOR = 4.0
_VI_LETTERS = frozenset("đĐăĂơƠưƯ")


def has_vietnamese_letters(token: str) -> bool:
    """True if the word has a letter only Vietnamese uses (checked after NFC)."""
    for ch in unicodedata.normalize("NFC", token):
        if ch in _VI_LETTERS or 0x1EA0 <= ord(ch) <= 0x1EF9:
            return True
    return False


def in_scope(first_language: str, second_language: str) -> bool:
    return second_language == "English" and first_language in OVERRIDE_L1


def override_tagged(tagged: list[tuple[str, str]], first_language: str, second_language: str,
                    is_entity: list[bool] | None = None, margin: float = MARGIN,
                    floor: float = FLOOR) -> tuple[list[tuple[str, str]], list[bool]]:
    """Return ``(tagged, changed)``: the corrected labels and which tokens moved."""
    if not in_scope(first_language, second_language):
        return list(tagged), [False] * len(tagged)
    l1, l2 = first_language, second_language
    c1, c2 = LANG_TO_CODE[l1], LANG_TO_CODE[l2]
    ents = is_entity if is_entity is not None else [False] * len(tagged)
    out, changed = [], []
    for (tok, lang), ent in zip(tagged, ents):
        new = lang
        if not ent and lang in (l1, l2) and any(ch.isalpha() for ch in tok):
            if l1 == "Vietnamese" and has_vietnamese_letters(tok):
                new = l1
            elif not (len(tok) >= 2 and tok.isupper()):    # acronyms: "AI" is not French "ai"
                w = tok.lower()
                f1, f2 = zipf(w, c1), zipf(w, c2)
                if lang == l2 and f1 >= floor and f1 - f2 >= margin:
                    new = l1
                elif lang == l1 and f2 >= floor and f2 - f1 >= margin:
                    new = l2
        out.append((tok, new))
        changed.append(new != lang)
    return out, changed
