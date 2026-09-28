"""Single entry point that routes a token stream to the right tagger.

Kept separate so ``segments``, ``inspection`` and ``persist`` share one copy of
the branch, and so the heavy imports (torch, the models) stay lazy - a
unicode-routed row loads none of them.
"""
from __future__ import annotations

import regex as re

from .routing import ane_to_language, route

_DEVANAGARI = re.compile(r"\p{Script=Devanagari}")
_LATIN = re.compile(r"\p{Script=Latin}")


def _hindi_is_devanagari(tokens: list[str]) -> bool:
    """Hindi rows come in two scripts; only the romanized half needs a model.

    Routing per *language* sent Devanagari rows to the LinCE model, which is
    trained on romanized Hinglish and so is out of distribution for them - while
    the Unicode tagger handles Devanagari-vs-Latin exactly. Decide per row.
    """
    dev = sum(1 for t in tokens if _DEVANAGARI.search(t))
    lat = sum(1 for t in tokens if _LATIN.search(t))
    return dev > lat


def row_strategy(tokens: list[str], first_language: str, second_language: str) -> str:
    """``route`` plus the per-row Devanagari check for Hindi."""
    strategy = route(first_language, second_language)
    if (strategy in ("ane", "bert") and first_language == "Hindi"
            and _hindi_is_devanagari(tokens)):
        return "unicode"
    return strategy


def _devanagari_is_hindi(tagged, first_language):
    """Romanized-majority Hindi rows still carry some Devanagari words, and the
    LinCE model called 24,113 of them English. Script settles those exactly."""
    if first_language != "Hindi":
        return tagged
    return [(t, first_language if _DEVANAGARI.search(t) else lang) for t, lang in tagged]


def _names_from_context(tagged, is_entity, ane_labels, first_language, second_language):
    """Script-tagged rows: a Latin-script *name* takes AnE-LID's contextual label.

    Script alone calls every Latin token English, which is right for
    "Game of Thrones" but invents a switch for "Messi" in a Cantonese sentence.
    AnE-LID, run on a window around the name, tells the two apart. Names in the
    row's own script (中關村) stay L1, and non-entity tokens keep their script
    label. An empty label means the name was never scored; script stands.
    """
    out = []
    for (t, lang), ent, lab in zip(tagged, is_entity, ane_labels):
        if ent and lab and _LATIN.search(t):
            lang = ane_to_language(lab, first_language, second_language)
        out.append((t, lang))
    return out


def tag_tokens(tokens: list[str], first_language: str, second_language: str,
               ane_labels: list[str] | None = None,
               is_entity: list[bool] | None = None) -> list[tuple[str, str]]:
    """Tag tokens as first/second language via whichever tier the pair needs.

    ``ane_labels`` are precomputed raw AnE-LID word labels (``persist.ane_labels``
    batches them over the corpus). For ``ane`` rows they are the labels; without
    them the model runs on this row. For script-tagged rows they are optional
    and only cover the windows around Latin-script names (see
    ``_names_from_context``); ``""`` marks unscored positions.
    """
    strategy = row_strategy(tokens, first_language, second_language)
    if strategy == "ane":
        from .ane_tagger import tag_tokens_ane
        return _devanagari_is_hindi(
            tag_tokens_ane(tokens, first_language, second_language, labels=ane_labels),
            first_language)
    if strategy == "unicode":
        from .unicode_tagger import tag_tokens_unicode
        tagged = tag_tokens_unicode(tokens, first_language, second_language)
        if (ane_labels is not None and is_entity is not None
                and len(ane_labels) == len(tokens) == len(is_entity)):
            tagged = _names_from_context(tagged, is_entity, ane_labels,
                                         first_language, second_language)
        return tagged
    if strategy == "bert":
        from .bert_tagger import tag_tokens_bert
        return _devanagari_is_hindi(
            tag_tokens_bert(tokens, first_language, second_language), first_language)
    if strategy == "dict":
        from .dict_tagger import tag_frequency
        return tag_frequency(tokens, first_language, second_language)
    from .model_tagger import tag_tokens_model
    return tag_tokens_model(tokens, first_language, second_language)
