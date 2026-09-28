"""Language-pair -> tagging-strategy lookup.

Two regimes, confirmed empirically against the cached SwitchLingua CSVs:

* ``unicode`` - first_language uses a script disjoint from Latin, so a token's
  script *is* its language. Exact, no model. ~68.7% of rows.
* ``model``   - first_language is Latin-script (or romanized), so only
  vocabulary separates it from the second language. Needs constrained LID.

Note the plan's routing table listed Arabic's pair as Arabic<->French; the data
says ``Arabic_eng.csv`` is Arabic<->English, and French appears as a *second*
language only in ``German_French.csv``. Nothing here hardcodes the second
language - callers pass the row's own ``second_language``.
"""

SCRIPT_DISJOINT_CONFIRMED = {
    "Cantonese", "Chinese", "Japanese", "Korean", "Russian", "Arabic",
}

# Resolved from the plan's `unicode_pending_check` gate: 54.7% of Hindi rows are
# romanized Hinglish ("Woh kal Prime Minister Carney se milne wali hain"), so
# Devanagari-vs-Latin is not a reliable split for this language. Routed to model.
NEEDS_MODEL = {
    "German", "Philippines", "Vietnamese", "Malay", "Spanish", "French",
    "Italian", "Hindi",
}

# fastText lid.176 label codes.
LANG_TO_CODE = {
    "German": "de", "Philippines": "tl", "Vietnamese": "vi", "Malay": "ms",
    "Spanish": "es", "French": "fr", "Italian": "it", "English": "en",
    "Hindi": "hi", "Russian": "ru", "Arabic": "ar", "Chinese": "zh",
    "Cantonese": "zh", "Japanese": "ja", "Korean": "ko",
}

# Script expected for each unicode-routed first_language. A token in one of
# these scripts is the first language; a Latin token is the second language.
_EXPECTED_SCRIPT = {
    "Cantonese": {"Han"},
    "Chinese": {"Han"},
    "Japanese": {"Han", "Hiragana", "Katakana"},
    "Korean": {"Hangul"},
    "Russian": {"Cyrillic"},
    "Arabic": {"Arabic"},
    "Hindi": {"Devanagari"},
}

# fastText's `hi` model is trained on Devanagari. Romanized Hindi tokens are
# out of distribution for every label, so rows here are flagged low-confidence
# rather than silently trusted.
ROMANIZATION_RISK = {"Hindi"}


# --- tier 3: context-aware LinCE BERT token classifiers ---------------------
#
# Measured on hand-labelled tokens from the real rows (NE excluded), fastText
# vs. the spaeng model used cross-lingually:
#
#     language      fastText   spaeng    -> routed to
#     German           100%      80%        fasttext
#     French            90%      68%        fasttext
#     Vietnamese        90%      88%        fasttext   (tie; diacritics carry it)
#     Italian           96%     100%        spaeng
#     Spanish           94%     100%        spaeng
#     Philippines       86%      94%        spaeng
#     Malay             75%      88%        spaeng
#     Hindi          broken       -         hineng
#
# spaeng transfers as a generic Latin-L1-vs-English discriminator to Romance and
# Austronesian languages but *regresses* Germanic/French, so routing is per
# language and empirical, not by family. hineng does not transfer at all (17%
# Malay / 64% Tagalog) - it is Hindi-only.
#
# Sample sizes are small (n=26-48 per language); treat Vietnamese and Italian,
# where the gap is inside a couple of tokens, as ties rather than firm results.
MODEL_REPO = {
    "spaeng": "sagorsarker/codeswitch-spaeng-lid-lince",
    "hineng": "sagorsarker/codeswitch-hineng-lid-lince",
}
MODEL_L1_LABEL = {"spaeng": "spa", "hineng": "hin"}
MODEL_FOR = {
    "Hindi": "hineng",
    "Spanish": "spaeng", "Italian": "spaeng",
    "Malay": "spaeng", "Philippines": "spaeng",
}


# --- tier 4: dictionary frequency comparison (measured best) ----------------
#
# Measured on data/gold_eval.json (324 hand-labelled tokens) and confirmed on
# 767 rows against the dataset's own cs_ratio_result:
#
#   language      dict-freq   best model      ratio err (dict vs prod)
#   Malay            100%      88% mBERT        0.103 vs 0.183
#   Philippines      100%      94% mBERT        0.147 vs 0.222
#   French           100%      90% fastText     0.121 vs 0.153
#   German           100%     100% fastText     0.133 vs 0.151
#   Vietnamese       100%      90% fastText     0.084 vs 0.093
#   Spanish          100%     100% mBERT        0.105 vs 0.104
#   Italian          100%     100% mBERT        0.132 vs 0.124
#   Hindi (roman)     66%     100% mBERT        - model required
#
# dict-freq wins or ties on every Latin-script pair and needs no model at all.
# It also handles a non-English second language (the German<->French rows),
# which neither LinCE model can, so fastText and mBERT-spaeng both retire.
#
# Romanized Hindi is the sole exception: wordfreq's `hi` list is Devanagari and
# romanized Hinglish has no standard orthography, so no word list can cover it.
# Four independent techniques agree the context-aware models are required there
# (fastText 56%, dict-freq 66%, mBERT 100%, XLM-R 100%, CANINE 98%).
USES_DICT = {
    "Spanish", "Italian", "Malay", "Philippines",
    "German", "French", "Vietnamese",
}




# --- tier 5: AnE-LID (current production for every "<Latin L1> vs English") --
#
# The gold set above saturated (every tier at ~99-100% on 26-59 self-labelled
# tokens), so the decision was made on human-annotated public data instead
# (scripts/eval_public.py, L1/EN tokens only):
#
#   set                      n     dict-freq   mBERT    AnE-LID
#   LinCE spaeng dev      31,667     94.1%     98.8%     99.4%
#   LinCE hineng dev      12,303     80.5%     97.4%     98.8%
#   Denglisch (de-en)     59,485     96.2%     53.0%     99.9%  (likely seen in training)
#
# dict-freq's "100%" on the gold set was ~94-96% on human labels. AnE-LID is
# multi-pair trained, so it also covers romanized Hindi and replaces mBERT
# hineng there. It only answers "English or not", so German->French keeps
# dict-freq. Where AnE and dict-freq agree, tokens are 99.3-100% correct.
ANE_L2 = "English"


def ane_to_language(label: str, first_language: str, second_language: str = "English") -> str:
    """Map a raw AnE-LID label to the row's language. ``Mixed`` (an English stem
    in L1 morphology, "rewatchen") goes to L1, its grammatical host."""
    if label == "English":
        return second_language
    if label in ("notEnglish", "Mixed"):
        return first_language
    return "other"


def route(first_language: str, second_language: str = "English") -> str:
    """Return ``"unicode"``, ``"ane"`` or ``"dict"`` for a row.

    Hindi routes per *row* rather than per language - see
    ``dispatch._hindi_is_devanagari``. Devanagari rows are script-disjoint from
    English and go to the exact tagger; only romanized rows need the model.
    ``"bert"`` (mBERT hineng) is retired but still dispatchable for comparison.
    """
    if first_language in SCRIPT_DISJOINT_CONFIRMED:
        return "unicode"
    if second_language != ANE_L2:
        return "dict"
    return "ane"


def lang_code(language: str) -> str | None:
    """Map a dataset language name to its fastText lid.176 label code."""
    return LANG_TO_CODE.get(language)


def expected_script(language: str) -> set[str]:
    """Unicode script name(s) that indicate ``language`` in a mixed token stream."""
    return _EXPECTED_SCRIPT.get(language, {"Latin"})


# --- V1 quality tiers (measured, not assumed) -------------------------------
#
# The plan's escalation rule needs a per-pair accuracy number. Manual labelling
# was not required to get one: the six script-disjoint languages are tagged
# *exactly*, so their disagreement with the dataset's own `computed_ratio`
# (-0.134 .. +0.117, mean |diff| ~0.075) measures the noise in that reference
# signal rather than any tagger error. That band is the yardstick - a
# model-routed pair sitting inside it is behaving like an exact tagger; one far
# outside it is genuinely broken.
#
# Measured on a 150-row-per-language stratified sample (see notebook cell):
#
#   PASS  (inside the exact-tagger band)   Vietnamese .056  Spanish .086
#                                          German .118  French .128  Italian .129
#   FAIL  (2-5x outside it)                Malay .277  Hindi .369  Philippines .406
#
# This inverts the plan's expectation, which predicted German/Malay/Tagalog
# would pass easily and the Romance pairs would fail on shared function words.
# The Romance pairs are fine; lid.176 simply has far less Malay/Tagalog training
# data, and romanized Hindi is not in its label space at all.
V1_PASSES = {"Vietnamese", "Spanish", "German", "French", "Italian"}
V1_FAILS_NEEDS_V2 = {"Malay", "Philippines", "Hindi"}


# Measured token accuracy after routing each language to its best tagger.
# `low` now means "still under the plan's 90% bar with the best option tried".
# Accuracy of the tagger each language is now routed to, on the gold set.
# Every Latin pair scored 100% under dict-freq - but on only 26-50 tokens each,
# self-labelled. `provisional` marks that: good enough to ship, not yet a
# number to publish. The review tool exists to replace these.
_MEASURED_ACCURACY = {
    "Spanish": 1.00, "Italian": 1.00, "Malay": 1.00, "Philippines": 1.00,
    "German": 1.00, "French": 1.00, "Vietnamese": 1.00,   # dict-freq
    "Hindi": 1.00,                                        # mBERT hineng
}


def tagger_quality(first_language: str, second_language: str = "English") -> str:
    """``exact`` | ``provisional`` - how far to trust a row's tags.

    ``exact`` is correct by construction (script-disjoint). ``provisional``
    means measured at >=90% on a small self-labelled sample; human review is
    what promotes it to a defensible number.
    """
    if first_language in SCRIPT_DISJOINT_CONFIRMED:
        return "exact"
    return "provisional" if _MEASURED_ACCURACY.get(first_language, 0) >= 0.90 else "low"
