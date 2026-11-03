"""The lexical override: dict-freq overrules AnE-LID only on overwhelming evidence."""
import unicodedata

import pandas as pd

from src.lexical_override import has_vietnamese_letters, override_tagged


def _tag(words, langs):
    return list(zip(words, langs))


def test_malay_function_words_tagged_english_become_malay():
    tagged = _tag(["Saya", "akan", "contact", "supplier", "esok"],
                  ["Malay", "English", "English", "English", "Malay"])
    out, changed = override_tagged(tagged, "Malay", "English")
    assert [l for _, l in out] == ["Malay", "Malay", "English", "English", "Malay"]
    assert changed == [False, True, False, False, False]


def test_vietnamese_letters_force_vietnamese_even_without_frequency():
    # "chắn" carries a dot-below-free but Vietnamese-only composite; "sẽ" is very common.
    tagged = _tag(["nhưng", "chắc", "chắn", "sẽ", "add", "some", "ideas"],
                  ["Vietnamese", "English", "English", "English", "English", "English", "English"])
    out, changed = override_tagged(tagged, "Vietnamese", "English")
    assert [l for _, l in out] == ["Vietnamese"] * 4 + ["English"] * 3
    assert changed == [False, True, True, True, False, False, False]


def test_decomposed_unicode_is_normalised_before_the_letter_check():
    nfd = unicodedata.normalize("NFD", "của")
    assert nfd != "của" and has_vietnamese_letters(nfd)


def test_french_and_spanish_accents_are_not_vietnamese():
    for w in ("café", "être", "côté", "año", "über", "résumé"):
        assert not has_vietnamese_letters(w), w


def test_shared_and_ambiguous_words_keep_the_ane_label():
    # "a", "in", "me", "online" are common in both languages: no override either way.
    for l1, words in (("Spanish", ["a", "me"]), ("Italian", ["in", "a"]),
                      ("Philippines", ["online"]), ("French", ["a"])):
        for lab in (l1, "English"):
            tagged = _tag(words, [lab] * len(words))
            out, changed = override_tagged(tagged, l1, "English")
            assert out == tagged and not any(changed), (l1, words, lab)


def test_clear_english_word_tagged_l1_becomes_english():
    tagged = _tag(["es", "incredible"], ["Spanish", "Spanish"])
    out, changed = override_tagged(tagged, "Spanish", "English")
    assert [l for _, l in out] == ["Spanish", "English"]
    assert changed == [False, True]


def test_acronyms_skip_the_frequency_rule():
    # "AI" lowercases to French "ai" (j'ai): the lookup would be of an unrelated word.
    tagged = _tag(["especially", "in", "AI", "ai"], ["English", "English", "English", "English"])
    out, changed = override_tagged(tagged, "French", "English")
    assert changed == [False, False, False, True]


def test_names_are_never_touched():
    tagged = _tag(["Saya", "akan"], ["Malay", "English"])
    out, changed = override_tagged(tagged, "Malay", "English", is_entity=[False, True])
    assert out == tagged and changed == [False, False]


def test_punctuation_and_other_labels_are_never_touched():
    tagged = _tag([",", "akan", "123"], ["other", "other", "other"])
    out, changed = override_tagged(tagged, "Malay", "English")
    assert out == tagged and not any(changed)


def test_out_of_scope_pairs_are_unchanged():
    # Hindi (dict-freq can't read romanized Hindi), German->French (dict-freq is production),
    # and script-tagged pairs are all left alone.
    cases = [("Hindi", "English", ["main", "tha"], ["English", "English"]),
             ("German", "French", ["und", "avec"], ["French", "German"]),
             ("Chinese", "English", ["我", "akan"], ["Chinese", "English"])]
    for l1, l2, words, langs in cases:
        tagged = _tag(words, langs)
        out, changed = override_tagged(tagged, l1, l2)
        assert out == tagged and not any(changed), l1


def test_dataframe_pass_touches_only_ane_rows_and_rederives_runs():
    from src.persist import apply_lexical_override, derive_name_policies

    rows = pd.DataFrame({
        "row_id": [1, 2],
        "first_language": ["Malay", "German"],
        "second_language": ["English", "French"],
        "method": ["ane", "dict"],
        "tokens": [["Saya", "akan", "cuba", "update"], ["und", "avec"]],
        "token_lang": [["Malay", "English", "Malay", "English"], ["German", "French"]],
        "is_entity": [[False] * 4, [False] * 2],
    })
    rows = derive_name_policies(rows)
    assert list(rows.l2_runs[0]) == [1, 1]           # the fake switch on "akan" + the real one
    out = apply_lexical_override(rows)
    assert list(out.token_lang[0]) == ["Malay", "Malay", "Malay", "English"]
    assert list(out.lexical_override[0]) == [False, True, False, False]
    assert list(out.l2_runs[0]) == [1]               # only "update" is a switch now
    assert list(out.token_lang[1]) == ["German", "French"]
    assert not any(out.lexical_override[1])
