"""The review page shows each token as the training labels count it (names resolved)."""
from scripts.labelling.review_server import counted_labels


def test_opaque_name_shows_its_clause_language_not_other():
    toks = ["Saya", "rasa", "Elanga", "ada", "potensi", "."]
    langs = ["Malay", "Malay", "other", "Malay", "Malay", "other"]
    ent = [False, False, True, False, False, False]
    out = counted_labels(toks, langs, ent, "Malay", "English")
    assert out == ["Malay", "Malay", "Malay", "Malay", "Malay", "other"]   # "." stays other


def test_name_with_clear_english_words_shows_english():
    toks = ["di", "Harbin", "Asia", "Winter", "Games", "akan"]
    langs = ["Malay", "Malay", "Malay", "English", "English", "Malay"]
    ent = [False, True, True, True, True, False]
    out = counted_labels(toks, langs, ent, "Malay", "English")
    assert out[1:5] == ["English"] * 4


def test_non_name_tokens_are_unchanged():
    toks = ["Saya", "akan", "contact", "supplier"]
    langs = ["Malay", "Malay", "English", "English"]
    assert counted_labels(toks, langs, [False] * 4, "Malay", "English") == langs
