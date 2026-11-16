import numpy as np

from scripts.analysis.grammar_breakdown import freq_band, row_features
from src.anticipation import L1, L2, NONE

# "yo quiero the coche , pero no ."  ->  es es en es , es es .
PIECES = ["▁yo", "▁quiero", "▁the", "▁co", "che", ",", "▁pero", "▁no", "."]
LANGS = [L1, L1, L2, L1, L1, NONE, L1, L1, NONE]
WORDS = [0, 1, 2, 3, 3, 4, 5, 6, 7]
UTTS = [0] * 9


def _f():
    return row_features(PIECES, LANGS, WORDS, UTTS, ("es", "en"))


def test_run_length_counts_words_not_pieces():
    # yo=1, quiero=2, the=1 (switch), coche=1 at both pieces, pero=2, no=3
    assert _f()["run"].tolist() == [0, 1, 0, 0, 0, -1, 1, 2, -1]


def test_clause_position_resets_after_a_comma():
    # 1, 2, 3, 4 (2-3 bin=1 for 2..3; 4 -> bin 2), then pero=1, no=2
    assert _f()["clause"].tolist() == [0, 1, 1, 2, 2, -1, 0, 1, -1]


def test_word_end_marks_only_the_last_piece_of_each_word():
    assert _f()["word_end"].tolist() == [True, True, True, False, True, False, True, True, False]


def test_frequency_uses_the_word_so_far_in_its_own_language():
    f = _f()
    assert f["freq"][2] == 0                     # "the" in English: a function word
    assert f["direction"].tolist()[:4] == [0, 0, 1, 0]


def test_frequency_bands():
    assert [freq_band(z) for z in (7.0, 5.0, 3.5, 1.0, 0.0)] == [0, 1, 2, 3, 4]


def test_zipf_needs_no_tokenizer_and_reads_traditional_chinese():
    from scripts.analysis.grammar_breakdown import zipf
    assert zipf("は", "ja") > 7 and zipf("는", "ko") > 7      # no MeCab needed
    assert zipf("們", "zh") == zipf("们", "zh") > 5             # traditional -> simplified
    assert zipf("qwxzv", "en") == 0.0
