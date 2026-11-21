import numpy as np

from scripts.analysis.error_breakdown import (CATS, NO_CAT, _suffix_table, classify, ending_categories,
                                              relocate, standardised_f1)
from src.anticipation import L1, L2, NONE
from src.metrics import prf


def test_longest_suffix_wins():
    t = _suffix_table("Korean")
    assert CATS[classify("했습니다", t)] == "final"
    assert CATS[classify("있었는데", t)] == "connective"      # 는데, not the particle 는
    assert CATS[classify("앱을", t)] == "particle"
    assert CATS[classify("구글", t)] == "other"         # 플레이 would be "particle" (이): known noise


def test_endings_are_causal_and_word_scoped():
    # XLM-R pieces: a word is built from its pieces; L2 and punctuation reset it.
    pieces = ["▁했", "었", "어", ",", "▁which", "▁앱", "을"]
    langs = [L1, L1, L1, NONE, L2, L1, L1]
    e = ending_categories(pieces, langs, "Korean", "sentencepiece")
    cats = [CATS[c] if c != NO_CAT else None for c in e]
    assert cats == ["other", "other", "informal", None, None, "other", "particle"]
    # changing a later piece never changes an earlier category
    e2 = ending_categories(pieces[:3] + [",", "▁x", "▁다", "를"], langs, "Korean", "sentencepiece")
    assert e2[:3].tolist() == e[:3].tolist()


def test_wordpiece_scheme_and_unlisted_language():
    e = ending_categories(["했", "##어요"], [L1, L1], "Korean", "wordpiece")
    assert CATS[e[1]] == "final"
    assert (ending_categories(["▁hola"], [L1], "Spanish", "sentencepiece") == NO_CAT).all()


def test_relocate_moves_every_array():
    a, b = np.array([1, 2, 3]), np.array([7, 8, 9])
    ra, rb = relocate({0: 2}, a, b, fill=-1)
    assert ra.tolist() == [-1, 2, 1] and rb.tolist() == [-1, 8, 7]


def test_standardised_f1_equals_f1_at_own_mix_and_reweights_otherwise():
    rng = np.random.default_rng(0)
    y = rng.random(400) < 0.3
    pred = y ^ (rng.random(400) < 0.2)
    bucket = rng.integers(0, 2, 400)
    cell = bucket * 2 + y
    own = np.bincount(cell, minlength=4) / len(cell)
    assert abs(standardised_f1(y, pred, cell, own) - prf(y, pred)["f1"]) < 1e-9
    # put all the weight on bucket 0: equals plain F1 on bucket 0
    ref = own.copy()
    ref[2:] = 0
    ref /= ref.sum()
    m = bucket == 0
    assert abs(standardised_f1(y, pred, cell, ref) - prf(y[m], pred[m])["f1"]) < 1e-9
