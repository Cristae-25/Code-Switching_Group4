import numpy as np

from scripts.punct_breakdown import ADJ, COMMA, NO_LABEL, OTHER, SENT, TURN, gap_buckets
from src.anticipation import L1, L2, NONE


def test_buckets_by_gap_content():
    pieces = ["▁I", "▁went", "▁home", ",", "▁然后", "。", "▁ok", "▁(", "5", ")", "▁yes", "▁no"]
    langs = [L2, L2, L2, NONE, L1, NONE, L2, NONE, NONE, NONE, L2, L2]
    b = gap_buckets(pieces, langs)
    assert b.tolist() == [ADJ, ADJ, COMMA, NO_LABEL, SENT, NO_LABEL, OTHER,
                          NO_LABEL, NO_LABEL, NO_LABEL, ADJ, NO_LABEL]


def test_blank_piece_is_adjacent_and_turn_wins():
    pieces = ["▁a", "▁", "b", "▁c", ".", "▁d"]
    langs = [L1, NONE, L1, L2, NONE, L1]
    utts = [0, 0, 0, 0, 0, 1]
    assert gap_buckets(pieces, langs, utts).tolist() == [ADJ, NO_LABEL, ADJ, TURN, NO_LABEL, NO_LABEL]


def test_sentence_end_beats_comma():
    assert gap_buckets(["▁a", ",", "。", "▁b"], [L1, NONE, NONE, L1]).tolist()[0] == SENT
