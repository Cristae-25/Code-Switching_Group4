"""Readable per-token labels, the form shared with the team."""
import numpy as np
import pandas as pd

from src.anticipation import NONE
from src.token_labels import readable, subword_labels, word_labels


def test_readable_labels_every_unit():
    out = readable(["hola", ",", "amigo", "hello", "friend"], [0, NONE, 0, 1, 1])
    assert out["lang"] == ["L1", "-", "L1", "L2", "L2"]
    assert out["y_sw"] == [0, -1, 1, 0, -1]
    assert out["y_len"] == [0, 0, 2, 0, 0]          # the English run lasts 2 words
    assert out["y_dur"] == ["", "", "small", "", ""]
    assert out["censored"][2] and not out["dur_known"][2]


def test_word_labels_one_row_per_conversation():
    frame = pd.DataFrame([dict(row_id=18, first_language="Spanish", second_language="English",
                               quality="provisional", tokens=["hola", "amigo", "hello"],
                               token_lang=["Spanish", "Spanish", "English"],
                               is_entity=[False, False, False])])
    out = word_labels(frame)
    r = out.iloc[0]
    assert list(out.columns[:5]) == ["row_id", "first_language", "second_language", "split", "quality"]
    assert r.split == "val" and list(r.tokens) == ["hola", "amigo", "hello"]
    assert list(r.lang) == ["L1", "L1", "L2"] and list(r.y_sw) == [0, 1, -1]
    assert list(r.is_entity) == [0, 0, 0]


def test_subword_labels_keep_pieces_and_ids():
    sw = pd.DataFrame([dict(row_id=9, first_language="Cantonese", second_language="English",
                            quality="exact", pieces=["▁", "今日", "▁good"], piece_ids=[6, 100, 200],
                            piece_lang=[NONE, 0, 1])])
    r = subword_labels(sw).iloc[0]
    assert r.split == "test" and list(r.pieces) == ["▁", "今日", "▁good"]
    assert list(r.lang) == ["-", "L1", "L2"] and list(r.y_sw) == [-1, 1, -1]
    assert list(r.piece_ids) == [6, 100, 200]


def test_to_subword_frame_restores_the_training_format():
    from src.token_labels import to_subword_frame
    labels = pd.DataFrame([dict(row_id=r, first_language="Spanish", second_language="English",
                                split="train", quality="provisional", pieces=["▁a", ",", "▁b"],
                                piece_ids=[10 + r, 4, 20 + r], lang=["L1", "-", "L2"])
                           for r in range(6)])
    out = to_subword_frame(labels, seed=0)
    assert sorted(out.row_id) == list(range(6)) and list(out.row_id) != list(range(6))
    r = out.set_index("row_id").loc[3]
    assert list(r.piece_lang) == [0, NONE, 1] and list(r.piece_ids) == [13, 4, 23]
    assert list(out.row_id) == list(to_subword_frame(labels, seed=0).row_id)   # deterministic
