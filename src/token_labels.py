"""Readable per-token labels: the form shared with the team.

One row per conversation, with parallel lists aligned to its units (words, or
XLM-R pieces):

    lang       "L1" | "L2" | "-"   ("-" = punctuation/digits, skipped by the labels)
    y_sw       1 if the NEXT unit switches language, 0 if not, -1 where undefined
    y_len      at a switch, how many units the new language lasts (0 elsewhere)
    y_dur      class of y_len: "small" 1-2 | "medium" 3-6 | "large" 7+ ("" elsewhere)
    censored   the new run is still open when the text ends (y_len is a lower bound)
    dur_known  the class is certain: the run closed, or it is cut off but already 7+

Everything is derived from the final word labels by ``anticipation`` - these
functions only change the representation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .anticipation import DUR_LABELS, anticipatory_labels, token_slots

LANG_CODES = np.array(["L1", "L2", "-"])
META = ["row_id", "first_language", "second_language", "split", "quality"]


def split_of(row_id: int) -> str:
    """The row-level split used everywhere: row_id % 10 -> 0-7 train, 8 val, 9 test."""
    k = int(row_id) % 10
    return "train" if k < 8 else "val" if k == 8 else "test"


def readable(units, slots) -> dict:
    """Labels for one row as plain lists, aligned to ``units``."""
    slots = np.asarray(slots, dtype=np.int8)
    lab = anticipatory_labels(slots)
    return {
        "lang": LANG_CODES[slots].tolist(),
        "y_sw": lab["y_sw"].astype(int).tolist(),
        "y_len": lab["y_len"].astype(int).tolist(),
        "y_dur": [DUR_LABELS[d] if d >= 0 else "" for d in lab["y_dur"]],
        "censored": lab["censored"].tolist(),
        "dur_known": lab["dur_known"].tolist(),
    }


def _meta(r) -> dict:
    return {"row_id": int(r.row_id), "first_language": r.first_language,
            "second_language": r.second_language, "split": split_of(r.row_id),
            "quality": r.quality}


def word_labels(frame: pd.DataFrame, names: str | None = None) -> pd.DataFrame:
    """Per-word labels from a tagged frame (``data/tagged.parquet`` columns)."""
    rows = []
    for r in frame.itertuples(index=False):
        ent = list(r.is_entity)
        slots = token_slots(list(r.tokens), list(r.token_lang), r.first_language,
                            r.second_language, ent, names)
        rows.append({**_meta(r), "tokens": list(r.tokens), **readable(r.tokens, slots),
                     "is_entity": [int(e) for e in ent]})
    return pd.DataFrame(rows)


def subword_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-piece labels from a ``data/subword.parquet``-shaped frame."""
    rows = []
    for r in frame.itertuples(index=False):
        rows.append({**_meta(r), "pieces": list(r.pieces),
                     **readable(r.pieces, r.piece_lang), "piece_ids": list(r.piece_ids)})
    return pd.DataFrame(rows)
