#!/usr/bin/env python
"""Write the shareable per-token label files (switch + duration targets).

    ./.venv/bin/python scripts/labelling/export_token_labels.py

Reads data/tagged.parquet (word labels) and data/subword.parquet (built by
build_subword.py) and writes, into data/final/:

    token_labels_word.parquet      one row per conversation; per word:
                                   tokens, lang, y_sw, y_len, y_dur, censored, dur_known, is_entity
    token_labels_subword.parquet   the same per XLM-R piece: pieces, piece_ids, lang, y_sw, ...

Column meanings are in src/token_labels.py. The names policy is the default
(script_words); kaggle/load_labels.ipynb builds the other policies.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.token_labels import subword_labels, word_labels  # noqa: E402


def describe(name: str, frame: pd.DataFrame) -> None:
    sw = np.concatenate([np.asarray(x) for x in frame.y_sw])
    known = np.concatenate([np.asarray(x) for x in frame.dur_known])
    dur = np.concatenate([np.asarray(x, dtype=object) for x in frame.y_dur])[known]
    counts = {c: int((dur == c).sum()) for c in ("small", "medium", "large")}
    print(f"{name}: {len(frame):,} rows | {int((sw >= 0).sum()):,} scored units | "
          f"{int((sw == 1).sum()):,} switches | "
          + " ".join(f"{c} {n:,}" for c, n in counts.items()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tagged", default="data/tagged.parquet")
    ap.add_argument("--subword", default="data/subword.parquet")
    ap.add_argument("--out-dir", default="data/final")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    t0 = time.time()
    tagged = pd.read_parquet(args.tagged, columns=["row_id", "first_language", "second_language",
                                                   "quality", "tokens", "token_lang", "is_entity"])
    words = word_labels(tagged).sort_values("row_id", ignore_index=True)
    path = os.path.join(args.out_dir, "token_labels_word.parquet")
    words.to_parquet(path, index=False)
    describe(path, words)

    sub = pd.read_parquet(args.subword, columns=["row_id", "first_language", "second_language",
                                                 "quality", "pieces", "piece_ids", "piece_lang"])
    pieces = subword_labels(sub).sort_values("row_id", ignore_index=True)
    path = os.path.join(args.out_dir, "token_labels_subword.parquet")
    pieces.to_parquet(path, index=False)
    describe(path, pieces)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
