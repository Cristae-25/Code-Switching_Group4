#!/usr/bin/env python
"""Write data/subword.parquet: XLM-R pieces with each piece's word language.

    ./.venv/bin/python scripts/labelling/build_subword.py              # full corpus, ~3 min CPU
    ./.venv/bin/python scripts/labelling/build_subword.py --limit 2000 --dst /tmp/sw.parquet

Joins the original SwitchLingua text (the cached CSVs, positional index =
row_id) with the final per-word labels in data/tagged.parquet, then writes the
rows in shuffled order (seed 0) so streaming it is not a curriculum by
language. The row_id % 10 split assignment travels with each row.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)

from run_tagging import load_corpus  # noqa: E402
from src.segments import parse_utterances  # noqa: E402
from src.subword import build_frame, write_parquet  # noqa: E402

COLUMNS = ["row_id", "first_language", "second_language", "quality",
           "tokens", "token_lang", "is_entity"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/tagged.parquet")
    ap.add_argument("--dst", default="data/subword.parquet")
    ap.add_argument("--tokenizer", default="xlm-roberta-base")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="random sample of rows (0 = all)")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    t0 = time.time()
    frame = pd.read_parquet(args.src, columns=COLUMNS)
    if args.limit:
        frame = frame.sample(n=args.limit, random_state=args.seed)
    corpus = load_corpus()
    cells = corpus["data_generation_result"]
    texts = {int(r): parse_utterances(cells.iloc[int(r)]) for r in frame.row_id}
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    out, stats = build_frame(frame, texts, tok)
    out = out.iloc[np.random.default_rng(args.seed).permutation(len(out))]
    write_parquet(out, args.dst)

    print(f"{args.dst}: {stats['rows']:,} rows, {stats['pieces']:,} pieces "
          f"in {(time.time() - t0) / 60:.1f} min")
    print(f"alignment failures {stats['failed']:,} | mixed-language pieces "
          f"{stats['mixed_pieces']:,} ({stats['mixed_pieces'] / max(stats['pieces'], 1):.4%})")
    lang = np.concatenate([np.asarray(x, dtype=np.int8) for x in out.piece_lang])
    print(f"piece slots: L1 {np.mean(lang == 0):.1%} | L2 {np.mean(lang == 1):.1%} "
          f"| NONE {np.mean(lang == 2):.1%}")
    y_sw = np.concatenate([np.asarray(x) for x in out.y_sw])
    y_dur = np.concatenate([np.asarray(x) for x in out.y_dur])
    known = np.concatenate([np.asarray(x) for x in out.dur_known])
    print(f"scored pieces {int((y_sw >= 0).sum()):,} | switches {int((y_sw == 1).sum()):,} "
          f"({(y_sw == 1).sum() / max((y_sw >= 0).sum(), 1):.2%})")
    counts = np.bincount(y_dur[known], minlength=3)
    print("duration classes (known): " + " | ".join(
        f"{lab} {c:,} ({c / max(counts.sum(), 1):.1%})"
        for lab, c in zip(["small", "medium", "large"], counts)))


if __name__ == "__main__":
    main()
