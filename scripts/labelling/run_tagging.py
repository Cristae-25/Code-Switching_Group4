#!/usr/bin/env python
"""Tag the whole SwitchLingua corpus and write data/tagged.parquet.

Run this instead of the notebook cells - it pins its own interpreter, so there
is no kernel to choose and no way to end up on the wrong environment:

    ./.venv/bin/python scripts/labelling/run_tagging.py            # full run
    ./.venv/bin/python scripts/labelling/run_tagging.py --no-entities  # skip the NE pass
    ./.venv/bin/python scripts/labelling/run_tagging.py --limit 2000   # quick check

The notebook then only needs to *read* the parquet, which works on any kernel
with pandas + pyarrow - including the Anaconda one.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def load_corpus():
    """Prefer the already-downloaded CSVs; fall back to the Hub.

    The snapshot is on disk from the notebook's first run, so the default path
    needs no network and no HF token - one less thing that can fail mid-run.
    """
    import glob

    import pandas as pd

    pattern = os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--Shelton1013--SwitchLingua_text/"
        "snapshots/*/*.csv")
    files = sorted(glob.glob(pattern))
    if files:
        print(f"reading {len(files)} cached CSVs")
        return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    print("cache miss - downloading from the Hub")
    from dotenv import load_dotenv
    from datasets import load_dataset
    load_dotenv()
    return load_dataset("Shelton1013/SwitchLingua_text")["train"].to_pandas()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="tag only the first N rows (smoke test)")
    ap.add_argument("--out", default=None, help="output parquet path")
    ap.add_argument("--no-entities", action="store_true",
                    help="skip the AnE-NER pass; names keep their tagger's language")
    args = ap.parse_args()

    # Fail loudly and early rather than 80 minutes in.
    try:
        import pandas as pd
        import torch
        import transformers
    except ImportError as exc:
        print(f"FATAL: {exc}\nRun with ./.venv/bin/python, not plain `python`.")
        return 1
    if int(transformers.__version__.split(".")[0]) < 5:
        print(f"FATAL: transformers {transformers.__version__} segfaults on "
              f"torch {torch.__version__}. Use ./.venv/bin/python.")
        return 1

    from src.bert_tagger import diagnose
    from src.persist import (DEFAULT_PATH, ane_labels, entity_masks, load_tagged,
                             save_tagged, tag_dataframe)

    print("=" * 62)
    diagnose()
    print("=" * 62)

    df = load_corpus()
    if args.limit:
        df = df.head(args.limit)
    print(f"tagging {len(df):,} rows\n")

    t0 = time.time()
    ents = None
    if not args.no_entities:
        # Cached in data/entities.parquet; resumes if interrupted.
        ents = entity_masks(df)
        print(f"entity pass done in {(time.time() - t0) / 60:.1f} min")
    # Batched AnE-LID over the Latin->English rows, cached in data/ane_lid.parquet.
    t1 = time.time()
    lid = ane_labels(df)
    print(f"AnE-LID pass done in {(time.time() - t1) / 60:.1f} min")
    tagged = tag_dataframe(df, entities=ents, ane=lid)
    elapsed = time.time() - t0

    path = save_tagged(tagged, args.out or DEFAULT_PATH)
    size = os.path.getsize(path) / 1e6
    print(f"\nwrote {path}  ({len(tagged):,} rows, {size:.0f} MB, "
          f"{elapsed / 60:.1f} min)")

    back = load_tagged(path)
    assert len(back) == len(tagged), "row count changed on round-trip"
    assert all(sum(r.segment_len) == len(r.tokens)
               for r in back.head(1000).itertuples()), "segments do not reconstruct"
    print("round-trip verified")

    print("\nrows by trust tier:")
    print(back.quality.value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
