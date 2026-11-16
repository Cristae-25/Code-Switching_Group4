#!/usr/bin/env python
"""Write a row-shuffled copy of the tagged corpus.

`ParquetFile.iter_batches` can only read forward, and the source CSVs are
contiguous per language: Cantonese occupies the first ~119 K rows, then Arabic,
and so on. A bounded shuffle buffer (4,096 windows) is far smaller than a
language block, so streaming the file in order trains the model as an
unintended *curriculum by language* - tens of thousands of consecutive steps on
one pair, then a hard switch. The training loss jumps visibly at each boundary
and the final model is biased toward whichever language came last.

Shuffling the rows once, offline, fixes it without costing anything at training
time. `row_id` travels with the row, so the `row_id % 10` split assignment is
unchanged - train/val/test membership is identical to the unshuffled file.
"""
from __future__ import annotations

import argparse

import numpy as np
import pyarrow.parquet as pq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/tagged.parquet")
    ap.add_argument("--dst", default="data/tagged_shuffled.parquet")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--row-group-size", type=int, default=8192)
    args = ap.parse_args()

    table = pq.read_table(args.src)
    perm = np.random.default_rng(args.seed).permutation(table.num_rows)
    shuffled = table.take(perm)
    pq.write_table(shuffled, args.dst, compression="snappy",
                   row_group_size=args.row_group_size)

    check = pq.ParquetFile(args.dst)
    ids = shuffled.column("row_id").to_numpy()
    src_ids = table.column("row_id").to_numpy()
    assert sorted(ids.tolist()) == sorted(src_ids.tolist()), "row set changed"
    print(f"{args.dst}: {check.metadata.num_rows:,} rows in "
          f"{check.metadata.num_row_groups} row groups (seed {args.seed})")
    print(f"first 8 row_ids: {ids[:8].tolist()}")
    print(f"languages in the first 2,000 rows: "
          f"{len(set(shuffled.column('first_language')[:2000].to_pylist()))}")


if __name__ == "__main__":
    main()
