#!/usr/bin/env python
"""Recompute the names-policy columns of a tagged parquet and re-export the CSV.

Only tokens/token_lang/is_entity are read, no model runs, so this takes about
a minute for the full corpus. ``--from-v1`` first converts v1 labels (names as
``NE``) with ``persist.from_v1``.

    ./.venv/bin/python scripts/apply_names_policy.py data/hf/tagged.parquet --from-v1 \
        --out data/final/tagged.parquet
    # -> writes the parquet and switchlingua_tagged_rows.csv beside it
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("--out", default=None, help="output parquet (default: overwrite input)")
    ap.add_argument("--csv", default=None, help="output CSV (default: beside the parquet)")
    ap.add_argument("--from-v1", action="store_true",
                    help="input is v1 (names labelled NE): convert with persist.from_v1")
    args = ap.parse_args()

    import numpy as np
    import pandas as pd

    from src.persist import derive_name_policies, export_rows_csv, from_v1, save_tagged

    t0 = time.time()
    frame = pd.read_parquet(args.parquet)
    if args.from_v1:
        frame = from_v1(frame)
    elif "is_entity" not in frame.columns:
        raise SystemExit("no is_entity column - pass --from-v1 if names are labelled NE")
    else:
        frame = derive_name_policies(frame)
    out = save_tagged(frame, args.out or args.parquet)
    csv = export_rows_csv(frame, args.csv or os.path.join(os.path.dirname(out),
                                                          "switchlingua_tagged_rows.csv"))
    print(f"{len(frame):,} rows in {time.time() - t0:.0f}s -> {out}, {csv}\n")

    rows = []
    for (l1, l2), g in frame.groupby(["first_language", "second_language"]):
        d = {"pair": f"{l1}->{l2}"}
        for col, name in (("l2_runs", "default"), ("l2_runs_transparent", "transparent"),
                          ("l2_runs_script", "script")):
            r = [x for v in g[col] for x in v]
            d[f"switches_{name}"] = len(r)
            d[f"median_{name}"] = float(np.median(r)) if r else 0.0
        rows.append(d)
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
