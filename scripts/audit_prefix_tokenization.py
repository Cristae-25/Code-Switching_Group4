#!/usr/bin/env python
"""Audit the pieces for a tokenization leak: would the prefix alone produce them?

    ./.venv/bin/python scripts/audit_prefix_tokenization.py
    ./.venv/bin/python scripts/audit_prefix_tokenization.py --preds data/preds_xlmr_ctx1.npz

``data/subword.parquet`` was built by tokenizing each whole utterance. A model
that really streams only has the prefix, so for every scored piece this
re-tokenizes ``text[:end of piece]`` and checks that it reproduces the stored
piece (``last_ok``: what a current-token model reads) and every piece before it
(``all_ok``: what a full-prefix model reads). It prints the mismatch rate per
pair and writes ``data/prefix_consistency.parquet`` (one flag list per row,
aligned with ``piece_ids``).

With ``--preds`` (an .npz from ``train_anticipation.py --save-preds``), it also
re-scores the test predictions on the consistent positions only, using the
threshold tuned on val. If the scores barely move, no result depends on the leak.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from run_tagging import load_corpus  # noqa: E402
from src.evaluation import full_report, format_report  # noqa: E402
from src.segments import parse_utterances  # noqa: E402
from src.streaming import iter_rows  # noqa: E402
from src.subword import prefix_consistency  # noqa: E402

SCRIPT_PAIRS = ["Cantonese", "Chinese", "Japanese", "Korean", "Russian", "Arabic"]


def audit_row(utterances, piece_ids, tokenizer):
    """``(last_ok, all_ok)`` per stored piece, or None if re-tokenizing disagrees."""
    ids, last_ok, all_ok = [], [], []
    for text in utterances:
        if not text:
            continue
        i, lo, ao = prefix_consistency(text, tokenizer)
        ids += i
        last_ok += lo
        all_ok += ao
    if ids != list(piece_ids):
        return None
    return last_ok, all_ok


def rescore(preds_path: str, flags: dict, tune) -> None:
    z = np.load(preds_path)
    val = {k: z[f"val_{k}"] for k in ("logit", "y")}
    tau, _ = tune(1 / (1 + np.exp(-val["logit"])), val["y"])
    t = {k: z[f"test_{k}"] for k in ("logit", "y", "pair", "lang", "row", "pos")}
    keep = np.array([flags[int(r)][int(p)] if int(r) in flags else True
                     for r, p in zip(t["row"], t["pos"])], dtype=bool)
    yp = (1 / (1 + np.exp(-t["logit"])) >= tau).astype(np.int64)
    print(f"\n{preds_path}: tau={tau:.3f} (tuned on val) | "
          f"{(~keep).sum():,} of {len(keep):,} test positions inconsistent")
    for name, m in (("all positions", np.ones_like(keep)), ("consistent positions only", keep)):
        rep = full_report(t["y"][m], yp[m], t["pair"][m], t["lang"][m])
        print("\n" + format_report(rep, f"TEST - {name}"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="data/subword.parquet")
    ap.add_argument("--tokenizer", default="xlm-roberta-base")
    ap.add_argument("--langs", default=",".join(SCRIPT_PAIRS))
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--limit", type=int, default=0, help="rows per split (0 = all)")
    ap.add_argument("--out", default="data/prefix_consistency.parquet")
    ap.add_argument("--preds", nargs="*", default=[], help="saved predictions to re-score")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    cells = load_corpus()["data_generation_result"]

    t0, out, stats = time.time(), [], {}
    cols = ["row_id", "first_language", "piece_ids", "y_sw"]
    for split in args.splits.split(","):
        for i, row in enumerate(iter_rows(args.path, split=split, languages=langs, columns=cols)):
            if args.limit and i >= args.limit:
                break
            res = audit_row(parse_utterances(cells.iloc[int(row["row_id"])]), row["piece_ids"], tok)
            s = stats.setdefault(row["first_language"], dict(rows=0, failed=0, scored=0,
                                                             bad_last=0, bad_all=0))
            s["rows"] += 1
            if res is None:
                s["failed"] += 1
                continue
            last_ok, all_ok = res
            scored = np.asarray(row["y_sw"]) >= 0
            s["scored"] += int(scored.sum())
            s["bad_last"] += int((~np.asarray(last_ok) & scored).sum())
            s["bad_all"] += int((~np.asarray(all_ok) & scored).sum())
            out.append(dict(row_id=int(row["row_id"]), last_ok=last_ok, all_ok=all_ok))

    print(f"audited {sum(s['rows'] for s in stats.values()):,} rows in {time.time() - t0:.0f}s "
          f"({args.splits}; {args.tokenizer})\n")
    print(f"{'pair':<12} {'rows':>7} {'unaligned':>9} {'scored':>11} "
          f"{'last piece differs':>19} {'some prefix piece differs':>26}")
    tot = dict(scored=0, bad_last=0, bad_all=0)
    for lang in langs:
        s = stats.get(lang)
        if not s:
            continue
        for k in tot:
            tot[k] += s[k]
        n = max(s["scored"], 1)
        print(f"{lang:<12} {s['rows']:>7,} {s['failed']:>9,} {s['scored']:>11,} "
              f"{s['bad_last']:>9,} ({s['bad_last'] / n:.4%}) {s['bad_all']:>14,} ({s['bad_all'] / n:.4%})")
    n = max(tot["scored"], 1)
    print(f"{'all':<12} {'':>7} {'':>9} {tot['scored']:>11,} {tot['bad_last']:>9,} "
          f"({tot['bad_last'] / n:.4%}) {tot['bad_all']:>14,} ({tot['bad_all'] / n:.4%})")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pd.DataFrame(out).to_parquet(args.out, index=False)
    print(f"\nwrote {args.out}")

    if args.preds:
        from train_anticipation import tune_threshold
        flags = {r["row_id"]: r["last_ok"] for r in out}
        for p in args.preds:
            rescore(p, flags, tune_threshold)


if __name__ == "__main__":
    main()
