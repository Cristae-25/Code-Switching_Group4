#!/usr/bin/env python
"""One language pair's results page: every model, and (optionally) its actual errors.

    ./.venv/bin/python scripts/pair_report.py --pair "German->French"
    ./.venv/bin/python scripts/pair_report.py --pair "German->French" \\
        --preds data/preds/ablation_combined.npz --md docs/pairs/German-French.md

Section 1 reads every result file that scripts/compare_models.py knows about
and shows this pair's switch P/R/F1 (on each model's own pieces) and its shared
F1 (the same words for every tokenizer).

Section 2, for each --preds file (written by train_anticipation.py --save-preds),
counts hits and errors by direction at the val-tuned threshold and prints the
most confident missed switches and false alarms as text. ``┃`` marks the
prediction point: the model has read everything left of it and is asked whether
the next language-bearing piece switches. Error text comes from the pieces in
--subword (``data/subword.parquet``, XLM-R). Use the matching file for runs on
another tokenizer, and note that positions from --turn-sep runs include the
inserted separators.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from compare_models import DEFAULT_SPECS, iter_reports  # noqa: E402
from tune_thresholds import _best_threshold  # noqa: E402
from src.evaluation import DIRECTIONS, PAIR_KEYS  # noqa: E402


def _f(x) -> str:
    return "" if x is None else f"{x:.3f}"


def results_table(pair: str, specs) -> list[str]:
    lines = ["| Model | Switch F1 | P | R | Test positions | Shared F1 |", "|---|---:|---:|---:|---:|---:|"]
    for label, rep in iter_reports(specs):
        v = rep.get("per_pair", {}).get(pair)
        if v is None:
            continue
        sh = (rep.get("shared") or {}).get("per_pair", {}).get(pair, {})
        lines.append(f"| {label} | {_f(v['f1'])} | {_f(v.get('precision'))} | {_f(v.get('recall'))} | "
                     f"{v.get('n', 0):,} | {_f(sh.get('f1'))} |")
    return lines


def _window(pieces, pos: int, before: int = 12, after: int = 4) -> str:
    text = lambda ps: "".join(p.replace("▁", " ").replace("Ġ", " ") for p in ps).strip()  # noqa: E731
    return f"{text(pieces[max(0, pos - before):pos + 1])} ┃ {text(pieces[pos + 1:pos + 1 + after])}"


def error_section(pair: str, path: str, subword: str, examples: int) -> list[str]:
    z = np.load(path)
    p_idx = PAIR_KEYS.index(pair)
    pv = 1 / (1 + np.exp(-z["val_logit"]))
    tau = _best_threshold(pv, z["val_y"])
    m = z["test_pair"] == p_idx
    prob = 1 / (1 + np.exp(-z["test_logit"][m]))
    y, lang, pred = z["test_y"][m], z["test_lang"][m], prob >= tau
    lines = [f"### Errors: `{os.path.basename(path)}` (threshold {tau:.3f}, tuned on val)", "",
             "| Direction | Positions | Switches | Caught | Missed switches | False alarms | F1 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for k, name in DIRECTIONS.items():
        d = lang == k
        tp, fn, fp = int((y[d] == 1)[pred[d]].sum()), int(((y[d] == 1) & ~pred[d]).sum()), int(((y[d] == 0) & pred[d]).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        lines.append(f"| {name} | {int(d.sum()):,} | {int((y[d] == 1).sum()):,} | {tp:,} | {fn:,} | {fp:,} | {f1:.3f} |")
    if "test_row" not in z.files:
        lines += ["", "*No error examples: this file has no row/position information; "
                  "re-run with the current --save-preds to include them.*"]
        return lines
    row, pos = z["test_row"][m], z["test_pos"][m]
    fn_i = np.flatnonzero((y == 1) & ~pred)
    fp_i = np.flatnonzero((y == 0) & pred)
    fn_i = fn_i[np.argsort(prob[fn_i])][:examples]               # most confident misses first
    fp_i = fp_i[np.argsort(-prob[fp_i])][:examples]              # most confident false alarms first
    need = sorted({int(r) for r in np.r_[row[fn_i], row[fp_i]]})
    if not need:
        return lines + ["", "*No errors to show.*"]
    df = pd.read_parquet(subword, columns=["row_id", "pieces"], filters=[("row_id", "in", need)])
    pieces = {int(r): list(p) for r, p in zip(df.row_id, df.pieces)}
    for title, idx in (("Missed switches (the next word switched; the model said no)", fn_i),
                       ("False alarms (the model said switch; it didn't)", fp_i)):
        lines += ["", f"**{title}**", ""]
        for i in idx:
            r, q = int(row[i]), int(pos[i])
            if r in pieces:
                lines.append(f"- row {r}, p={prob[i]:.2f} ({DIRECTIONS[int(lang[i])]}): `{_window(pieces[r], q)}`")
    return lines


def pair_report(pair: str, specs, preds, subword: str, examples: int = 5) -> str:
    if pair not in PAIR_KEYS:
        raise SystemExit(f"unknown pair {pair!r}; one of: {', '.join(PAIR_KEYS)}")
    lines = [f"# {pair}", "", "## All models (test split)", "",
             "Switch F1 is on each model's own pieces (tokenizers differ). Shared F1 is on the same "
             "words for every model, the fair cross-model number.", ""]
    lines += results_table(pair, specs)
    for path in preds:
        lines += [""] + error_section(pair, path, subword, examples)
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, help='e.g. "German->French"')
    ap.add_argument("--specs", nargs="*", default=DEFAULT_SPECS, help="result files (as for compare_models)")
    ap.add_argument("--preds", nargs="*", default=[], help=".npz files from --save-preds")
    ap.add_argument("--subword", default="data/subword.parquet", help="pieces for the error text")
    ap.add_argument("--examples", type=int, default=5)
    ap.add_argument("--md", default=None, help="also write the page here (e.g. docs/pairs/<pair>.md)")
    args = ap.parse_args()
    md = pair_report(args.pair, args.specs, args.preds, args.subword, args.examples)
    print(md)
    if args.md:
        os.makedirs(os.path.dirname(args.md) or ".", exist_ok=True)
        with open(args.md, "w") as fh:
            fh.write(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
