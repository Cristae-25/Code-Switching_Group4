#!/usr/bin/env python
"""Compare every model's test results, overall and language pair by language pair.

    ./.venv/bin/python scripts/compare_models.py             # all known result files
    ./.venv/bin/python scripts/compare_models.py data/anticipation_xlmr.json \\
        data/baseline_anticipation_subword.json:hazard_pair@tuned:per-pair hazard (pieces)

Each argument is a result file written by train_anticipation.py, or
``baseline_file.json:entry[:label]`` for one baseline inside a baseline file.
Every run was scored with src.evaluation.full_report on the same test split,
so the numbers are directly comparable within a unit. Word and piece positions
differ, so compare across units with care. Writes docs/model_comparison.md and
data/model_comparison.json.
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SPECS = [
    "data/baseline_anticipation.json:hazard@tuned:hazard, pooled (words)",
    "data/baseline_anticipation.json:hazard_pair@tuned:hazard, per pair (words)",
    "data/baseline_anticipation_subword.json:hazard@tuned:hazard, pooled (pieces)",
    "data/baseline_anticipation_subword.json:hazard_pair@tuned:hazard, per pair (pieces)",
    "data/anticipation_word.json",
    "data/anticipation_model_subword.json",
    "data/anticipation_xlmr.json",
    "data/anticipation_xlmr_zeroshot.json",
]


def _summary(label: str, rep: dict) -> dict:
    d = rep.get("per_direction", {})
    dur = rep.get("duration") or {}
    return dict(label=label, f1=rep["overall"]["f1"], macro_f1=rep["macro_f1"],
                into_l2=d.get("into L2", {}).get("f1"), into_l1=d.get("into L1", {}).get("f1"),
                dur_acc=dur.get("accuracy"), dur_macro_f1=dur.get("macro_f1"),
                per_pair={k: v["f1"] for k, v in rep["per_pair"].items()})


def load_runs(specs) -> list[dict]:
    runs = []
    for spec in specs:
        path, *rest = spec.split(":", 2)
        if not os.path.exists(path):
            continue
        data = json.load(open(path))
        if rest:                                          # one entry of a baseline file
            key = rest[0]
            if key not in data:
                continue
            runs.append(_summary(rest[1] if len(rest) > 1 else f"{key} (baseline)", data[key]))
            continue
        if "per_direction" not in data:                  # written before the shared report
            print(f"skipping {path}: old result format (re-run it to include it)")
            continue
        label = f"{data.get('model', 'scratch')} ({data.get('unit', 'word')})"
        runs.append(_summary(label, data))
        if "zero_shot" in data:
            runs.append(_summary(label + ", zero-shot pairs", data["zero_shot"]))
    return runs


def pair_table(runs) -> pd.DataFrame:
    """Switch F1 per pair (rows) and run (columns); easiest pair first."""
    t = pd.DataFrame({r["label"]: r["per_pair"] for r in runs})
    return t.loc[t.mean(axis=1).sort_values(ascending=False).index]


def _f(x) -> str:
    return "" if x is None or pd.isna(x) else f"{x:.3f}"


def to_markdown(runs, table: pd.DataFrame) -> str:
    lines = ["# Model comparison (test split)", "",
             "Every run is scored the same way (`src/evaluation.py`): the switch threshold is "
             "tuned on val and applied unchanged to test; duration is scored at true switch "
             "points whose class is certain. **Words** and **pieces** are different units "
             "(a word is 1.3–1.7 pieces), so compare models within a unit.", "",
             "## Overall", "",
             "| Model | Switch F1 | Macro-F1 over pairs | F1 into L2 | F1 into L1 | Duration acc | Duration macro-F1 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in runs:
        lines.append(f"| {r['label']} | {_f(r['f1'])} | {_f(r['macro_f1'])} | {_f(r['into_l2'])} | "
                     f"{_f(r['into_l1'])} | {_f(r['dur_acc'])} | {_f(r['dur_macro_f1'])} |")
    lines += ["", "## Switch F1 by language pair", "",
              "Pairs are sorted from easiest to hardest (mean F1 across the models shown). "
              "**Bold** marks the best model for each pair.", "",
              "| Pair | " + " | ".join(table.columns) + " |",
              "|---|" + "---:|" * len(table.columns)]
    for pair, row in table.iterrows():
        best = row.max()
        cells = [f"**{v:.3f}**" if v == best else _f(v) for v in row]
        lines.append(f"| {pair} | " + " | ".join(cells) + " |")
    spread = table.max(axis=0) - table.min(axis=0)
    lines += ["", "Spread between the easiest and hardest pair: "
              + ", ".join(f"{c} {s:.3f}" for c, s in spread.items()) + "."]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="*", default=DEFAULT_SPECS)
    ap.add_argument("--md", default=os.path.join(ROOT, "docs", "model_comparison.md"))
    ap.add_argument("--json", default=os.path.join(ROOT, "data", "model_comparison.json"))
    args = ap.parse_args()
    runs = load_runs(args.specs)
    if not runs:
        raise SystemExit("no result files found")
    table = pair_table(runs)
    md = to_markdown(runs, table)
    open(args.md, "w").write(md)
    json.dump(dict(runs=runs, pair_table=table.to_dict()), open(args.json, "w"), indent=2, default=float)
    print(md)
    print(f"wrote {args.md} and {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
