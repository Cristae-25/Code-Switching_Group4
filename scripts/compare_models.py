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
    "data/baseline_anticipation_roberta.json:hazard_pair@tuned:hazard, per pair (RoBERTa pieces)",
    "data/baseline_anticipation_mbert.json:hazard_pair@tuned:hazard, per pair (mBERT pieces)",
    "data/anticipation_word.json",
    "data/anticipation_model_subword.json",
    "data/anticipation_xlmr.json",
    "data/anticipation_xlmr_zeroshot.json",
    "data/anticipation_roberta.json",
    "data/anticipation_mbert.json",
    "data/anticipation_model_subword_zeroshot.json",
]


def _summary(label: str, rep: dict) -> dict:
    d = rep.get("per_direction", {})
    dur = rep.get("duration") or {}
    sh = rep.get("shared")
    return dict(label=label, f1=rep["overall"]["f1"], macro_f1=rep["macro_f1"],
                shared_f1=sh["overall"]["f1"] if sh else None,
                shared_macro_f1=sh["macro_f1"] if sh else None,
                shared_per_pair={k: v["f1"] for k, v in sh["per_pair"].items()} if sh else {},
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
        trained_on = [x for x in (data.get("args") or {}).get("train_langs", "").split(",") if x]
        if trained_on:                                   # a zero-shot run: say what it saw
            label = label[:-1] + f", trained on {len(trained_on)} pairs)"
        runs.append(_summary(label, data))
        if "zero_shot" in data:
            runs.append(_summary(label[:-1] + ", held-out pairs)", data["zero_shot"]))
    labels = [r["label"] for r in runs]
    dup = {x for x in labels if labels.count(x) > 1}
    if dup:
        raise SystemExit(f"two result files share a label, which would merge them: {sorted(dup)}")
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
             "**Shared F1** is scored only at the last piece of each non-CJK word, positions that "
             "every tokenizer (and the word model) has in common. It is the fair comparison "
             "between models with different tokenizers. It is empty for runs made before it "
             "existed.", "",
             "| Model | Switch F1 | Macro-F1 over pairs | Shared F1 (non-CJK words) | F1 into L2 | F1 into L1 | Duration acc | Duration macro-F1 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in runs:
        lines.append(f"| {r['label']} | {_f(r['f1'])} | {_f(r['macro_f1'])} | {_f(r['shared_f1'])} | "
                     f"{_f(r['into_l2'])} | {_f(r['into_l1'])} | {_f(r['dur_acc'])} | "
                     f"{_f(r['dur_macro_f1'])} |")
    lines += ["", "## Switch F1 by language pair", "",
              "Pairs are sorted from easiest to hardest (mean F1 across the models shown). "
              "**Bold** marks the best model for each pair.", "",
              "| Pair | " + " | ".join(table.columns) + " |",
              "|---|" + "---:|" * len(table.columns)]
    for pair, row in table.iterrows():
        best = row.max()
        cells = [f"**{v:.3f}**" if v == best else _f(v) for v in row]
        lines.append(f"| {pair} | " + " | ".join(cells) + " |")
    shared = [r for r in runs if r["shared_per_pair"]]
    if shared:
        st = pd.DataFrame({r["label"]: r["shared_per_pair"] for r in shared})
        lines += ["", "## Shared positions by language pair", "",
                  "Switch F1 on the last piece of each non-CJK word (the same words for every model).", "",
                  "| Pair | " + " | ".join(st.columns) + " |", "|---|" + "---:|" * len(st.columns)]
        for pair, row in st.loc[st.mean(axis=1).sort_values(ascending=False).index].iterrows():
            best = row.max()
            lines.append(f"| {pair} | " + " | ".join(f"**{v:.3f}**" if v == best else _f(v)
                                                     for v in row) + " |")
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
    with open(args.md, "w") as fh:
        fh.write(md)
    with open(args.json, "w") as fh:
        json.dump(dict(runs=runs, pair_table=table.to_dict()), fh, indent=2, default=float)
    print(md)
    print(f"wrote {args.md} and {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
