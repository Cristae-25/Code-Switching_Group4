#!/usr/bin/env python
"""Per-pair F1 bar charts for the all-15-pair XLM-R runs: switch and duration.

    ./.venv/bin/python scripts/plot_pair_f1.py --out docs/figures/chris/all_pairs
    ./.venv/bin/python scripts/plot_pair_f1.py --runs backbones --out docs/figures/chris/mbert

Two figures, one bar per pair and run (untrained, linear probe, fine-tuned),
pairs sorted by the fine-tuned score, with a side note naming the best and
worst pairs:

    switch_f1.png    switch F1 on the test split
    duration_f1.png  duration macro-F1 (Small / Medium / Large) at the true
                     switch points, with the val-tuned decision

The runs are the ``*_12lang.json`` files (they cover all 15 pairs; the name
predates the run). Untrained and probe use a frozen backbone; the probe
trained its heads for 12,000 steps, the fine-tuned run 17,500 with balanced
duration weights, so "probe vs fine-tuned" also differs in training length.

``--runs backbones`` instead compares the fine-tuned XLM-R and mBERT runs
(identical settings: 17,500 steps, balanced duration weights, same test rows),
sorted and annotated by mBERT. The two tokenizers cut the text differently, so
the position counts differ; ``bootstrap_preds.py`` gives the paired row-level
interval of the difference.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

# Chart chrome and the sequential blue, from the dataviz reference palette (as plot_confusion.py).
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
# untrained -> probe -> fine-tuned is an order (more of the model trained), so one hue, light to dark.
# Two backbones are two identities, not an order: categorical slots 1 and 2 (as plot_direction_f1.py).
# Each set is (runs, the run that sorts the bars and fills the side note).
RUN_SETS = {
    "xlmr": ({"Untrained XLM-R": ("xlmr_untrained_12lang", "#b7d3f6"),
              "Linear probe (12k steps)": ("xlmr_probe_12lang", "#5598e7"),
              "Fine-tuned XLM-R (17.5k steps)": ("xlmr_durw_12lang", "#184f95")},
             "Fine-tuned XLM-R (17.5k steps)"),
    "backbones": ({"Fine-tuned XLM-R (17.5k steps)": ("xlmr_durw_12lang", "#2a78d6"),
                   "Fine-tuned mBERT (17.5k steps)": ("mbert_durw", "#eb6834")},
                  "Fine-tuned mBERT (17.5k steps)"),
}
RUNS, FINE = RUN_SETS["xlmr"]


def pair_name(p: str) -> str:
    l1, l2 = p.split("->")
    l1 = "Tagalog" if l1 == "Philippines" else l1
    return l1 if l2 == "English" and l1 != "German" else f"{l1} → {l2}"   # two German pairs


def scores(metric: str) -> pd.DataFrame:
    rows = []
    for label, (tag, _) in RUNS.items():
        r = json.load(open(f"data/anticipation_{tag}.json"))
        if metric == "switch":
            per = {p: v["f1"] for p, v in r["per_pair"].items()}
        else:
            dpp = (r.get("duration_tuned") or {}).get("report", {}).get("duration_per_pair") \
                or r["duration_per_pair"]
            per = {p: v["macro_f1"] for p, v in dpp.items()}
        rows += [dict(pair=pair_name(p), run=label, f1=f) for p, f in per.items()]
    return pd.DataFrame(rows)


def side_note(ax, df: pd.DataFrame, mean: float) -> None:
    ft = df[df.run == FINE].sort_values("f1", ascending=False)
    best, worst = ft.head(3), ft.tail(3).iloc[::-1]
    who = FINE.split(" (")[0]
    fmt = lambda d: "\n".join(f"{r.pair:<18}{r.f1:.3f}" for r in d.itertuples())
    spread = ft.f1.iloc[0] - ft.f1.iloc[-1]
    ax.axis("off")
    y = 0.98
    for title, body in ((f"{who}, best", fmt(best)), (f"{who}, worst", fmt(worst)),
                        ("Mean over pairs", f"{mean:.3f}"),
                        ("Best − worst", f"{spread:.3f}")) + versus(df):
        ax.text(0, y, title, color=MUTED, fontsize=9, va="top", transform=ax.transAxes)
        ax.text(0, y - 0.045, body, color=INK, fontsize=9.5, va="top", family="monospace",
                linespacing=1.5, transform=ax.transAxes)
        y -= 0.115 + 0.034 * body.count("\n")


def versus(df: pd.DataFrame) -> tuple:
    """With exactly two runs: how often and by how much the focus run beats the other."""
    if len(RUNS) != 2:
        return ()
    other = next(k for k in RUNS if k != FINE)
    w = df.pivot(index="pair", columns="run", values="f1")
    d = w[FINE] - w[other]
    name = other.split(" (")[0].replace("Fine-tuned ", "")
    return ((f"Beats {name} in", f"{(d > 0).sum()} of {len(d)} pairs"),
            (f"Mean gap vs {name}", f"{d.mean():+.3f}"),
            ("Largest gaps", f"{d.idxmax():<18}{d.max():+.3f}\n{d.idxmin():<18}{d.min():+.3f}"))


def figure(metric: str, title: str, xlabel: str, path: str) -> None:
    df = scores(metric)
    order = df[df.run == FINE].sort_values("f1", ascending=False).pair.tolist()
    mean = df[df.run == FINE].f1.mean()
    sns.set_theme(style="whitegrid", rc={"axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
                                         "grid.color": GRID, "axes.edgecolor": BASELINE})
    fig, (ax, note) = plt.subplots(1, 2, figsize=(12, 8.6), facecolor=SURFACE,
                                   gridspec_kw=dict(width_ratios=[3.4, 1], wspace=0.08))
    sns.barplot(data=df, y="pair", x="f1", hue="run", order=order, hue_order=list(RUNS),
                palette={k: c for k, (_, c) in RUNS.items()}, orient="h", ax=ax,
                width=0.8, edgecolor=SURFACE, linewidth=1.5)
    for cont, label in zip(ax.containers, RUNS):              # value labels on the focus run (every run when comparing two)
        if label == FINE or len(RUNS) == 2:
            ax.bar_label(cont, fmt="%.3f", padding=4, fontsize=8.5, color=INK)
    ax.set_xlim(0, 1)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9.5, labelpad=8)
    ax.set_ylabel("")
    ax.tick_params(axis="y", colors=INK_2, labelsize=10, length=0)
    ax.tick_params(axis="x", colors=MUTED, labelsize=9, length=0)
    ax.grid(axis="y", visible=False)
    sns.despine(ax=ax, left=True)
    ax.legend(title=None, frameon=False, fontsize=9, labelcolor=INK_2, ncol=3,
              loc="upper left", bbox_to_anchor=(0, -0.06))
    side_note(note, df, mean)
    fig.suptitle(title, x=0.125, y=0.95, ha="left", color=INK, fontsize=14, fontweight="bold")
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures/chris/all_pairs")
    ap.add_argument("--runs", choices=list(RUN_SETS), default="xlmr")
    args = ap.parse_args()
    global RUNS, FINE
    RUNS, FINE = RUN_SETS[args.runs]
    os.makedirs(args.out, exist_ok=True)
    figure("switch", "Switch prediction F1 by language pair (test split)",
           "Switch F1", os.path.join(args.out, "switch_f1.png"))
    figure("duration", "Switch duration macro-F1 by language pair (test split, true switch points)",
           "Duration macro-F1 (Small 1–2 / Medium 3–6 / Large 7+)",
           os.path.join(args.out, "duration_f1.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
