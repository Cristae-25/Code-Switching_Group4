#!/usr/bin/env python
"""Switch-F1 summary figure for the 6 script pairs, from the results JSONs.

    ./.venv/bin/python scripts/plot_f1_summary.py --out docs/figures/team_meeting1/f1_summary.png

Left: mean switch F1 over the 6 pairs (the average of per-pair F1, i.e.
``macro_f1``) for every model and training-free baseline, on one 0-1 axis.
Right: a dumbbell per pair, untrained XLM-R (random heads, 0 steps) to
fine-tuned XLM-R, so the gain from training reads as the length of each bar.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Chart chrome and the sequential blue, from the dataviz reference palette (as plot_confusion.py).
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
BLUE, BLUE_LIGHT, NEUTRAL = "#2a78d6", "#86b6ef", "#c3c2b7"

BASELINES = "data/baseline_anticipation_subword_6pairs.json"
RUN = "data/anticipation_{}.json"
# (label, source, key, is XLM-R): a run tag, or the baseline file and its key.
MODELS = [
    ("Untrained XLM-R (random heads)", "xlmr_untrained_6pairs", None, True),
    ("Always switch", BASELINES, "always", False),
    ("Naive (current language)", BASELINES, "naive_l2", False),
    ("Run-length hazard", BASELINES, "hazard_pair@tuned", False),
    ("Piece lookup", BASELINES, "piece_lookup@tuned", False),
    ("Fine-tuned XLM-R", "xlmr_6pairs", None, True),
    ("Fine-tuned XLM-R, balanced duration", "xlmr_durw_6pairs", None, True),
]


def load(src: str, key: str | None) -> dict:
    if key is None:
        return json.load(open(RUN.format(src)))
    return json.load(open(src))[key]


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8.5, length=0)
    ax.tick_params(axis="y", colors=INK_2, labelsize=9.5, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures/team_meeting1/f1_summary.png")
    args = ap.parse_args()

    rows = [(label, load(src, key)["macro_f1"], xlmr) for label, src, key, xlmr in MODELS]
    rows.sort(key=lambda r: r[1])
    untrained, tuned = load("xlmr_untrained_6pairs", None)["per_pair"], load("xlmr_6pairs", None)["per_pair"]
    pairs = sorted(tuned, key=lambda p: tuned[p]["f1"])

    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.9), facecolor=SURFACE,
                                      gridspec_kw=dict(width_ratios=[1, 1], wspace=0.62))
    for ax in (left, right):
        style(ax)

    # Left: mean F1 per model.
    y = range(len(rows))
    left.barh(y, [r[1] for r in rows], height=0.62, color=[BLUE if r[2] else NEUTRAL for r in rows],
              edgecolor=SURFACE, linewidth=2)
    left.set_yticks(list(y), [r[0] for r in rows])
    for i, (_, f1, xlmr) in enumerate(rows):
        left.text(f1 + 0.015, i, f"{f1:.3f}", va="center", fontsize=9,
                  color=INK if xlmr else INK_2, fontweight="bold" if f1 > 0.6 else "normal")
    left.set_xlabel("Mean switch F1 over the 6 pairs", color=MUTED, fontsize=9, labelpad=6)
    left.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE, label="XLM-R"),
                         plt.Rectangle((0, 0), 1, 1, color=NEUTRAL, label="Training-free baseline")],
                loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK_2)

    # Right: per-pair dumbbells, untrained -> fine-tuned.
    for i, p in enumerate(pairs):
        a, b = untrained[p]["f1"], tuned[p]["f1"]
        right.plot([a, b], [i, i], color=BASELINE, linewidth=2, zorder=1, solid_capstyle="round")
        right.scatter([a], [i], s=70, color=BLUE_LIGHT, edgecolor=SURFACE, linewidth=2, zorder=2)
        right.scatter([b], [i], s=70, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
        right.text(a - 0.02, i, f"{a:.3f}", ha="right", va="center", fontsize=9, color=INK_2)
        right.text(b + 0.02, i, f"{b:.3f}", ha="left", va="center", fontsize=9, color=INK,
                   fontweight="bold")
        right.text((a + b) / 2, i + 0.22, f"+{b - a:.3f}", ha="center", va="bottom", fontsize=8,
                   color=MUTED)
    right.set_yticks(range(len(pairs)), [p.replace("->", " → ") for p in pairs])
    right.set_ylim(-0.6, len(pairs) - 0.3)
    right.set_xlabel("Switch F1 (test split)", color=MUTED, fontsize=9, labelpad=6)
    right.legend(handles=[plt.Line2D([], [], marker="o", ls="", ms=8, color=BLUE_LIGHT,
                                     label="Untrained XLM-R"),
                          plt.Line2D([], [], marker="o", ls="", ms=8, color=BLUE,
                                     label="Fine-tuned XLM-R (12k steps)")],
                 loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False,
                 fontsize=8.5, labelcolor=INK_2)

    top = left.get_position().y1
    fig.text(left.get_position().x0 - 0.2, top + 0.11, "Switch prediction on the 6 script pairs",
             color=INK, fontsize=13, fontweight="bold", ha="left")
    for ax, t in ((left, "All models · mean F1 over pairs"),
                  (right, "Per pair · untrained → fine-tuned XLM-R")):
        fig.text(ax.get_position().x0 - (0.2 if ax is left else 0.13), top + 0.04, t,
                 color=INK, fontsize=10.5, ha="left")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
