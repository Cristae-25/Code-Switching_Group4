#!/usr/bin/env python
"""Confusion-matrix figures, one PNG per language pair, from a results JSON.

    ./.venv/bin/python scripts/plot_confusion.py data/anticipation_xlmr_6pairs.json \
        --out docs/figures/team_meeting1

Each figure has two panels: the switch matrix (2x2, every scored test
position) and the duration matrix (3x3, true switch points; the val-tuned
decision when the file has one). Colour is the share of each row, i.e. the
recall of that actual class, on one sequential blue ramp shared by every
panel. Raw counts would not work as a colour scale: "no switch, predicted no
switch" is about 50x every other cell. Each cell prints its count and share.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Sequential blue (light -> dark) and chart chrome, from the dataviz reference palette.
BLUES = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
         "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
CMAP = LinearSegmentedColormap.from_list("seq_blue", BLUES)
DUR = ["Small\n(1–2)", "Medium\n(3–6)", "Large\n(7+)"]


def _luminance(hex_rgb) -> float:
    r, g, b = (np.array(hex_rgb[:3]) <= 0.04045) * (np.array(hex_rgb[:3]) / 12.92) + \
        (np.array(hex_rgb[:3]) > 0.04045) * (((np.array(hex_rgb[:3]) + 0.055) / 1.055) ** 2.4)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def panel(ax, counts, labels):
    counts = np.asarray(counts, dtype=float)
    share = counts / np.maximum(counts.sum(1, keepdims=True), 1)
    im = ax.imshow(share, cmap=CMAP, vmin=0, vmax=1, aspect="auto")
    n = len(labels)
    ax.set_xticks(range(n), labels, color=INK_2, fontsize=9)
    ax.set_yticks(range(n), labels, color=INK_2, fontsize=9)
    ax.set_xlabel("Predicted", color=MUTED, fontsize=9, labelpad=6)
    ax.set_ylabel("Actual", color=MUTED, fontsize=9, labelpad=6)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # 2px surface gap between cells
    ax.set_xticks(np.arange(-0.5, n), minor=True)
    ax.set_yticks(np.arange(-0.5, n), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=3)
    ax.tick_params(which="minor", length=0)
    for i in range(n):
        for j in range(n):
            dark_cell = _luminance(CMAP(share[i, j])) < 0.18
            ax.text(j, i, f"{int(counts[i, j]):,}\n{share[i, j]:.0%}", ha="center", va="center",
                    fontsize=9, color=SURFACE if dark_cell else INK,
                    fontweight="bold" if i == j else "normal")
    return im


def figure(pair: str, sw: dict, dur: dict, dur_label: str, path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), facecolor=SURFACE,
                             gridspec_kw=dict(width_ratios=[2, 3], wspace=0.28))
    for ax in axes:
        ax.set_facecolor(SURFACE)
    switch = [[sw["tp"], sw["fn"]], [sw["fp"], sw["tn"]]]
    panel(axes[0], switch, ["Switch", "No switch"])
    im = panel(axes[1], dur["confusion"], DUR)
    titles = (f"Switch  ·  F1 {sw['f1']:.3f}   P {sw['precision']:.3f}   R {sw['recall']:.3f}",
              f"Duration ({dur_label})  ·  macro-F1 {dur['macro_f1']:.3f}   acc {dur['accuracy']:.3f}")
    top = axes[0].get_position().y1
    for ax, t in zip(axes, titles):                 # one baseline for both, above the tick labels
        fig.text(ax.get_position().x0, top + 0.17, t, color=INK, fontsize=10.5, ha="left")
    fig.text(axes[0].get_position().x0, top + 0.26, pair.replace("->", " → "), color=INK,
             fontsize=13, fontweight="bold", ha="left")
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.03)
    cb.set_label("Share of the actual row (recall)", color=MUTED, fontsize=8.5)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, labelsize=8, length=0)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1], labels=["0%", "25%", "50%", "75%", "100%"])
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--raw", action="store_true", help="use the raw argmax duration, not val-tuned")
    args = ap.parse_args()
    r = json.load(open(args.results))
    tuned = (r.get("duration_tuned") or {}).get("report", {}).get("duration_per_pair")
    dur_pp, label = (r["duration_per_pair"], "raw") if args.raw or not tuned else (tuned, "val-tuned")
    os.makedirs(args.out, exist_ok=True)
    for pair, sw in r["per_pair"].items():
        path = os.path.join(args.out, f"{pair.split('->')[0].lower()}.png")
        figure(pair, sw, dur_pp[pair], label, path)
        print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
