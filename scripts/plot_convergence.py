"""Learning curves of one run (switch vs duration) from its `--eval-every` log.

    ./.venv/bin/python scripts/plot_convergence.py xlmr_prev_gap_36k_rs --out docs/figures/chris/gap_36k

Reads `data/anticipation_<TAG>.json` -> `curve` (a fixed val slice scored every N steps)
and writes `<out>/convergence.png`: four small multiples sharing one step axis.
Switch: val F1 and val loss. Duration: val macro-F1 (raw, no tuned offsets). Total:
the training loss L_sw + lambda_2 * L_dur over the preceding N steps. Dotted lines mark
epoch ends (11,711 steps at batch 16 on all 15 pairs).
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EPOCH = 11_711
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SW, DUR, TOTAL = "#2a78d6", "#eb6834", "#52514e"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()
    d = json.load(open(f"data/anticipation_{args.tag}.json"))
    c = d["curve"]
    if not c:
        raise SystemExit("no learning curve: the run needs --eval-every")
    steps = [p["step"] for p in c]
    last = d["steps"]
    final = d.get("val_f1")          # val F1 at the end of training (full val split)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, facecolor=SURFACE)
    panels = [
        (axes[0, 0], [p["val_f1"] for p in c], SW, "Switch: val F1 (higher is better)", "{:.3f}"),
        (axes[0, 1], [p["val_switch_loss"] for p in c], SW, "Switch: val loss (BCE)", "{:.3f}"),
        (axes[1, 0], [p["val_dur_macro_f1"] for p in c], DUR,
         "Duration: val macro-F1, 3 classes (raw)", "{:.3f}"),
        (axes[1, 1], [p["train_loss"] for p in c], TOTAL,
         "Total training loss  L_sw + 0.3·L_dur", "{:.3f}"),
    ]
    for ax, ys, col, title, fmt in panels:
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for e in range(1, int(last // EPOCH) + 1):
            ax.axvline(e * EPOCH, color=INK2, lw=0.8, ls=":", zorder=0)
        ax.plot(steps, ys, color=col, lw=2, marker="o", ms=5, mec=SURFACE, mew=1.5, zorder=3)
        ax.annotate(fmt.format(ys[-1]), (steps[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=9)
        ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")
        lo, hi = min(ys), max(ys)
        pad = (hi - lo) * 0.25 or 0.01
        ax.set_ylim(lo - pad, hi + pad)
    for ax in axes[1]:
        ax.set_xlabel("training step")
        ax.set_xticks(range(0, last + 1, 6000))
        ax.set_xticklabels([f"{s // 1000}k" for s in range(0, last + 1, 6000)])
        ax.set_xlim(0, last * 1.07)
    for ax in axes[0]:
        top = ax.get_ylim()[1]
        for e in range(1, int(last // EPOCH) + 1):
            ax.text(e * EPOCH, top, f" end of epoch {e}", color=INK2, fontsize=8, va="top", ha="left")

    # Where the switch-F1 gain arrives: after epoch 1, by epoch 2, and in epoch 3.
    ax = axes[0, 0]
    f1 = dict(zip(steps, [p["val_f1"] for p in c]))
    if all(k in f1 for k in (12_000, 24_000)):
        a, b, z = f1[12_000], f1[24_000], f1[steps[-1]]
        ax.text(18_000, a - 0.004, f"+{b - a:.3f}  (12k→24k)", color=INK, fontsize=9, ha="center", va="top")
        ax.text(30_000, z - 0.004, f"+{z - b:.4f}  (24k→{steps[-1] // 1000}k)", color=INK, fontsize=9,
                ha="center", va="top")
        ax.axvspan(24_000, steps[-1], color=SW, alpha=0.06, lw=0)
    dax = axes[1, 0]
    dax.text(0.03, 0.94, "noisy until ~21k, then flat", transform=dax.transAxes, ha="left", va="top",
             color=INK2, fontsize=9)
    fig.suptitle(f"{args.tag}: performance plateaus after epoch 2", x=0.01, ha="left", y=0.985,
                 color=INK, fontsize=13, fontweight="bold")
    fig.text(0.01, 0.925, f"Fixed val slice, {c[0]['n']:,} positions, scored every "
             f"{steps[1] - steps[0]:,} steps. Training loss is the mean of the preceding window.",
             color=INK2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.915))
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "convergence.png")
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    print("wrote", path, "| final full-val F1:", final)


if __name__ == "__main__":
    main()
