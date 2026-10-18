#!/usr/bin/env python
"""Switch F1 per language pair, split by direction: L1 -> L2 and L2 -> L1.

    ./.venv/bin/python scripts/plot_direction_f1.py \
        --preds data/preds_xlmr_durw.npz --results data/anticipation_xlmr_durw_12lang.json \
        --out docs/figures/all_pairs/direction_f1.png

The results JSON only splits direction pooled over pairs, so this rebuilds the
per-pair split from the saved test predictions (``--save-preds``) and the
run's val-tuned threshold - the same decision rule as ``train_anticipation.score``
(it reproduces the JSON's per-pair F1 exactly).

Direction is the *current* piece's language: an L1 piece followed by an L2
piece is an L1 -> L2 switch ("into L2" in the JSON), and the reverse is
L2 -> L1 ("into L1"). The tick on each bar is blind guessing at that
direction's switch rate: guessing "switch" with probability equal to the base
rate gives precision = recall = F1 = the base rate.

Also writes the numbers next to the figure, as JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation import PAIR_KEYS  # noqa: E402

SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
# Two directions are two identities: categorical slots 1 and 2 of the reference palette.
DIRS = {0: ("L1 → L2 (into the second language)", "#2a78d6"), 1: ("L2 → L1 (back to the first)", "#eb6834")}


def pair_name(p: str) -> str:
    l1, l2 = p.split("->")
    l1 = "Tagalog" if l1 == "Philippines" else l1
    return l1 if l2 == "English" and l1 != "German" else f"{l1} → {l2}"


def scores(preds: str, tau: float) -> dict:
    if os.path.isdir(preds):        # an unzipped .npz: a folder of <name>.npy files
        z = {f[:-4]: np.load(os.path.join(preds, f)) for f in os.listdir(preds) if f.endswith(".npy")}
    else:
        z = np.load(preds)
    y, pair, lang = z["test_y"].astype(bool), z["test_pair"], z["test_lang"]
    fired = 1 / (1 + np.exp(-z["test_logit"])) >= tau
    out = {}
    for p in np.unique(pair):
        out[PAIR_KEYS[int(p)]] = row = {}
        for d in DIRS:
            m = (pair == p) & (lang == d)
            tp, fp, fn = (fired & y & m).sum(), (fired & ~y & m).sum(), (~fired & y & m).sum()
            prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
            row["into L2" if d == 0 else "into L1"] = dict(
                f1=float(2 * tp / max(2 * tp + fp + fn, 1)), precision=float(prec), recall=float(rec),
                switches=int(y[m].sum()), positions=int(m.sum()), base_rate=float(y[m].mean()))
    return out


def figure(s: dict, title: str, path: str) -> None:
    keys = sorted(s, key=lambda p: s[p]["into L2"]["f1"])
    n, h = len(keys), 0.38
    fig, (ax, note) = plt.subplots(1, 2, figsize=(12, 0.52 * n + 2.4), facecolor=SURFACE,
                                   gridspec_kw=dict(width_ratios=[3.4, 1], wspace=0.08))
    ax.set_facecolor(SURFACE)
    for d, (label, color) in DIRS.items():
        key = "into L2" if d == 0 else "into L1"
        yy = np.arange(n) + (h / 2 if d == 0 else -h / 2)
        f1 = [s[p][key]["f1"] for p in keys]
        ax.barh(yy, f1, height=h, color=color, edgecolor=SURFACE, linewidth=1.5, label=label)
        for yi, v in zip(yy, f1):
            ax.text(v + 0.008, yi, f"{v:.3f}", va="center", fontsize=8, color=INK)
        base = [s[p][key]["base_rate"] for p in keys]       # blind guessing, per bar
        ax.scatter(base, yy, marker="|", s=110, color=INK, linewidths=1.6, zorder=3)
    ax.scatter([], [], marker="|", s=110, color=INK, linewidths=1.6, label="Blind guess (F1 = switch rate)")
    ax.set_yticks(range(n), [pair_name(p) for p in keys])
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("Switch F1 (test split)", color=MUTED, fontsize=9.5, labelpad=8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(axis="y", colors=INK_2, labelsize=10, length=0)
    ax.tick_params(axis="x", colors=MUTED, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, ncol=3, loc="upper left",
              bbox_to_anchor=(0, -0.6 / (0.52 * n + 1.2)))

    into2 = np.array([s[p]["into L2"]["f1"] for p in keys])
    into1 = np.array([s[p]["into L1"]["f1"] for p in keys])
    gap = into1 - into2
    lines = [("L2 → L1 easier in", f"{(gap > 0).sum()} of {n} pairs"),
             ("Mean F1, L1 → L2", f"{into2.mean():.3f}"),
             ("Mean F1, L2 → L1", f"{into1.mean():.3f}"),
             ("Largest gap", f"{pair_name(keys[gap.argmax()])}  +{gap.max():.3f}"),
             ("Smallest gap", f"{pair_name(keys[gap.argmin()])}  {gap.min():+.3f}"),
             ("Hardest L1 → L2", f"{pair_name(keys[0])}  {into2[0]:.3f}"),
             ("Easiest L1 → L2", f"{pair_name(keys[-1])}  {into2[-1]:.3f}")]
    note.axis("off")
    for i, (t, v) in enumerate(lines):
        note.text(0, 0.98 - i * 0.12, t, color=MUTED, fontsize=9, va="top", transform=note.transAxes)
        note.text(0, 0.98 - i * 0.12 - 0.04, v, color=INK, fontsize=9.5, va="top",
                  family="monospace", transform=note.transAxes)
    fig.suptitle(title, x=0.125, y=0.97, ha="left", color=INK, fontsize=14, fontweight="bold")
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="data/preds_xlmr_durw.npz", help=".npz, or the folder it unzips to")
    ap.add_argument("--results", default="data/anticipation_xlmr_durw_12lang.json")
    ap.add_argument("--out", default="docs/figures/all_pairs/direction_f1.png")
    ap.add_argument("--title", default="Switch F1 by direction, fine-tuned XLM-R")
    args = ap.parse_args()
    r = json.load(open(args.results))
    s = scores(args.preds, r["tau"])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    figure(s, args.title, args.out)
    out = os.path.splitext(args.out)[0] + ".json"
    json.dump(dict(preds=args.preds, tau=r["tau"], per_pair=s), open(out, "w"), indent=1)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
