#!/usr/bin/env python
"""Duration as two classes, Short (1-6, Small + Medium) vs Large (7+), from saved predictions.

    ./.venv/bin/python scripts/plot_duration_binary.py xlmr_prev_gap --out docs/figures/chris/gap

No retraining: P(Short) = P(Small) + P(Medium) from the three-class head in
``data/preds_<TAG>``, and the Short/Large cut is tuned on val for macro-F1.
Scored at the true switch points, as the three-class report is.

Writes:
    pair_f1.png    per pair: switch F1 | duration macro-F1, three classes vs two
    confusion.png  every pair pooled: switch 2x2 | duration Short-vs-Large 2x2
    data/duration_binary_<TAG>.json

Two-class scores sit on a higher chance floor than three-class ones, so the
figure also prints Cohen's kappa (agreement beyond chance) for both.
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

from scripts.plot_confusion import INK, INK_2, MUTED, SURFACE, panel  # noqa: E402
from src.evaluation import PAIR_KEYS  # noqa: E402

GRID, BASELINE = "#e1e0d9", "#c3c2b7"
SWITCH_C = "#2a78d6"            # categorical slot 1: the switch head
THREE_C, TWO_C = "#c3c2b7", "#eb6834"   # three classes = the reference (neutral), two classes = slot 2


def load(tag: str) -> dict:
    path = f"data/preds_{tag}"
    if os.path.isdir(path):
        return {f[:-4]: np.load(os.path.join(path, f)) for f in os.listdir(path) if f.endswith(".npy")}
    return dict(np.load(path + ".npz"))


def f1_per_class(y, p, k):
    out = []
    for c in range(k):
        tp = ((p == c) & (y == c)).sum()
        out.append(2 * tp / max(int((p == c).sum() + (y == c).sum()), 1))
    return np.array(out)


def kappa(y, p, k):
    po = float(np.mean(y == p))
    pe = float(sum(np.mean(y == c) * np.mean(p == c) for c in range(k)))
    return (po - pe) / (1 - pe)


def pair_name(key) -> str:
    l1, l2 = str(key).split("->")
    l1 = "Tagalog" if l1 == "Philippines" else l1
    return l1 if l2 == "English" and l1 != "German" else f"{l1} → {l2}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--out", default="docs/figures/chris/gap")
    ap.add_argument("--n-boot", type=int, default=500)
    a = ap.parse_args()
    z = load(a.tag)
    r = json.load(open(f"data/anticipation_{a.tag}.json"))
    tuned = r["duration_tuned"]["report"]

    # Short (0) vs Large (1); cut tuned on val for macro-F1
    sv, yv = z["val_dprob"][:, 2], (z["val_dtrue"] == 2).astype(int)
    cuts = np.round(np.linspace(0.05, 0.95, 91), 2)
    cut = float(cuts[np.argmax([f1_per_class(yv, (sv > c).astype(int), 2).mean() for c in cuts])])
    y3 = z["test_dtrue"]
    y2 = (y3 == 2).astype(int)
    p2 = (z["test_dprob"][:, 2] > cut).astype(int)
    pairs, rows = z["test_dpair"], z["test_drow"]

    # row-level bootstrap of the two-class macro-F1
    ur, inv = np.unique(rows, return_inverse=True)
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(a.n_boot):
        w = np.bincount(rng.integers(0, len(ur), len(ur)), minlength=len(ur))[inv]
        # F1 = 2TP / (2TP + FP + FN) = 2TP / (|pred c| + |true c|)
        f = [2 * (w * ((p2 == c) & (y2 == c))).sum() / ((w * (p2 == c)).sum() + (w * (y2 == c)).sum())
             for c in (0, 1)]
        boot.append(np.mean(f))

    d3 = tuned["duration"]
    c3 = np.asarray(d3["confusion"])
    po3 = np.trace(c3) / c3.sum()
    pe3 = float((c3.sum(1) * c3.sum(0)).sum() / c3.sum() ** 2)
    k3 = (po3 - pe3) / (1 - pe3)
    f2 = f1_per_class(y2, p2, 2)
    two = dict(cut=cut, n=int(len(y2)), accuracy=float(np.mean(y2 == p2)), macro_f1=float(f2.mean()),
               macro_f1_ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
               per_class_f1={"short": float(f2[0]), "large": float(f2[1])}, kappa=kappa(y2, p2, 2),
               confusion=[[int(((y2 == i) & (p2 == j)).sum()) for j in (0, 1)] for i in (0, 1)],
               majority_macro_f1=float(f1_per_class(y2, np.full_like(y2, np.bincount(y2).argmax()), 2).mean()),
               share_large=float(y2.mean()))
    three = dict(accuracy=d3["accuracy"], macro_f1=d3["macro_f1"], kappa=float(k3),
                 offsets=r["duration_tuned"]["offsets"])
    per_pair = {}
    for i, key in enumerate(PAIR_KEYS):
        m = pairs == i
        if not m.sum():
            continue
        k = str(key)
        per_pair[k] = dict(n=int(m.sum()), switch_f1=r["per_pair"][k]["f1"],
                           three_macro_f1=tuned["duration_per_pair"][k]["macro_f1"],
                           three_accuracy=tuned["duration_per_pair"][k]["accuracy"],
                           two_macro_f1=float(f1_per_class(y2[m], p2[m], 2).mean()),
                           two_accuracy=float(np.mean(y2[m] == p2[m])))
    out_json = f"data/duration_binary_{a.tag}.json"
    json.dump(dict(tag=a.tag, three_class_val_tuned=three, two_class=two, per_pair=per_pair),
              open(out_json, "w"), indent=1)
    print(f"three classes (val-tuned): acc {three['accuracy']:.3f} | macro-F1 {three['macro_f1']:.3f} | kappa {k3:.3f}")
    print(f"two classes (cut {cut}):   acc {two['accuracy']:.3f} | macro-F1 {two['macro_f1']:.3f} "
          f"[{two['macro_f1_ci'][0]:.3f}, {two['macro_f1_ci'][1]:.3f}] | kappa {two['kappa']:.3f} "
          f"| majority-class macro-F1 {two['majority_macro_f1']:.3f}")
    print("wrote", out_json)
    os.makedirs(a.out, exist_ok=True)
    pair_figure(per_pair, r, three, two, os.path.join(a.out, "pair_f1.png"))
    confusion_figure(r["overall"], two, os.path.join(a.out, "confusion.png"))
    return 0


def _style(ax, xlabel):
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(axis="x", colors=MUTED, labelsize=9, length=0)
    ax.tick_params(axis="y", colors=INK_2, labelsize=10, length=0)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9.5, labelpad=8)


def pair_figure(per_pair, r, three, two, path):
    order = sorted(per_pair, key=lambda k: per_pair[k]["switch_f1"])   # best at the top
    names = [pair_name(k) for k in order]
    y = np.arange(len(order))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 8.4), facecolor=SURFACE, sharey=True,
                                   gridspec_kw=dict(width_ratios=[1, 1.35], wspace=0.1))
    sw = [per_pair[k]["switch_f1"] for k in order]
    ax1.barh(y, sw, height=0.62, color=SWITCH_C, edgecolor=SURFACE, linewidth=1.5)
    for yi, v in zip(y, sw):
        ax1.text(v + 0.012, yi, f"{v:.3f}", va="center", fontsize=8.5, color=INK)
    _style(ax1, "Switch F1")
    ax1.set_xticks([0, 0.25, 0.5, 0.75])            # the right panel's 0 sits where 1.00 would
    ax1.set_xlim(0, 1)
    ax1.set_yticks(y, names)
    ax1.set_title(f"Switch  ·  pooled F1 {r['overall']['f1']:.3f}, macro over pairs {r['macro_f1']:.3f}",
                  loc="left", color=INK, fontsize=10.5, pad=10)

    h = 0.38
    t3 = [per_pair[k]["three_macro_f1"] for k in order]
    t2 = [per_pair[k]["two_macro_f1"] for k in order]
    ax2.barh(y - h / 2, t3, height=h, color=THREE_C, edgecolor=SURFACE, linewidth=1.5,
             label=f"3 classes: Small 1–2 / Medium 3–6 / Large 7+  (pooled {three['macro_f1']:.3f})")
    ax2.barh(y + h / 2, t2, height=h, color=TWO_C, edgecolor=SURFACE, linewidth=1.5,
             label=f"2 classes: Short 1–6 / Large 7+  (pooled {two['macro_f1']:.3f})")
    for yi, a3, a2 in zip(y, t3, t2):
        ax2.text(a3 + 0.012, yi - h / 2, f"{a3:.3f}", va="center", fontsize=8, color=INK_2)
        ax2.text(a2 + 0.012, yi + h / 2, f"{a2:.3f}", va="center", fontsize=8, color=INK)
    _style(ax2, "Duration macro-F1 at the true switch points (val-tuned)")
    ax2.tick_params(axis="y", length=0)
    ax2.set_title("Duration  ·  three classes vs Small + Medium merged", loc="left", color=INK,
                  fontsize=10.5, pad=10)
    # bars run bottom-to-top with y; legend order should match the top bar of each pair (2 classes)
    hs, ls = ax2.get_legend_handles_labels()
    ax2.legend(hs[::-1], ls[::-1], frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left",
               bbox_to_anchor=(0, -0.07))
    fig.text(0.125, 0.955, "Per-pair F1 · XLM-R, decide at gap + previous-word language (test split)",
             color=INK, fontsize=13.5, fontweight="bold", ha="left")
    fig.text(0.125, 0.925, "Chance floors differ: a blind guess at the class frequencies scores macro-F1 "
             f"0.333 with three classes, 0.500 with two. Agreement beyond chance (Cohen's κ): "
             f"{three['kappa']:.2f} → {two['kappa']:.2f}.",
             color=MUTED, fontsize=9, ha="left")
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def confusion_figure(sw, two, path):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), facecolor=SURFACE,
                             gridspec_kw=dict(width_ratios=[1, 1], wspace=0.55))
    for ax in axes:
        ax.set_facecolor(SURFACE)
    panel(axes[0], [[sw["tp"], sw["fn"]], [sw["fp"], sw["tn"]]], ["Switch", "No switch"])
    c = two["confusion"]                                  # rows/cols: Short, Large -> show Large first
    im = panel(axes[1], [[c[1][1], c[1][0]], [c[0][1], c[0][0]]], ["Large\n(7+)", "Small + Medium\n(1–6)"])
    titles = (f"Switch  ·  F1 {sw['f1']:.3f}   P {sw['precision']:.3f}   R {sw['recall']:.3f}",
              f"Duration, 2 classes  ·  macro-F1 {two['macro_f1']:.3f}   acc {two['accuracy']:.3f}")
    top = axes[0].get_position().y1
    for ax, t in zip(axes, titles):
        fig.text(ax.get_position().x0, top + 0.17, t, color=INK, fontsize=10.5, ha="left")
    fig.text(axes[0].get_position().x0, top + 0.26, "All 15 pairs pooled · XLM-R, decide at gap + previous-word language",
             color=INK, fontsize=13, fontweight="bold", ha="left")
    p = (sw["tp"] + sw["fn"]) / sw["n"]
    q = two["share_large"]
    notes = (f"{sw['n']:,} scored positions. Blind guess at the switch\nrate ({p:.1%}) → F1 {p:.3f}",
             f"{two['n']:,} true switch points; Large if P(Large) > {two['cut']:.2f}\n"
             f"(tuned on val). Blind guess at {q:.0%} Large → macro-F1 0.500;\nκ {two['kappa']:.2f}")
    bottom = axes[0].get_position().y0
    for ax, t in zip(axes, notes):
        fig.text(ax.get_position().x0, bottom - 0.05, t, color=MUTED, fontsize=8.5, ha="left", va="top",
                 linespacing=1.5)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.03)
    cb.set_label("Share of the actual row (recall)", color=MUTED, fontsize=8.5)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, labelsize=8, length=0)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1], labels=["0%", "25%", "50%", "75%", "100%"])
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    raise SystemExit(main())
