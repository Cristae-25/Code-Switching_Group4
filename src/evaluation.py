"""The one evaluation report every model is scored with.

Switch prediction is scored on the positive class at every labelled position:
precision, recall and F1 with Wilson 95% intervals, overall, per language
pair, macro-averaged across pairs, and per direction. Direction is where the
switch goes: a position whose current token is L1 predicts a switch "into
L2". Duration is scored at true switch points whose class is certain, with
3-class accuracy, macro-F1 and a confusion matrix.

Pairs are keyed "L1->L2", so German->French and German->English are never
merged.
"""
from __future__ import annotations

import numpy as np

from .anticipation import DUR_LABELS
from .metrics import duration_scores, fmt_table, prf, wilson

PAIR_KEYS = [
    "Arabic->English", "Cantonese->English", "Chinese->English", "French->English",
    "German->English", "German->French", "Hindi->English", "Italian->English",
    "Japanese->English", "Korean->English", "Malay->English", "Philippines->English",
    "Russian->English", "Spanish->English", "Vietnamese->English",
]
DIRECTIONS = {0: "into L2", 1: "into L1"}


def pair_key(first_language: str, second_language: str) -> str:
    return f"{first_language}->{second_language}"


def switch_block(truth, pred) -> dict:
    s = prf(truth, pred)
    s["precision_ci"] = wilson(s["tp"], s["tp"] + s["fp"])
    s["recall_ci"] = wilson(s["tp"], s["tp"] + s["fn"])
    return s


def full_report(truth, pred, pairs, langs, dur_true=None, dur_pred=None) -> dict:
    """Everything reported for one model on one split.

    ``pairs`` are indices into ``PAIR_KEYS``; ``langs`` is the current token's
    slot (0 = L1, 1 = L2) at each scored position.
    """
    truth, pred = np.asarray(truth).astype(int), np.asarray(pred).astype(int)
    pairs, langs = np.asarray(pairs), np.asarray(langs)
    per_pair = {PAIR_KEYS[int(p)]: switch_block(truth[pairs == p], pred[pairs == p])
                for p in np.unique(pairs)}
    out = dict(
        overall=switch_block(truth, pred),
        per_pair=per_pair,
        macro_f1=float(np.mean([v["f1"] for v in per_pair.values()])) if per_pair else 0.0,
        per_direction={name: switch_block(truth[langs == k], pred[langs == k])
                       for k, name in DIRECTIONS.items() if (langs == k).any()},
    )
    if dur_true is not None:
        out["duration"] = duration_scores(dur_true, dur_pred)
    return out


def _row(name, s) -> dict:
    return dict(name=name, n=f"{s['n']:,}", base=f"{s['base_rate']:.3f}",
                P=f"{s['precision']:.3f}", R=f"{s['recall']:.3f}", F1=f"{s['f1']:.3f}")


def format_report(report: dict, title: str) -> str:
    cols = ["name", "n", "base", "P", "R", "F1"]
    o = report["overall"]
    parts = [f"{title}\n  switch F1 {o['f1']:.3f}  (P {o['precision']:.3f}, R {o['recall']:.3f}, "
             f"recall 95% CI [{o['recall_ci'][0]:.3f}, {o['recall_ci'][1]:.3f}])  "
             f"| macro-F1 over pairs {report['macro_f1']:.3f}"]
    parts.append(fmt_table([_row(k, v) for k, v in report["per_direction"].items()], cols,
                           "\nBy direction:"))
    pp = sorted(report["per_pair"].items(), key=lambda kv: -kv[1]["f1"])
    parts.append(fmt_table([_row(k, v) for k, v in pp], cols, "\nBy language pair:"))
    if "duration" in report:
        d = report["duration"]
        parts.append(f"\nDuration (true switch points, n={d['n']:,}): 3-class accuracy "
                     f"{d['accuracy']:.3f} | macro-F1 {d['macro_f1']:.3f} | within one "
                     f"{d['within_one']:.3f} | per-class F1 "
                     + ", ".join(f"{c} {f:.3f}" for c, f in zip(DUR_LABELS, d["per_class_f1"])))
    return "\n".join(parts)
