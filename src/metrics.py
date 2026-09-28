"""Anticipatory metrics.

``switch`` is the rare class (~5% of positions), so accuracy is useless here -
a model that never predicts a switch scores 95%. Everything below is reported
on the positive class.

**Anticipatory F1** is the harmonic mean of precision and recall for
"token t+1 begins a new language", evaluated only at positions with a defined
label and a full causal prefix.
"""
from __future__ import annotations

import numpy as np


def prf(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Precision / recall / F1 on the positive class, plus the raw counts."""
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    n = tp + fp + fn + tn
    return dict(precision=p, recall=r, f1=f1, tp=tp, fp=fp, fn=fn, tn=tn,
                n=n, base_rate=(tp + fn) / n if n else 0.0,
                accuracy=(tp + tn) / n if n else 0.0)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval - used for the small per-language slices."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def by_group(y_true, y_pred, groups) -> dict[str, dict]:
    """Per-group PRF, for slicing by language pair or switch direction."""
    groups = np.asarray(groups)
    return {str(g): prf(np.asarray(y_true)[groups == g],
                        np.asarray(y_pred)[groups == g])
            for g in np.unique(groups)}


def macro_f1(per_group: dict[str, dict]) -> float:
    """Unweighted mean F1 - the universality number.

    Cantonese is 51% of the corpus, so a micro average measures Cantonese.
    A macro average over language pairs is what says whether the approach
    transfers.
    """
    return float(np.mean([g["f1"] for g in per_group.values()])) if per_group else 0.0


def duration_scores(y_true, y_pred) -> dict:
    """Bucketed-duration accuracy, plus off-by-one-bucket tolerance."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true) == 0:
        return dict(n=0, accuracy=0.0, within_one=0.0, mae_buckets=0.0)
    return dict(
        n=int(len(y_true)),
        accuracy=float(np.mean(y_true == y_pred)),
        within_one=float(np.mean(np.abs(y_true - y_pred) <= 1)),
        mae_buckets=float(np.mean(np.abs(y_true - y_pred))),
    )


def fmt_table(rows: list[dict], cols: list[str], title: str = "") -> str:
    """Plain-text table - these numbers go straight into the write-up."""
    head = [c for c in cols]
    widths = [max(len(h), *(len(f"{r.get(h, '')}") for r in rows)) for h in head]
    line = "  ".join(h.ljust(w) for h, w in zip(head, widths))
    out = ([title] if title else []) + [line, "  ".join("-" * w for w in widths)]
    for r in rows:
        out.append("  ".join(f"{r.get(h, '')}".ljust(w) for h, w in zip(head, widths)))
    return "\n".join(out)
