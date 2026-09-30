"""Training helpers: reproducible seeds, class-weighted duration loss, saved predictions."""
from __future__ import annotations

import random

import numpy as np

from .anticipation import N_DUR
from .streaming import COLUMNS, example_for, iter_rows


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (model initialisation and dropout)."""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def count_duration_classes(path: str, unit: str = "subword", languages=None,
                           predict_at: str = "piece") -> np.ndarray:
    """How often each duration class is a training target, on the TRAIN split only.

    Counts the positions the duration loss actually uses: a switch whose class
    is certain (``dur_known``), and with ``predict_at="word_end"`` only the
    last piece of each word.
    """
    make = example_for(unit)
    counts = np.zeros(N_DUR, dtype=np.int64)
    for row in iter_rows(path, split="train", languages=languages, columns=COLUMNS[unit]):
        ex = make(row)
        if ex is None:
            continue
        m = ex["dur_known"]
        if predict_at == "word_end" and ex.get("unit_end") is not None:
            m = m & ex["unit_end"]
        counts += np.bincount(ex["y_dur"][m].astype(np.int64), minlength=N_DUR)
    return counts


def duration_class_weights(counts) -> np.ndarray:
    """Inverse-frequency weights, rescaled to mean 1 (an empty class gets weight 0)."""
    counts = np.asarray(counts, dtype=float)
    w = np.where(counts > 0, 1.0 / np.maximum(counts, 1), 0.0)
    return w * (len(w) / w.sum()) if w.sum() else np.ones(len(w))


def save_predictions(path: str, val: dict, test: dict) -> None:
    """Every scored position's switch logit and labels, for offline threshold and
    calibration experiments; ``dprob`` is the duration head's softmax."""
    arrays = {f"{split}_{k}": np.asarray(v) for split, res in (("val", val), ("test", test))
              for k, v in res.items()}
    np.savez_compressed(path, **arrays)
