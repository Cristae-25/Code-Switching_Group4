"""Training helpers: reproducible seeds, class-weighted duration loss, the evaluation
loop and its scoring (shared by train_anticipation.py and eval_lince.py), saved predictions."""
from __future__ import annotations

import contextlib
import random

import numpy as np

from .anticipation import N_DUR, collapse_duration
from .causal_model import duration_probs
from .evaluation import full_report
from .metrics import prf
from .models import pair_ids
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


# --------------------------------------------------------------------------- evaluation
def autocast(dev):
    """fp16 on CUDA (a T4 has no bf16); full precision elsewhere."""
    import torch
    if dev.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def scored_positions(b):
    """``(row_id, position in the row)`` of every scored position in a batch, in order."""
    import torch
    m = b["sw_mask"]
    row = b["row_id"].unsqueeze(1).expand_as(m)[m]
    pos = (b["start"].unsqueeze(1) + torch.arange(m.shape[1])).expand_as(m)[m]
    return row.numpy(), pos.numpy()


def head_classes(model, dur) -> int:
    """Duration classes a model predicts: a softmax head's width, or cuts + 1 (ordinal)."""
    return dur.shape[-1] + (1 if getattr(model, "dur_ordinal", False) else 0)


def batch_dur_labels(b, n_dur: int):
    """The batch's ``y_dur`` for an ``n_dur``-class head: as stored (3), or Short / Large (2)."""
    return collapse_duration(b["y_dur"]) if n_dur == 2 else b["y_dur"]


def evaluate(model, loader, dev, max_batches=0):
    """Logits and labels at every scored position, with pair and direction."""
    import torch
    with torch.no_grad():
        return _evaluate(model, loader, dev, max_batches)


def _evaluate(model, loader, dev, max_batches=0):
    import torch
    model.eval()
    out = {k: [] for k in ("logit", "y", "pair", "lang", "dtrue", "dpred", "dprob", "dpair",
                           "dlang", "dlogit", "drow", "shared", "row", "pos")}
    for i, b in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        with autocast(dev):
            sw, dur = model(b["word_ids"].to(dev), b["lang_ids"].to(dev), pair_ids(model, b).to(dev))
        m = b["sw_mask"]
        out["logit"].append(sw.float().cpu()[m].numpy())
        out["y"].append(b["y_sw"][m].numpy())
        out["pair"].append(b["report_pair"].unsqueeze(1).expand_as(m)[m].numpy())
        out["lang"].append(b["cur_lang"][m].numpy())        # direction: current language
        dm = b["dur_mask"]
        if dm.any():
            out["dtrue"].append(batch_dur_labels(b, head_classes(model, dur))[dm].numpy())
            probs = duration_probs(dur.float().cpu()[dm], getattr(model, "dur_ordinal", False))
            out["dpred"].append(probs.argmax(-1).numpy())
            out["dprob"].append(probs.numpy())
            # aligned with dtrue: dur_mask is a subset of sw_mask
            out["dpair"].append(b["report_pair"].unsqueeze(1).expand_as(dm)[dm].numpy())
            out["dlang"].append(b["cur_lang"][dm].numpy())
            out["dlogit"].append(sw.float().cpu()[dm].numpy())
            out["drow"].append(b["row_id"].unsqueeze(1).expand_as(dm)[dm].numpy())
        out["shared"].append(b["shared_mask"][m].numpy())
        row, pos = scored_positions(b)                      # trace errors back to the text
        out["row"].append(row)
        out["pos"].append(pos)
    return {k: np.concatenate(v) if v else np.empty(0) for k, v in out.items()}


def tune_threshold(prob, truth):
    """Pick the F1-maximising threshold on a held-out split."""
    best = (0.0, 0.5)
    for t in np.quantile(prob, np.linspace(0.50, 0.999, 60)):
        best = max(best, (prf(truth, prob >= t)["f1"], float(t)))
    return best[1], best[0]


def score(res, tau):
    """The full report, plus ``shared``: the same report on the positions every
    model shares (the last piece of each non-CJK word), comparable across tokenizers."""
    pred = (1 / (1 + np.exp(-res["logit"])) >= tau).astype(int)
    # Duration side arrays are aligned with dtrue; older prediction files lack them.
    def aligned(k):
        return res[k] if k in res and len(res[k]) == len(res["dtrue"]) else None

    n_dur = res["dprob"].shape[1] if np.ndim(res.get("dprob")) == 2 else N_DUR
    dlogit = aligned("dlogit")
    fired = None if dlogit is None else 1 / (1 + np.exp(-np.asarray(dlogit))) >= tau
    rep = full_report(res["y"], pred, res["pair"], res["lang"], res["dtrue"], res["dpred"],
                      dur_pairs=aligned("dpair"), dur_langs=aligned("dlang"), dur_fired=fired,
                      n_dur=n_dur)
    sh = np.asarray(res.get("shared", np.zeros(len(pred), bool)), dtype=bool)
    if sh.any():
        rep["shared"] = full_report(res["y"][sh], pred[sh], res["pair"][sh], res["lang"][sh])
    return rep
