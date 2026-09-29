#!/usr/bin/env python
"""Training-free baselines for the anticipatory switch task.

Every baseline here is *causal by construction*: it is a function of features
computable from tokens 1..t alone. The two count-based ones estimate their
tables on the TRAIN split and are evaluated on TEST, so even a lookup table
never sees its evaluation data.

    never       always predict "no switch"      - the accuracy trap
    always      always predict "switch"         - recall ceiling
    prior       Bernoulli(train base rate)      - chance
    naive_l2    switch iff the current token is second-language
    hazard      P(switch | current language, length of current run so far)

The hazard model emits a probability, so it needs a decision threshold. That
threshold is tuned on the VALIDATION split and then applied unchanged to test -
tuning it on test would be reporting an oracle. The default 0.5 is reported
too, because what happens there is the whole lesson: the switch hazard never
exceeds 0.26, so a correctly-calibrated model thresholded at 0.5 predicts "no
switch" forever and scores 91% accuracy with an F1 of zero.

``naive_l2`` is the checklist's "naive model": one bit of context, the language
of the current token. ``hazard`` adds the one extra causal feature that matters
- how long the current run has already gone on - and is the bar a learned model
has to clear to have earned its parameters.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.anticipation import L2, NONE, N_DUR, DUR_LABELS  # noqa: E402
from src.metrics import (by_group, confusion_table, duration_scores, fmt_table,  # noqa: E402
                         macro_f1, prf, wilson)
from src.streaming import COLUMNS, DEFAULT_PATHS, example_for, iter_rows  # noqa: E402
from src.evaluation import PAIR_KEYS, format_report, full_report, pair_key  # noqa: E402

RUNPOS_CAP = 10          # run-position feature is capped; the tail is flat


def run_position(langs: np.ndarray) -> np.ndarray:
    """How many language-bearing positions the current run has covered, inclusive of t.

    Strictly causal. NONE positions are transparent: they carry the count
    through unchanged and are never scored.
    """
    pos = np.zeros(len(langs), dtype=np.int64)
    run, last = 0, None
    for i, lang in enumerate(langs):
        if lang != NONE:
            run = run + 1 if lang == last else 1
            last = lang
        pos[i] = run
    return pos


def collect(path: str, split: str, limit: int | None = None, unit: str = "word"):
    """Stream a split into flat arrays of causal features and labels."""
    cur_lang, runpos, y_sw, y_dur, dur_ok, pair, shared = [], [], [], [], [], [], []
    make = example_for(unit)
    for i, row in enumerate(iter_rows(path, split=split, columns=COLUMNS[unit])):
        if limit and i >= limit:
            break
        ex = make(row)
        if ex is None:
            continue
        m = ex["y_sw"] >= 0                       # drop the final position
        langs = ex["langs"].astype(np.int64)
        cur_lang.append(langs[m])
        runpos.append(np.minimum(run_position(langs), RUNPOS_CAP)[m])
        y_sw.append(ex["y_sw"][m].astype(np.int64))
        y_dur.append(ex["y_dur"][m].astype(np.int64))
        dur_ok.append(ex["dur_known"][m])
        shared.append(ex["shared"][m])
        pair.append(np.full(int(m.sum()), PAIR_KEYS.index(
            pair_key(ex["first_language"], ex["second_language"]))))
    cat = lambda xs: np.concatenate(xs) if xs else np.empty(0, np.int64)  # noqa: E731
    return dict(cur_lang=cat(cur_lang), runpos=cat(runpos), y_sw=cat(y_sw),
                y_dur=cat(y_dur), dur_ok=cat(dur_ok).astype(bool), pair=cat(pair),
                shared=cat(shared).astype(bool))


def fit_hazard(train: dict) -> np.ndarray:
    """P(switch at t+1 | current language, capped run position). Train only."""
    table = np.zeros((2, RUNPOS_CAP + 1))
    for lang in (0, 1):
        for pos in range(1, RUNPOS_CAP + 1):
            m = (train["cur_lang"] == lang) & (train["runpos"] == pos)
            table[lang, pos] = train["y_sw"][m].mean() if m.any() else 0.0
    return table


def fit_hazard_per_pair(train: dict, n_pairs: int = len(PAIR_KEYS),
                        min_count: int = 1) -> np.ndarray:
    """The same hazard fitted separately for each language pair: table[pair, lang, pos].

    A cell with fewer than ``min_count`` training positions falls back to the
    pooled rate, so an unseen (pair, language, run length) is never a silent 0.
    """
    pooled = fit_hazard(train)
    table = np.repeat(pooled[None], n_pairs, axis=0)
    for p in range(n_pairs):
        mp = train["pair"] == p
        for lang in (0, 1):
            for pos in range(1, RUNPOS_CAP + 1):
                m = mp & (train["cur_lang"] == lang) & (train["runpos"] == pos)
                if m.sum() >= min_count:
                    table[p, lang, pos] = train["y_sw"][m].mean()
    return table


def tune_on_val(prob_val, y_val, candidates):
    """F1-maximising threshold on VAL (never on test)."""
    grid = np.unique(np.round(np.sort(np.ravel(candidates)), 6))
    grid = grid[grid > 0]
    return max((prf(y_val, (prob_val >= t).astype(np.int64))["f1"], float(t)) for t in grid)


def report_with_shared(y, yp, pair, lang, shared) -> dict:
    """``full_report``, plus the same on the shared positions (non-CJK word ends)."""
    rep = full_report(y, yp, pair, lang)
    shared = np.asarray(shared, dtype=bool)
    if shared.any():
        rep["shared"] = full_report(y[shared], yp[shared], pair[shared], lang[shared])
    return rep


def fit_duration(train: dict) -> np.ndarray:
    """Most likely duration bucket given the language being switched INTO."""
    out = np.zeros(2, dtype=np.int64)
    for lang in (0, 1):
        # switching into `lang` means the current token is the other language
        m = (train["cur_lang"] == (1 - lang)) & (train["y_sw"] == 1) & train["dur_ok"]
        if m.any():
            out[lang] = np.bincount(train["y_dur"][m], minlength=N_DUR).argmax()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=None)
    ap.add_argument("--unit", choices=["word", "subword"], default="word")
    ap.add_argument("--limit", type=int, default=None, help="rows per split, for a smoke run")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--json", default=None)
    ap.add_argument("--names", choices=["script_words", "transparent", "script", "hybrid", "language"],
                    default="script_words",
                    help="how named entities count (see src/segments.py); needs an "
                         "is_entity column, else names count as tagged")
    args = ap.parse_args()
    import src.anticipation as _ant
    _ant.NAMES = args.names
    args.path = args.path or DEFAULT_PATHS[args.unit]
    if args.unit == "subword" and args.names != "script_words":
        raise SystemExit("--names only applies to --unit word: the sub-word labels were "
                         "built with script_words")
    args.json = args.json or ("data/baseline_anticipation.json" if args.unit == "word"
                              else "data/baseline_anticipation_subword.json")
    print(f"unit {args.unit} | {args.path}")

    print("streaming train split ...", flush=True)
    train = collect(args.path, "train", args.limit, args.unit)
    print(f"  {len(train['y_sw']):,} scored positions", flush=True)
    print("streaming val split ...", flush=True)
    val = collect(args.path, "val", args.limit, args.unit)
    print(f"  {len(val['y_sw']):,} scored positions", flush=True)
    print("streaming test split ...", flush=True)
    test = collect(args.path, "test", args.limit, args.unit)
    print(f"  {len(test['y_sw']):,} scored positions\n", flush=True)

    base = train["y_sw"].mean()
    hazard = fit_hazard(train)
    rng = np.random.default_rng(0)

    # Tune both hazard thresholds on VAL, never on test.
    best_f1, tau = tune_on_val(hazard[val["cur_lang"], val["runpos"]], val["y_sw"], hazard)
    hz_pair = fit_hazard_per_pair(train)
    pair_f1, tau_pair = tune_on_val(hz_pair[val["pair"], val["cur_lang"], val["runpos"]],
                                    val["y_sw"], hz_pair)
    print(f"hazard threshold tuned on val: tau={tau:.3f} (val F1 {best_f1:.3f}) | "
          f"per-pair hazard: tau={tau_pair:.3f} (val F1 {pair_f1:.3f})\n", flush=True)

    preds = {
        "never": np.zeros_like(test["y_sw"]),
        "always": np.ones_like(test["y_sw"]),
        "prior": (rng.random(len(test["y_sw"])) < base).astype(np.int64),
        "naive_l2": (test["cur_lang"] == L2).astype(np.int64),
        "hazard@0.5": (hazard[test["cur_lang"], test["runpos"]] >= args.threshold).astype(np.int64),
        "hazard@tuned": (hazard[test["cur_lang"], test["runpos"]] >= tau).astype(np.int64),
        "hazard_pair@tuned": (hz_pair[test["pair"], test["cur_lang"], test["runpos"]]
                              >= tau_pair).astype(np.int64),
    }

    rows, detail = [], {}
    for name, yp in preds.items():
        rep = report_with_shared(test["y_sw"], yp, test["pair"], test["cur_lang"], test["shared"])
        s = rep["overall"]
        lo, hi = s["precision_ci"]
        rows.append(dict(model=name,
                         P=f"{s['precision']:.3f}", R=f"{s['recall']:.3f}",
                         F1=f"{s['f1']:.3f}", macroF1=f"{rep['macro_f1']:.3f}",
                         acc=f"{s['accuracy']:.3f}",
                         prec95=f"[{lo:.3f},{hi:.3f}]"))
        detail[name] = rep

    print(fmt_table(rows, ["model", "P", "R", "F1", "macroF1", "acc", "prec95"],
                    f"Anticipatory switch prediction - TEST split "
                    f"(base rate {test['y_sw'].mean():.4f})"))

    # Per-direction, for the best training-free baseline.
    for name in ("hazard@tuned", "hazard_pair@tuned"):
        print("\n" + format_report(detail[name], f"{name}:"))

    print("\nHazard table  P(switch at t+1 | current run length so far):")
    hz = [dict(run_len=str(pos),
               after_L1=f"{hazard[0, pos]:.3f}", after_L2=f"{hazard[1, pos]:.3f}")
          for pos in range(1, RUNPOS_CAP + 1)]
    print(fmt_table(hz, ["run_len", "after_L1", "after_L2"]))

    # --- duration ---------------------------------------------------------
    dur_mode = fit_duration(train)
    m = (test["y_sw"] == 1) & test["dur_ok"]
    target_lang = 1 - test["cur_lang"][m]
    d_true = test["y_dur"][m]
    d_pred = dur_mode[target_lang]
    ds = duration_scores(d_true, d_pred)
    for name in ("hazard@tuned", "hazard_pair@tuned"):   # the training-free duration baseline
        detail[name]["duration"] = ds
    print(f"\nDuration at oracle switch positions (n={ds['n']:,}), "
          f"mode-per-direction baseline:")
    print(f"  3-class accuracy {ds['accuracy']:.3f} | macro-F1 {ds['macro_f1']:.3f} "
          f"| within one class {ds['within_one']:.3f}")
    print(confusion_table(ds["confusion"], DUR_LABELS))
    dur_rows = []
    for lang, label in ((1, "into L2"), (0, "into L1")):
        sel = target_lang == lang
        if not sel.any():
            continue
        hist = np.bincount(d_true[sel], minlength=N_DUR) / int(sel.sum())
        row = dict(target=label, n=f"{int(sel.sum()):,}",
                   mode=DUR_LABELS[int(dur_mode[lang])],
                   acc=f"{duration_scores(d_true[sel], d_pred[sel])['accuracy']:.3f}")
        row.update({lab: f"{h:.3f}" for lab, h in zip(DUR_LABELS, hist)})
        dur_rows.append(row)
    print(fmt_table(dur_rows, ["target", "n", "mode", "acc"] + DUR_LABELS))

    detail["_meta"] = dict(
        train_positions=int(len(train["y_sw"])),
        test_positions=int(len(test["y_sw"])),
        train_base_rate=float(base), test_base_rate=float(test["y_sw"].mean()),
        hazard_table=hazard.tolist(), duration_mode=dur_mode.tolist(),
        duration=ds, pairs=PAIR_KEYS, hazard_tau=float(tau), hazard_pair_tau=float(tau_pair),
        val_positions=int(len(val["y_sw"])), hazard_val_f1=float(best_f1),
    )
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w") as fh:
        json.dump(detail, fh, indent=2, default=float)
    print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
