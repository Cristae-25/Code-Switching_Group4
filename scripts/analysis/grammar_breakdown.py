#!/usr/bin/env python
"""Do the model's mid-clause misses follow grammatical patterns? (no GPU)

    ./.venv/bin/python scripts/analysis/grammar_breakdown.py [TAG] [--view word_end|all]

Mid-clause switches (nothing between word t and word t+1; ``adjacent`` in
``scripts/analysis/punct_breakdown.py``) cause most of the missed switches. This script
groups those positions by properties of word t and its prefix, using only the
prefix, so any group that matters could become a legal causal input:

* ``freq``     how common word t is in its own language (wordfreq Zipf).
  ``top`` (Zipf >= 6) is mostly function words (the, of, de, 的); ``unknown`` is
  0, e.g. romanized Hindi (wordfreq's Hindi is Devanagari) or names.
* ``run``      words so far in the current language run, including t (1 = a
  single inserted word, the classic insertion pattern)
* ``clause``   words since the last comma, sentence end or turn, including t
* ``length``   characters in word t (a CJK character is one word)
* ``direction`` the current language: a switch from L1 goes into L2, and vice versa

For every group it reports the switch base rate next to the model's **mean
predicted probability** (the switch loss is unweighted BCE, so probabilities are
roughly calibrated), plus P/R/F1, the misses, and each group's share of all misses.

**How to read it:**
- A group whose base rate is far from the model's mean probability is a cue the
  model is not using. That is the case for building a grammar input.
- If the two track each other, the model already has that pattern, even when the
  group's recall is low.

``--view word_end`` (default) keeps the last piece of each word, where a
switch decision is made; ``all`` keeps every scored piece. Micro numbers are
dominated by the largest pair (Cantonese is about 80% of the 6-pair test set),
so per-pair tables are in the json. Writes ``data/results/grammar_breakdown_<TAG>.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.analysis.punct_breakdown import ADJ, COMMA, NO_LABEL, SENT, TURN, gap_buckets, load_preds  # noqa: E402
from src.anticipation import L1, NONE  # noqa: E402
from src.evaluation import PAIR_KEYS  # noqa: E402
from src.metrics import fmt_table, prf  # noqa: E402
from src.subword import MARKERS  # noqa: E402

CODES = {"English": "en", "Spanish": "es", "German": "de", "French": "fr", "Italian": "it",
         "Malay": "ms", "Philippines": "fil", "Vietnamese": "vi", "Hindi": "hi", "Russian": "ru",
         "Arabic": "ar", "Chinese": "zh", "Cantonese": "zh", "Japanese": "ja", "Korean": "ko"}
BOUNDARY = (COMMA, SENT, TURN)
_FREQ: dict[str, dict] = {}


def zipf(word: str, code: str) -> float:
    """Zipf frequency of one word from wordfreq's table, with no tokenizer.

    ``wordfreq.zipf_frequency`` tokenizes first, which needs MeCab for Japanese
    and Korean. Every lookup here is a single word, so the table is read directly.
    Chinese is simplified first (wordfreq's own mapping): Cantonese is written in
    traditional characters, which the zh table stores only as simplified.
    """
    if code not in _FREQ:
        import wordfreq
        _FREQ[code] = wordfreq.get_frequency_dict(code)
    if code == "zh":
        from wordfreq.chinese import simplify_chinese
        word = simplify_chinese(word)
    f = _FREQ[code].get(word, 0.0)
    return float(np.log10(f * 1e9)) if f > 0 else 0.0

FEATURES = {
    "freq": ("top (>=6)", "common (4.5-6)", "mid (3-4.5)", "rare (<3)", "unknown (0)"),
    "run": ("1", "2", "3-6", "7+"),
    "clause": ("1", "2-3", "4-7", "8+"),
    "length": ("1", "2-3", "4-6", "7+"),
    "direction": ("from L1 (into L2)", "from L2 (into L1)"),
}


def _bin(x: int, cuts) -> int:
    """Index of the first cut that ``x`` is at most, else the last bin."""
    for i, c in enumerate(cuts):
        if x <= c:
            return i
    return len(cuts)


def freq_band(z: float) -> int:
    if z <= 0:
        return 4
    return 0 if z >= 6 else 1 if z >= 4.5 else 2 if z >= 3 else 3


def _clean(text: str) -> str:
    for m in MARKERS:
        text = text.replace(m, "")
    return text.strip().lower()


def row_features(pieces, langs, words, utts, codes) -> dict:
    """Per-piece feature codes (-1 where undefined) and the word-end mask.

    Everything at piece t reads pieces <= t only, except ``gap``, the bucket that
    selects mid-clause positions (it describes the label, like the label itself).
    """
    langs = np.asarray(langs)
    n = len(langs)
    gap = gap_buckets(pieces, langs, utts)
    out = {k: np.full(n, -1, dtype=np.int8) for k in FEATURES}
    word_end = np.zeros(n, dtype=bool)
    prev_word, run_lang, run_words, clause_words, after_boundary = None, None, 0, 0, True
    text, last_lb = "", None
    for i in range(n):
        w = int(words[i])
        if langs[i] == NONE or w < 0:
            continue
        if w != prev_word:                       # a new language-bearing word starts
            run_words = run_words + 1 if langs[i] == run_lang else 1
            run_lang = langs[i]
            clause_words = 1 if after_boundary else clause_words + 1
            prev_word, text = w, ""
        text += pieces[i]
        word = _clean(text)
        out["run"][i] = _bin(run_words, (1, 2, 6))
        out["clause"][i] = _bin(clause_words, (1, 3, 7))
        out["length"][i] = _bin(len(word), (1, 3, 6))
        out["direction"][i] = 0 if langs[i] == L1 else 1
        out["freq"][i] = freq_band(zipf(word, codes[int(langs[i] != L1)])) if word else 4
        after_boundary = gap[i] in BOUNDARY
        if last_lb is not None and int(words[last_lb]) != w:
            word_end[last_lb] = True
        last_lb = i
    if last_lb is not None:
        word_end[last_lb] = True
    out["gap"] = gap
    out["word_end"] = word_end
    return out


def load_test_features(path: str, rows: set) -> dict:
    cols = ["row_id", "first_language", "second_language", "pieces", "piece_lang", "piece_word", "piece_utt"]
    feats = {}
    for batch in pq.ParquetFile(path).iter_batches(batch_size=4096, columns=cols):
        d = batch.to_pydict()
        for i, r in enumerate(d["row_id"]):
            if r not in rows:
                continue
            codes = (CODES[d["first_language"][i]], CODES[d["second_language"][i]])
            feats[r] = row_features(d["pieces"][i], d["piece_lang"][i], d["piece_word"][i],
                                    d["piece_utt"][i], codes)
    return feats


def group_report(y, prob, pred, groups, labels) -> list[dict]:
    total_fn = int(((y == 1) & ~pred).sum())
    out = []
    for j, lab in enumerate(labels):
        m = groups == j
        if not m.any():
            continue
        r = prf(y[m], pred[m])
        out.append(dict(group=lab, n=int(m.sum()), base_rate=float(y[m].mean()),
                        mean_prob=float(prob[m].mean()), precision=r["precision"], recall=r["recall"],
                        f1=r["f1"], missed=r["fn"], share_of_misses=r["fn"] / total_fn if total_fn else 0.0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", nargs="?", default="xlmr_6pairs")
    ap.add_argument("--path", default="data/subword.parquet")
    ap.add_argument("--view", choices=["word_end", "all"], default="word_end")
    args = ap.parse_args()

    z = load_preds(args.tag)
    tau = json.load(open(f"data/results/anticipation_{args.tag}.json"))["tau"]
    rows, pos, y = z["test_row"], z["test_pos"], z["test_y"]
    prob = 1 / (1 + np.exp(-z["test_logit"].astype(np.float64)))
    pred = prob >= tau
    pair = np.array(PAIR_KEYS)[z["test_pair"]]
    print(f"{args.tag}: {len(y):,} test positions, tau={tau:.3f}; computing features ...", flush=True)
    feats = load_test_features(args.path, set(rows.tolist()))

    def take(key):
        return np.array([feats[r][key][p] for r, p in zip(rows.tolist(), pos.tolist())])
    gap, wend = take("gap"), take("word_end")
    if (gap == NO_LABEL).any():
        raise SystemExit("some scored positions have no next word: labels and parquet disagree")
    keep = (gap == ADJ) & (wend if args.view == "word_end" else True)
    print(f"mid-clause positions ({args.view}): {int(keep.sum()):,}, switches {int(y[keep].sum()):,}, "
          f"missed {int(((y == 1) & ~pred & keep).sum()):,}")

    out = {"tag": args.tag, "view": args.view, "tau": float(tau), "n": int(keep.sum()), "features": {}}
    cols = ["group", "n", "base_rate", "mean_prob", "precision", "recall", "f1", "missed", "share_of_misses"]
    for name, labels in FEATURES.items():
        g = take(name)
        m = keep & (g >= 0)
        rep = group_report(y[m], prob[m], pred[m], g[m], labels)
        per_pair = {p: group_report(y[m & (pair == p)], prob[m & (pair == p)], pred[m & (pair == p)],
                                    g[m & (pair == p)], labels) for p in np.unique(pair[m])}
        out["features"][name] = dict(overall=rep, per_pair=per_pair)
        fmt = [{k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()} for r in rep]
        print("\n" + fmt_table(fmt, cols, f"mid-clause, by {name}"))

    run = out["features"]["run"]["per_pair"]
    tbl = [{"pair": p, **{r["group"]: f"{r['base_rate']:.2f}/{r['mean_prob']:.2f}/{r['recall']:.2f}"
                          for r in rs}} for p, rs in run.items()]
    print("\n" + fmt_table(tbl, ["pair", *FEATURES["run"]],
                           "per pair, by run length: base rate / mean prob / recall"))
    dest = f"data/results/grammar_breakdown_{args.tag}.json"
    json.dump(out, open(dest, "w"), indent=1)
    print("wrote", dest)


if __name__ == "__main__":
    main()
