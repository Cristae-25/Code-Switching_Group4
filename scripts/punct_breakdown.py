#!/usr/bin/env python
"""Do switches sit behind punctuation, and does the model catch those? (no GPU)

    ./.venv/bin/python scripts/punct_breakdown.py [xlmr_6pairs] [--limit 5000]

A switch label sits on the last language-bearing piece *before* any
punctuation (``anticipatory_labels``: NONE slots are transparent). In
``I went home, 然后`` the decision is made at ``home``, and the causal model has
not yet seen the comma. This script buckets every labelled position by what
lies between it and the next language-bearing piece:

* ``turn``          the next piece starts a new utterance (speaker turn)
* ``sentence_end``  the gap holds . ! ? 。 ！ ？ …
* ``comma``         the gap holds , ; : 、 ， ； ：
* ``other_gap``     the gap holds something else (digits, quotes, brackets)
* ``adjacent``      nothing between them (a space at most)

Part A (labels, every split): positions, switches, each bucket's share of all
switches, and the switch rate with a Wilson interval, overall and per pair.
Part B (``data/preds_<TAG>.npz``, test split): the model's P/R/F1 per bucket at
the run's own threshold, on all scored positions and on the shared (word-end)
positions. Writes ``data/punct_breakdown_<TAG>.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.anticipation import NONE  # noqa: E402
from src.evaluation import PAIR_KEYS  # noqa: E402
from src.metrics import fmt_table, prf, wilson  # noqa: E402
from src.subword import MARKERS  # noqa: E402

BUCKETS = ("adjacent", "comma", "sentence_end", "other_gap", "turn")
ADJ, COMMA, SENT, OTHER, TURN = range(5)
NO_LABEL = -1
SENT_CHARS = set(".!?。！？…")
COMMA_CHARS = set(",;:、，；：")


def _gap_kind(gap_pieces) -> int:
    text = "".join(gap_pieces)
    for m in MARKERS:
        text = text.replace(m, "")
    chars = set(text.strip())
    if not chars:
        return ADJ
    if chars & SENT_CHARS:
        return SENT
    if chars & COMMA_CHARS:
        return COMMA
    return OTHER


def gap_buckets(pieces, langs, utts=None) -> np.ndarray:
    """Bucket code per piece; NO_LABEL where no language-bearing piece follows."""
    langs = np.asarray(langs)
    out = np.full(len(langs), NO_LABEL, dtype=np.int8)
    idx = np.flatnonzero(langs != NONE)
    for a, b in zip(idx[:-1], idx[1:]):
        if utts is not None and utts[a] != utts[b]:
            out[a] = TURN
        elif b == a + 1:
            out[a] = ADJ
        else:
            out[a] = _gap_kind(pieces[a + 1:b])
    return out


def scan(path: str, limit: int | None = None):
    """Yield (row_id, pair, buckets, y_sw) for every row of the subword parquet."""
    f = pq.ParquetFile(path)
    cols = ["row_id", "first_language", "second_language", "pieces", "piece_lang", "y_sw", "piece_utt"]
    seen = 0
    for batch in f.iter_batches(batch_size=4096, columns=cols):
        d = batch.to_pydict()
        for i in range(len(d["row_id"])):
            pair = f"{d['first_language'][i]}->{d['second_language'][i]}"
            yield (d["row_id"][i], pair,
                   gap_buckets(d["pieces"][i], d["piece_lang"][i], d["piece_utt"][i]),
                   np.asarray(d["y_sw"][i], dtype=np.int8))
            seen += 1
            if limit and seen >= limit:
                return


def part_a(counts: dict) -> dict:
    """counts[pair] = [n per bucket, switches per bucket]."""
    out = {}
    for pair, (n, k) in sorted(counts.items()):
        tot = k.sum()
        out[pair] = {b: dict(n=int(n[j]), switches=int(k[j]),
                             share_of_switches=float(k[j] / tot) if tot else 0.0,
                             switch_rate=float(k[j] / n[j]) if n[j] else 0.0,
                             rate_ci=wilson(int(k[j]), int(n[j])))
                     for j, b in enumerate(BUCKETS)}
    return out


def part_b(tag: str, lookup: dict, ys: dict, data_dir: str = "data") -> dict:
    """Join the saved test predictions to the buckets on (row_id, piece position)."""
    z = np.load(os.path.join(data_dir, f"preds_{tag}.npz"))
    tau = json.load(open(os.path.join(data_dir, f"anticipation_{tag}.json")))["tau"]
    rows, pos, y = z["test_row"], z["test_pos"], z["test_y"]
    pred = 1 / (1 + np.exp(-z["test_logit"])) >= tau
    bucket = np.array([lookup[r][p] for r, p in zip(rows.tolist(), pos.tolist())], dtype=np.int8)
    labels = np.array([ys[r][p] for r, p in zip(rows.tolist(), pos.tolist())], dtype=np.int8)
    if not np.array_equal(labels, y):
        raise SystemExit("parquet y_sw disagrees with the saved predictions' labels")
    if (bucket == NO_LABEL).any():
        raise SystemExit(f"{int((bucket == NO_LABEL).sum())} scored positions have no bucket")
    pair_names = np.array(PAIR_KEYS)[z["test_pair"]] if "test_pair" in z else None
    res = {"tau": float(tau), "overall": prf(y, pred)}
    for view, m in (("all", np.ones(len(y), bool)), ("shared", z["test_shared"])):
        res[view] = {b: prf(y[m & (bucket == j)], pred[m & (bucket == j)]) for j, b in enumerate(BUCKETS)}
        if pair_names is not None:
            res[f"{view}_per_pair"] = {
                p: {b: prf(y[m & (bucket == j) & (pair_names == p)], pred[m & (bucket == j) & (pair_names == p)])
                    for j, b in enumerate(BUCKETS)}
                for p in np.unique(pair_names[m])}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", nargs="?", default="xlmr_6pairs")
    ap.add_argument("--path", default="data/subword.parquet")
    ap.add_argument("--limit", type=int, default=None, help="rows to scan (Part B needs all)")
    args = ap.parse_args()

    counts, lookup, ys = {}, {}, {}
    for row_id, pair, b, y in scan(args.path, args.limit):
        n, k = counts.setdefault(pair, [np.zeros(5, np.int64), np.zeros(5, np.int64)])
        m = (b != NO_LABEL) & (y >= 0)
        np.add.at(n, b[m], 1)
        np.add.at(k, b[m & (y == 1)], 1)
        if row_id % 10 == 9:                      # test split: kept for the join
            lookup[row_id], ys[row_id] = b, y
    total = [sum(c[0] for c in counts.values()), sum(c[1] for c in counts.values())]
    a = part_a({**counts, "ALL": total})

    def a_rows(stats):
        return [{"bucket": b, "positions": s["n"], "switches": s["switches"],
                 "share_of_switches": f"{s['share_of_switches']:.3f}",
                 "switch_rate": f"{s['switch_rate']:.3f}",
                 "rate_95ci": f"[{s['rate_ci'][0]:.3f}, {s['rate_ci'][1]:.3f}]"} for b, s in stats.items()]
    print(fmt_table(a_rows(a["ALL"]), ["bucket", "positions", "switches", "share_of_switches",
                                       "switch_rate", "rate_95ci"], "Part A: all pairs, all splits"))
    share_rows = [{"pair": p, **{b: f"{s[b]['share_of_switches']:.2f} ({s[b]['switch_rate']:.2f})"
                                 for b in BUCKETS}} for p, s in a.items() if p != "ALL"]
    print(fmt_table(share_rows, ["pair", *BUCKETS], "Part A per pair: share of switches (switch rate)"))

    out = {"part_a": a}
    if os.path.exists(f"data/preds_{args.tag}.npz") and not args.limit:
        b = part_b(args.tag, lookup, ys)
        out["part_b"] = b
        print(f"\nPart B: {args.tag}, test, tau={b['tau']:.3f}, pooled F1 {b['overall']['f1']:.4f}")
        for view in ("all", "shared"):
            rows = [{"bucket": k, "n": v["n"], "base_rate": f"{v['base_rate']:.3f}",
                     "precision": f"{v['precision']:.3f}", "recall": f"{v['recall']:.3f}",
                     "f1": f"{v['f1']:.3f}", "missed_switches": v["fn"]} for k, v in b[view].items()]
            print(fmt_table(rows, ["bucket", "n", "base_rate", "precision", "recall", "f1",
                                   "missed_switches"], f"Part B ({view} positions)"))
        if "all_per_pair" in b:
            pp = [{"pair": p, **{k: f"{v['f1']:.2f}/{v['recall']:.2f}" for k, v in s.items()}}
                  for p, s in b["all_per_pair"].items()]
            print(fmt_table(pp, ["pair", *BUCKETS], "Part B per pair: F1/recall (all positions)"))
    dest = f"data/punct_breakdown_{args.tag}.json"
    json.dump(out, open(dest, "w"), indent=1, default=float)
    print("wrote", dest)


if __name__ == "__main__":
    main()
