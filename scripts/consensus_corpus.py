#!/usr/bin/env python
"""Cross-check production (AnE-LID) tags against dict-freq across the corpus.

No hand labels. The two taggers share no evidence: AnE-LID is XLM-R-large
trained on four code-switch pairs, dict-freq compares wordfreq Zipf scores.
On human-annotated data (``eval_public.py``) tokens where they agree were
99.3-100% correct, and where they split AnE was right 95-98% of the time - so
the agreement rate here is a calibrated accuracy proxy for every row.

Each word token (anything alphabetic) falls in one of three buckets:

    agree     both taggers name the same language
    disagree  both name a language, and they differ   <- the uncertain tokens
    abstain   either side said ``other``, or the token is ``NE``

Scope: rows routed to ``ane`` (Latin-script L1 with English L2, and romanized
Hindi). dict-freq is weak on romanized Hindi (80.5% on LinCE), so expect lower
agreement there without it meaning AnE is wrong. German->French is dict-freq
in production, so it has nothing independent to be checked against here.

Outputs:
    data/consensus.parquet        row_id + per-token dict-freq language + bucket
    data/consensus_summary.json   the per-pair table printed below

    ./.venv/bin/python scripts/consensus_corpus.py                     # every row
    ./.venv/bin/python scripts/consensus_corpus.py --rows-per-pair 500  # sample
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

LATIN = ["Spanish", "Italian", "Malay", "Philippines", "German", "French",
         "Vietnamese", "Hindi"]


def _is_word(tok: str) -> bool:
    return any(c.isalpha() for c in tok)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="data/tagged.parquet")
    ap.add_argument("--rows-per-pair", type=int, default=0, help="0 = all rows")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top", type=int, default=8, help="disagreeing words to list per pair")
    ap.add_argument("--out", default="data/consensus.parquet")
    ap.add_argument("--summary", default="data/consensus_summary.json")
    args = ap.parse_args()

    import pandas as pd

    from src.dict_tagger import tag_frequency
    from src.dispatch import _hindi_is_devanagari

    df = pd.read_parquet(args.path, columns=["row_id", "first_language",
                                             "second_language", "tokens", "token_lang"])
    df = df[df.first_language.isin(LATIN) & (df.second_language == "English")]
    hindi = df.first_language == "Hindi"
    deva = df[hindi].tokens.map(lambda t: _hindi_is_devanagari(list(t)))
    df = df.drop(deva[deva].index)
    if args.rows_per_pair:
        per_pair = min(args.rows_per_pair, int(df.first_language.value_counts().min()))
        df = df.groupby("first_language").sample(per_pair, random_state=args.seed)
    print(f"{len(df):,} rows, {df.tokens.map(len).sum():,} tokens "
          "(German->French and Devanagari Hindi excluded)", flush=True)

    stats = {l: Counter() for l in df.first_language.unique()}
    pairs = {l: Counter() for l in stats}
    out_rows = []
    t0 = time.time()
    recs = list(df.itertuples(index=False))
    for b, r in enumerate(recs):
        l1 = r.first_language
        check_lang, bucket = [], []
        toks = list(r.tokens)
        for tok, prod, (_, a) in zip(toks, r.token_lang,
                                     tag_frequency(toks, l1, "English")):
            check_lang.append(a)
            if not _is_word(tok):
                bucket.append("")
                continue
            if a == "other" or prod not in (l1, "English"):   # other or NE
                k = "abstain"
            elif a == prod:
                k = "agree"
            else:
                k = "disagree"
                pairs[l1][(tok.lower(), prod, a)] += 1
            bucket.append(k)
            stats[l1][k] += 1
        out_rows.append(dict(row_id=r.row_id, dict_lang=check_lang, consensus=bucket))
        if b % 2000 == 0:
            rate = (b + 1) / (time.time() - t0)
            print(f"  {b + 1:,}/{len(recs):,} rows  {rate:.0f} rows/s", end="\r", flush=True)
    print(" " * 70)

    pd.DataFrame(out_rows).to_parquet(args.out, index=False)

    summary = {}
    print(f"{'pair':22}{'words':>9}{'agree':>9}{'disagree':>10}{'abstain':>9}"
          f"{'agree|decided':>15}")
    tot = Counter()
    for l1 in LATIN:
        if l1 not in stats:
            continue
        s = stats[l1]
        tot.update(s)
        n = sum(s.values())
        decided = s["agree"] + s["disagree"]
        summary[l1] = dict(words=n, agree=s["agree"], disagree=s["disagree"],
                           abstain=s["abstain"],
                           agree_rate_decided=s["agree"] / max(decided, 1),
                           top_disagreements=[
                               dict(word=w, production=p, dict_freq=a, count=c)
                               for (w, p, a), c in pairs[l1].most_common(args.top)])
        print(f"{l1 + '->English':22}{n:>9,}{s['agree']/n:>8.1%}{s['disagree']/n:>9.1%}"
              f"{s['abstain']/n:>9.1%}{s['agree']/max(decided, 1):>14.1%}")
    n = sum(tot.values())
    print(f"{'ALL':22}{n:>9,}{tot['agree']/n:>8.1%}{tot['disagree']/n:>9.1%}"
          f"{tot['abstain']/n:>9.1%}{tot['agree']/max(tot['agree'] + tot['disagree'], 1):>14.1%}")

    print("\nmost frequent disagreements (word: production -> dict-freq):")
    for l1, s in summary.items():
        items = ", ".join(f"{d['word']} {d['production'][:2]}->{d['dict_freq'][:2]} ×{d['count']}"
                          for d in s["top_disagreements"])
        print(f"  {l1:12} {items}")

    json.dump(dict(rows=len(df), rows_per_pair=args.rows_per_pair, pairs=summary),
              open(args.summary, "w"), indent=1, ensure_ascii=False)
    print(f"\nwrote {args.out} and {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
