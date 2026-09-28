#!/usr/bin/env python
"""Compare the named-entity policies on hand-picked edge cases and on the corpus.

    ./.venv/bin/python scripts/compare_name_strategies.py            # both reports
    ./.venv/bin/python scripts/compare_name_strategies.py --refresh  # re-run the models
                                                                      # on the edge cases

Part 1 labels the 20 sentences in ``tests/fixtures/name_cases.json`` (real
AnE-NER / AnE-LID output, frozen) under every policy and prints each sentence's
second-language runs, so a fake switch is visible as an extra short run.

Part 2 estimates the corpus-wide effect from the v1 labels
(``data/hf/tagged.parquet``), whose names are ``NE``. v1 never kept AnE-LID's
label for a name, so a name's own label is approximated: script in non-Latin
rows, dict-freq in Latin rows. That affects ``language`` and ``hybrid`` (which
consult it); ``transparent`` and ``script`` never use it and are exact.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
FIXTURE = os.path.join(ROOT, "tests", "fixtures", "name_cases.json")
POLICIES = ["transparent", "language", "hybrid", "script", "script_words"]
SHORT = {"transparent": "T", "language": "A", "hybrid": "H", "script": "S", "script_words": "P"}

CASES = [
    ("severance_in_spanish", "Spanish", "English", "Anoche vi Severance Season 2 y me encantó"),
    ("messi_in_spanish", "Spanish", "English", "Ayer Leo Messi marcó dos goles , it was amazing"),
    ("messi_in_english_clause", "Spanish", "English", "I watched Leo Messi play yesterday , fue increíble"),
    ("real_madrid_in_french", "French", "English", "Après la victoire contre le Real Madrid , I'm so happy"),
    ("psg_at_switch", "French", "English", "I think PSG va gagner ce soir"),
    ("union_europeenne", "French", "English", "L'Union européenne a voté , which is big news"),
    ("arabic_script_name", "Arabic", "English", "أنا تابعت خبر ترامب عن الفيزا , honestly it is complicated"),
    ("severance_in_arabic", "Arabic", "English", "شاهدت Severance Season 2 أمس"),
    ("messi_latin_in_arabic", "Arabic", "English", "شاهدت مباراة Messi أمس , it was great"),
    ("got_in_cantonese", "Cantonese", "English", "我哋琴日睇咗 Game of Thrones ， really good"),
    ("han_place_name", "Chinese", "English", "昨天我去參觀了中關村論壇"),
    ("han_messi", "Chinese", "English", "梅西昨天進了兩球 , amazing"),
    ("iphone_in_japanese", "Japanese", "English", "ホンダは新しい iPhone 16 Pro を発表した"),
    ("acronym_in_korean", "Korean", "English", "WWE 가 이번에 경기를 발표했어"),
    ("acronyms_in_german", "German", "English", "Die EU debattiert , especially about NATO"),
    ("weekday_false_positive", "Italian", "English", "Credo che domenica il Monza farà bene , you know ?"),
    ("brand_in_russian", "Russian", "English", "В прошлом году Marvel сняли фильм , you know"),
    ("bollywood_in_hinglish", "Hindi", "English", "Kal maine Shraddha Kapoor ki movie dekhi , it was good"),
    ("title_in_malay", "Malay", "English", "Kajian di The University of Texas menunjukkan hasil"),
    ("name_in_german_french", "German", "French", "Ich habe Nel gelesen , mais le chemin reste long"),
]


def refresh() -> None:
    from src import ane_tagger
    from src.dispatch import row_strategy, tag_tokens
    from src.tokenizer import tokenize_mixed
    toks = [tokenize_mixed(s) for *_, s in CASES]
    ner, lid = ane_tagger.entity_mask_batch(toks), ane_tagger.labels_batch(toks)
    out = []
    for (cid, l1, l2, _), t, m, a in zip(CASES, toks, ner, lid):
        strat = row_strategy(t, l1, l2)
        base = [l for _, l in tag_tokens(t, l1, l2, ane_labels=a if strat == "ane" else None)]
        out.append(dict(id=cid, l1=l1, l2=l2, tokens=t, is_entity=m, ane=a,
                        strategy=strat, base=base))
    json.dump(out, open(FIXTURE, "w"), ensure_ascii=False, indent=1)
    print(f"wrote {FIXTURE}")


def edge_cases() -> None:
    from src.dispatch import tag_tokens
    from src.segments import NE, absorb_other, l2_word_runs, repair_spans, resolve_names
    cases = json.load(open(FIXTURE))
    print("PART 1 - edge cases. Name tokens are marked *; each policy line shows how the")
    print("sentence is labelled and its second-language runs (a fake switch = an extra run).\n")
    for c in cases:
        l1, l2, toks = c["l1"], c["l2"], c["tokens"]
        tagged = tag_tokens(toks, l1, l2, ane_labels=c["ane"], is_entity=c["is_entity"])
        ent = repair_spans(toks, c["is_entity"], l1, l2)
        print(f"[{c['id']}]  {l1} -> {l2}")
        for pol in POLICIES:
            out = resolve_names(tagged, c["is_entity"], l1, l2, pol)
            code = lambda lg: "1" if lg == l1 else "2" if lg == l2 else "·" if lg == NE else "o"
            shown = " ".join(f"{t}{'*' if e else ''}/{code(lg)}" for (t, lg), e in zip(out, ent)
                             if any(ch.isalpha() for ch in t))
            runs = l2_word_runs(absorb_other(out), l2)
            print(f"   {SHORT[pol]} {pol:12} runs={str(runs):14} {shown}")
        print()


def corpus(path: str, per_pair: int) -> None:
    import numpy as np
    import pandas as pd

    from src.dict_tagger import tag_frequency
    from src.segments import NE, absorb_other, l2_word_runs, resolve_names
    from src.unicode_tagger import tag_tokens_unicode

    if not os.path.exists(path):
        print(f"(part 2 skipped: {path} not found)")
        return
    f = pd.read_parquet(path, columns=["row_id", "first_language", "second_language",
                                       "tokens", "token_lang", "method"])
    f = f.groupby(["first_language", "second_language"]).sample(
        n=per_pair, random_state=0, replace=False) if per_pair else f
    print(f"PART 2 - corpus estimate from v1 labels, {len(f):,} rows "
          f"({per_pair or 'all'} per pair). Switch count and median run per policy.\n")
    rows = []
    for (l1, l2), g in f.groupby(["first_language", "second_language"]):
        res = {p: [] for p in POLICIES}
        for toks, labs, method in zip(g.tokens, g.token_lang, g.method):
            toks, labs = list(toks), list(labs)
            ent = [x == NE for x in labs]
            if any(ent):                                  # reconstruct a name's own label
                fill = (tag_tokens_unicode(toks, l1, l2) if method == "unicode"
                        else tag_frequency(toks, l1, l2))
                labs = [fl if e else x for x, (_, fl), e in zip(labs, fill, ent)]
            tagged = list(zip(toks, labs))
            for p in POLICIES:
                res[p].extend(l2_word_runs(absorb_other(resolve_names(tagged, ent, l1, l2, p)), l2))
        d = dict(pair=f"{l1}->{l2}")
        for p in POLICIES:
            d[f"{SHORT[p]}_sw"] = len(res[p])
        for p in POLICIES:
            d[f"{SHORT[p]}_med"] = float(np.median(res[p])) if res[p] else 0.0
        d["S_vs_T"] = f"{d['S_sw'] / max(d['T_sw'], 1) - 1:+.0%}"
        d["P_vs_T"] = f"{d['P_sw'] / max(d['T_sw'], 1) - 1:+.0%}"
        rows.append(d)
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    tot = {SHORT[p]: t[f"{SHORT[p]}_sw"].sum() for p in POLICIES}
    print("\nALL switches: " + "  ".join(f"{k} {v:,}" for k, v in tot.items())
          + f"   (S {tot['S'] / tot['T'] - 1:+.1%}, P {tot['P'] / tot['T'] - 1:+.1%} vs T)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-run AnE on the edge cases")
    ap.add_argument("--parquet", default=os.path.join(ROOT, "data", "hf", "tagged.parquet"))
    ap.add_argument("--per-pair", type=int, default=2000, help="corpus rows per pair (0 = all)")
    ap.add_argument("--skip-corpus", action="store_true")
    args = ap.parse_args()
    if args.refresh:
        refresh()
    edge_cases()
    if not args.skip_corpus:
        corpus(args.parquet, args.per_pair)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
