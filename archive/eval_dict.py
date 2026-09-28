#!/usr/bin/env python
"""Score dictionary lookup against the models, on the same gold tokens."""
from __future__ import annotations

import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main() -> int:
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    from src.dict_tagger import overlap_stats, tag_frequency, tag_naive
    from src.model_tagger import tag_tokens_model
    from src.routing import MODEL_FOR, MODEL_L1_LABEL, MODEL_REPO

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    gold = json.load(open(os.path.join(ROOT, "data", "gold_eval.json")))
    cache = {}

    def mbert(lang, words):
        key = MODEL_FOR.get(lang, "spaeng")
        if key not in cache:
            repo = MODEL_REPO[key]
            cache[key] = (AutoTokenizer.from_pretrained(repo),
                          AutoModelForTokenClassification.from_pretrained(repo).eval().to(dev))
        tk, md = cache[key]
        enc = tk(words, is_split_into_words=True, return_tensors="pt", truncation=True)
        ids = enc.word_ids(0)
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            pred = md(**enc).logits.argmax(-1)[0].tolist()
        out, seen = [], set()
        for pos, wid in enumerate(ids):
            if wid is not None and wid not in seen:
                seen.add(wid)
                out.append(md.config.id2label[pred[pos]])
        out += ["unk"] * (len(words) - len(out))
        lab = MODEL_L1_LABEL[key]
        return ["EN" if t == "en" else ("L1" if t == lab else "?") for t in out]

    names = ["dict-naive", "dict-freq", "fastText", "mBERT"]
    print(f"{'language':13}{'n':>5}" + "".join(f"{x:>12}" for x in names))
    print("-" * 66)
    tot = {k: [0, 0] for k in names}
    ov = {}

    for lang, rows in gold.items():
        sc = {k: 0 for k in names}
        n = 0
        agg = dict(words=0, both=0, neither=0, only_l1=0, only_l2=0)
        for text, glab in rows:
            w, g = text.split(), glab.split()
            preds = {
                "dict-naive": [("EN" if l == "English" else "L1")
                               for _, l in tag_naive(w, lang, "English")],
                "dict-freq": [("EN" if l == "English" else ("?" if l == "other" else "L1"))
                              for _, l in tag_frequency(w, lang, "English")],
                "fastText": [("EN" if l == "English" else "L1")
                             for _, l in tag_tokens_model(w, lang, "English")],
                "mBERT": mbert(lang, w),
            }
            s = overlap_stats(w, lang, "English")
            for k in agg:
                agg[k] += s[k]
            for i, gg in enumerate(g):
                if gg == "NE":
                    continue
                n += 1
                for k in names:
                    sc[k] += preds[k][i] == gg
        ov[lang] = agg
        for k in names:
            tot[k][0] += sc[k]
            tot[k][1] += n
        print(f"{lang:13}{n:>5}" + "".join(f"{sc[k]/n:>11.0%} " for k in names))

    print("-" * 66)
    print(f"{'OVERALL':13}{tot['mBERT'][1]:>5}"
          + "".join(f"{tot[k][0]/tot[k][1]:>11.0%} " for k in names))

    print(f"\n\nWHY: how many words are in BOTH dictionaries?")
    print(f"{'language':13}{'words':>7}{'both':>8}{'only L1':>9}{'only EN':>9}{'neither':>9}")
    for lang, a in ov.items():
        w = a["words"] or 1
        print(f"{lang:13}{a['words']:>7}{a['both']/w:>7.0%}{a['only_l1']/w:>9.0%}"
              f"{a['only_l2']/w:>9.0%}{a['neither']/w:>9.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
