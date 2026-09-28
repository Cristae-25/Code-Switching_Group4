#!/usr/bin/env python
"""Compare XLM-R against the current production taggers on the gold eval set.

    ./.venv/bin/python scripts/eval_xlmr.py

Three systems, same 324 hand-labelled tokens (data/gold_eval.json):

  fastText   the constrained lid.176 baseline
  mBERT      the LinCE models currently in production (hineng + spaeng)
  XLM-R      fine-tuned here on LinCE hineng

Two questions:
  in-language  does XLM-R beat mBERT on Hindi, the task it was trained on?
  zero-shot    does it transfer better to Malay/Tagalog, where mBERT is weakest?
"""
from __future__ import annotations

import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
XLMR = os.path.join(ROOT, "models", "xlmr-hineng-lid")
GOLD = os.path.join(ROOT, "data", "gold_eval.json")


def device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def word_labels(tok, model, words, dev):
    """Label per word; the first sub-token of each word decides."""
    enc = tok(words, is_split_into_words=True, return_tensors="pt",
              truncation=True, max_length=512)
    ids = enc.word_ids(0)
    enc = {k: v.to(dev) for k, v in enc.items()}
    with torch.no_grad():
        pred = model(**enc).logits.argmax(-1)[0].tolist()
    out, seen = [], set()
    for pos, wid in enumerate(ids):
        if wid is not None and wid not in seen:
            seen.add(wid)
            out.append(model.config.id2label[pred[pos]])
    return out + ["unk"] * (len(words) - len(out))


def main() -> int:
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    from src.model_tagger import tag_tokens_model
    from src.routing import MODEL_FOR, MODEL_L1_LABEL, MODEL_REPO

    if not os.path.isdir(XLMR):
        sys.exit(f"{XLMR} not found - run scripts/train_xlmr_lid.py first.")
    dev = device()
    gold = json.load(open(GOLD))

    xt = AutoTokenizer.from_pretrained(XLMR)
    xm = AutoModelForTokenClassification.from_pretrained(XLMR).eval().to(dev)

    mbert = {}
    for key, repo in MODEL_REPO.items():
        mbert[key] = (AutoTokenizer.from_pretrained(repo),
                      AutoModelForTokenClassification.from_pretrained(repo).eval().to(dev))

    print(f"{'language':13}{'n':>5}{'fastText':>10}{'mBERT':>9}{'XLM-R':>9}   winner")
    print("-" * 60)
    totals = {"fastText": [0, 0], "mBERT": [0, 0], "XLM-R": [0, 0]}

    for lang, rows in gold.items():
        score = {k: 0 for k in totals}
        n = 0
        # which mBERT model does production route this language to?
        key = MODEL_FOR.get(lang, "spaeng")
        mt, mm = mbert[key]
        l1lab = MODEL_L1_LABEL[key]

        for text, glab in rows:
            words, golds = text.split(), glab.split()
            ft = ["EN" if l == "English" else "L1"
                  for _, l in tag_tokens_model(words, lang, "English")]
            mb = ["EN" if t == "en" else ("L1" if t == l1lab else "?")
                  for t in word_labels(mt, mm, words, dev)]
            # XLM-R was trained with LinCE's lang1=English / lang2=Hindi scheme;
            # lang2 generalises to "the non-English language" for transfer.
            xr = ["EN" if t == "lang1" else ("L1" if t == "lang2" else "?")
                  for t in word_labels(xt, xm, words, dev)]
            for i, g in enumerate(golds):
                if g == "NE":
                    continue
                n += 1
                score["fastText"] += ft[i] == g
                score["mBERT"] += mb[i] == g
                score["XLM-R"] += xr[i] == g

        for k in totals:
            totals[k][0] += score[k]
            totals[k][1] += n
        best = max(score, key=score.get)
        print(f"{lang:13}{n:>5}" + "".join(f"{score[k]/n:>9.0%} " for k in
              ["fastText", "mBERT", "XLM-R"]) + f"  {best}")

    print("-" * 60)
    print(f"{'OVERALL':13}{totals['fastText'][1]:>5}" + "".join(
        f"{totals[k][0]/totals[k][1]:>9.0%} " for k in ["fastText", "mBERT", "XLM-R"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
