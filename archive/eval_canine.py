#!/usr/bin/env python
"""Evaluate CANINE on romanized Hindi against every other technique.

Two measurements:
  in-domain    LinCE hineng dev, word level (same distribution as training)
  out-domain   our gold Hindi tokens from SwitchLingua (clean generated prose)

The gap between them is the domain-shift cost, which matters because LinCE is
Twitter text and SwitchLingua is not.
"""
from __future__ import annotations

import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
CANINE = os.path.join(ROOT, "models", "canine-hineng-lid")


def device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def canine_word_labels(tok, model, words, dev):
    """Predict one label per word by averaging logits over its characters."""
    from train_canine_lid import spans
    text, sp = spans(words)
    enc = tok(text[:1022], return_tensors="pt", truncation=True, max_length=1024)
    enc = {k: v.to(dev) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits[0]          # [seq, labels]
    out = []
    for a, b in sp:
        lo, hi = a + 1, min(b + 1, logits.shape[0])   # +1 for [CLS]
        if lo >= logits.shape[0] or lo >= hi:
            out.append("unk")
            continue
        out.append(model.config.id2label[logits[lo:hi].mean(0).argmax().item()])
    return out


def main() -> int:
    from datasets import load_dataset
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    from src.dict_tagger import tag_frequency
    from src.model_tagger import tag_tokens_model
    from src.routing import MODEL_REPO

    if not os.path.isdir(CANINE):
        sys.exit(f"{CANINE} not found - run scripts/train_canine_lid.py first.")
    dev = device()
    ct = AutoTokenizer.from_pretrained(CANINE)
    cm = AutoModelForTokenClassification.from_pretrained(CANINE).eval().to(dev)

    # --- in-domain: LinCE dev, word level -------------------------------
    ds = load_dataset("HuggMachas/Lince_benchmark_lid_hineng_train_valid")["dev"]
    ok = tot = 0
    for ex in ds:
        pred = canine_word_labels(ct, cm, ex["sentence"], dev)
        for p, g in zip(pred, ex["labels"]):
            if g in ("lang1", "lang2"):        # score only the two languages
                tot += 1
                ok += p == g
    print(f"in-domain  LinCE hineng dev : {ok/tot:.1%}  ({tot:,} word tokens)")

    # --- out-of-domain: our gold Hindi ----------------------------------
    gold = json.load(open(os.path.join(ROOT, "data", "gold_eval.json")))["Hindi"]
    mt = AutoTokenizer.from_pretrained(MODEL_REPO["hineng"])
    mm = AutoModelForTokenClassification.from_pretrained(
        MODEL_REPO["hineng"]).eval().to(dev)

    def mbert(words):
        enc = mt(words, is_split_into_words=True, return_tensors="pt", truncation=True)
        ids = enc.word_ids(0)
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            pr = mm(**enc).logits.argmax(-1)[0].tolist()
        o, seen = [], set()
        for pos, wid in enumerate(ids):
            if wid is not None and wid not in seen:
                seen.add(wid)
                o.append(mm.config.id2label[pr[pos]])
        o += ["unk"] * (len(words) - len(o))
        return ["EN" if t == "en" else ("L1" if t == "hin" else "?") for t in o]

    sc = {k: 0 for k in ["dict-freq", "fastText", "mBERT", "CANINE"]}
    n = 0
    for text, glab in gold:
        words, golds = text.split(), glab.split()
        preds = {
            "dict-freq": ["EN" if l == "English" else ("?" if l == "other" else "L1")
                          for _, l in tag_frequency(words, "Hindi", "English")],
            "fastText": ["EN" if l == "English" else "L1"
                         for _, l in tag_tokens_model(words, "Hindi", "English")],
            "mBERT": mbert(words),
            "CANINE": ["EN" if t == "lang1" else ("L1" if t == "lang2" else "?")
                       for t in canine_word_labels(ct, cm, words, dev)],
        }
        for i, g in enumerate(golds):
            if g == "NE":
                continue
            n += 1
            for k in sc:
                sc[k] += preds[k][i] == g

    print(f"\nout-of-domain  SwitchLingua romanized Hindi ({n} gold tokens):")
    for k in ["dict-freq", "fastText", "mBERT", "CANINE"]:
        print(f"  {k:12} {sc[k]/n:>6.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
