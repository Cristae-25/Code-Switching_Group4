#!/usr/bin/env python
"""Fine-tune XLM-RoBERTa for code-switch token-level language ID.

    ./.venv/bin/python scripts/train_xlmr_lid.py

Tests the hypothesis that XLM-R's CommonCrawl pretraining (2.5TB, 100 langs)
beats mBERT's Wikipedia pretraining on low-resource languages - specifically
Malay and Tagalog, where our mBERT-based tagger is weakest.

Trains on LinCE Hindi-English LID (the one split still loadable - the official
`lince` dataset uses a loading script that `datasets` 4.x dropped). Evaluated
two ways afterwards by eval_xlmr.py: in-language on Hindi, and zero-shot
transfer to Malay/Tagalog.

Manual training loop rather than Trainer: the Trainer API moved in
transformers 5.x and a hand-rolled loop is fewer moving parts.
"""
from __future__ import annotations

import os
import time

import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "models", "xlmr-hineng-lid")
BASE = "xlm-roberta-base"
DATASET = "HuggMachas/Lince_benchmark_lid_hineng_train_valid"

LABELS = ["lang1", "lang2", "other", "ne", "fw", "mixed", "ambiguous", "unk"]
L2I = {l: i for i, l in enumerate(LABELS)}
EPOCHS, BATCH, LR, MAXLEN = 4, 16, 2e-5, 256


def device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def encode(batch, tok):
    """Tokenise pre-split words and align labels to the FIRST sub-token."""
    enc = tok(batch["sentence"], is_split_into_words=True, truncation=True,
              max_length=MAXLEN, padding=True, return_tensors="pt")
    all_labels = []
    for i, labels in enumerate(batch["labels"]):
        ids, seen, out = enc.word_ids(i), set(), []
        for wid in ids:
            if wid is None or wid in seen:
                out.append(-100)          # special token or continuation
            else:
                seen.add(wid)
                out.append(L2I.get(labels[wid], L2I["unk"]))
        all_labels.append(out)
    enc["labels"] = torch.tensor(all_labels)
    return enc


def main() -> int:
    from datasets import load_dataset
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    dev = device()
    print(f"device: {dev} | base: {BASE}")
    ds = load_dataset(DATASET)
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForTokenClassification.from_pretrained(
        BASE, num_labels=len(LABELS),
        id2label={i: l for l, i in L2I.items()}, label2id=L2I).to(dev)

    def collate(rows):
        return encode({"sentence": [r["sentence"] for r in rows],
                       "labels": [r["labels"] for r in rows]}, tok)

    train = DataLoader(ds["train"], batch_size=BATCH, shuffle=True, collate_fn=collate)
    devl = DataLoader(ds["dev"], batch_size=BATCH, collate_fn=collate)
    print(f"train {len(ds['train']):,} sents | dev {len(ds['dev']):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * len(train)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=steps, pct_start=0.1)

    def evaluate() -> float:
        model.eval()
        ok = tot = 0
        with torch.no_grad():
            for b in devl:
                b = {k: v.to(dev) for k, v in b.items()}
                pred = model(**{k: v for k, v in b.items() if k != "labels"}
                             ).logits.argmax(-1)
                m = b["labels"] != -100
                ok += (pred[m] == b["labels"][m]).sum().item()
                tot += m.sum().item()
        model.train()
        return ok / tot

    print(f"\n{'epoch':>6}{'loss':>10}{'dev acc':>10}{'mins':>8}")
    t0 = time.time()
    best = 0.0
    for ep in range(1, EPOCHS + 1):
        run = 0.0
        for i, b in enumerate(train):
            b = {k: v.to(dev) for k, v in b.items()}
            loss = model(**b).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            run += loss.item()
            if i % 50 == 0:
                print(f"    step {i}/{len(train)} loss {run/(i+1):.4f}", flush=True)
        acc = evaluate()
        print(f"{ep:>6}{run/len(train):>10.4f}{acc:>10.4f}{(time.time()-t0)/60:>8.1f}",
              flush=True)
        if acc > best:
            best = acc
            os.makedirs(OUT, exist_ok=True)
            model.save_pretrained(OUT)
            tok.save_pretrained(OUT)
            print(f"       saved (best so far) -> {OUT}", flush=True)

    print(f"\nbest dev token accuracy: {best:.4f}")
    print(f"model at {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
