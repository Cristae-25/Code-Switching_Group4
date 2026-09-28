#!/usr/bin/env python
"""Fine-tune CANINE (character-level) for romanized-Hindi code-switch LID.

    ./.venv/bin/python scripts/train_canine_lid.py

Motivation: romanized Hinglish is the one case where dictionary lookup cannot
work, because there is no standard orthography - `woh`/`wo`/`vo` are all valid
spellings of the same word, so no word list can contain them all. CANINE has no
subword vocabulary at all; it reads Unicode codepoints directly, so spelling
variation is a surface detail rather than an out-of-vocabulary event.

Alignment: CANINE emits one position per codepoint, and its tokenizer has no
`word_ids()`. We map words to character spans ourselves, label *every*
character of a word with that word's label (more signal than first-char only),
and at inference average the logits across a word's characters.
"""
from __future__ import annotations

import os
import time

import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "models", "canine-hineng-lid")
BASE = "google/canine-s"
DATASET = "HuggMachas/Lince_benchmark_lid_hineng_train_valid"

LABELS = ["lang1", "lang2", "other", "ne", "fw", "mixed", "ambiguous", "unk"]
L2I = {l: i for i, l in enumerate(LABELS)}
EPOCHS, BATCH, LR, MAXLEN = 3, 8, 3e-5, 1024


def device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def spans(words: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Join words with spaces and return each word's [start, end) char span."""
    text, out, pos = "", [], 0
    for i, w in enumerate(words):
        if i:
            text += " "
            pos += 1
        out.append((pos, pos + len(w)))
        text += w
        pos += len(w)
    return text, out


def encode(rows, tok):
    texts, char_labels = [], []
    for r in rows:
        words, labels = r["sentence"], r["labels"]
        text, sp = spans(words)
        text = text[:MAXLEN - 2]
        lab = [-100] * len(text)
        for (a, b), l in zip(sp, labels):
            for c in range(a, min(b, len(text))):
                lab[c] = L2I.get(l, L2I["unk"])   # every char carries the label
        texts.append(text)
        char_labels.append(lab)

    enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
              max_length=MAXLEN)
    width = enc["input_ids"].shape[1]
    padded = []
    for lab in char_labels:
        # position 0 is [CLS]; char i sits at position i+1
        row = [-100] + lab[:width - 2]
        padded.append(row + [-100] * (width - len(row)))
    enc["labels"] = torch.tensor(padded)
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

    collate = lambda rows: encode(rows, tok)
    train = DataLoader(ds["train"], batch_size=BATCH, shuffle=True, collate_fn=collate)
    devl = DataLoader(ds["dev"], batch_size=BATCH, collate_fn=collate)
    print(f"train {len(ds['train']):,} sents | dev {len(ds['dev']):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=EPOCHS * len(train), pct_start=0.1)

    def evaluate() -> float:
        model.eval()
        ok = tot = 0
        with torch.no_grad():
            for b in devl:
                b = {k: v.to(dev) for k, v in b.items()}
                pred = model(**{k: v for k, v in b.items()
                                if k != "labels"}).logits.argmax(-1)
                m = b["labels"] != -100
                ok += (pred[m] == b["labels"][m]).sum().item()
                tot += m.sum().item()
        model.train()
        return ok / tot

    print(f"\n{'epoch':>6}{'loss':>10}{'dev char acc':>14}{'mins':>8}")
    t0, best = time.time(), 0.0
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
            if i % 100 == 0:
                print(f"    step {i}/{len(train)} loss {run/(i+1):.4f}", flush=True)
        acc = evaluate()
        print(f"{ep:>6}{run/len(train):>10.4f}{acc:>14.4f}{(time.time()-t0)/60:>8.1f}",
              flush=True)
        if acc > best:
            best = acc
            os.makedirs(OUT, exist_ok=True)
            model.save_pretrained(OUT)
            tok.save_pretrained(OUT)
            print(f"       saved -> {OUT}", flush=True)

    print(f"\nbest dev char accuracy: {best:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
