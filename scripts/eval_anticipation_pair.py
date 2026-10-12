#!/usr/bin/env python
"""Score a saved anticipation checkpoint on one exact language PAIR.

``iter_rows`` filters on ``first_language``, which merges German->English with
German->French. That merge hides the single most informative pair in the corpus:
German->French is the only one whose second language is not English, so it is
the only evidence that a model learned "switch" rather than "switch to English".

    ./.venv/bin/python scripts/eval_anticipation_pair.py \
        --ckpt models/anticipation_zeroshot.pt --l1 German --l2 French
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.anticipation import N_DUR  # noqa: E402
from src.models import build_model  # noqa: E402
from src.metrics import prf  # noqa: E402
from src.streaming import (COLUMNS, DEFAULT_PATHS, VOCAB_PATHS,  # noqa: E402
                           SwitchWindowDataset, collate, example_for, iter_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/anticipation_causal.pt")
    ap.add_argument("--vocab", default=None, help="default: the vocabulary for the checkpoint's unit")
    ap.add_argument("--path", default=None, help="default: the parquet for the checkpoint's unit")
    ap.add_argument("--l1", required=True)
    ap.add_argument("--l2", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--names", choices=["script_words", "transparent", "script", "hybrid", "language"],
                    default="script_words",
                    help="how named entities count (see src/segments.py); needs an "
                         "is_entity column, else names count as tagged")
    args = ap.parse_args()
    import src.anticipation as _ant
    _ant.NAMES = args.names

    import json
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ck["args"]
    unit = a.get("unit", "word")          # checkpoints from before --unit are word models
    path = args.path or a.get("path") or DEFAULT_PATHS[unit]
    state = ck["state"]
    pretrained = a.get("model", "scratch") != "scratch"
    vocab = None if pretrained else json.load(open(args.vocab or VOCAB_PATHS[unit]))
    model = build_model(a, ck["vocab_size"], n_pairs=14,
                        use_pair=state.get("pair_emb.weight") is not None,
                        use_lang=state.get("lang_emb.weight") is not None,
                        n_dur=(state["head_dur.weight"].shape[0] if "head_dur.weight" in state
                               else N_DUR), pretrained=False)   # an ordinal head has no .weight
    model.load_state_dict(state)
    model.eval().to(args.device)
    tau = ck["tau"]

    sep_id = None
    if a.get("turn_sep") and pretrained:
        from transformers import AutoTokenizer
        from src.xlmr_model import BACKBONES
        sep_id = AutoTokenizer.from_pretrained(a.get("backbone") or BACKBONES[a["model"]]).sep_token_id
    ds = SwitchWindowDataset(path, vocab, split=args.split,
                             window=a["window"], stride=a["stride"], unit=unit,
                             predict_at=a.get("predict_at", "piece"), turn_sep=a.get("turn_sep", False),
                             sep_id=sep_id,
                             lang_input="prev_word" if a.get("prev_lang") else "current")
    make = example_for(unit)
    logits, truth = [], []
    batch = []
    with torch.no_grad():
        for row in iter_rows(path, split=args.split, columns=COLUMNS[unit]):
            if row["first_language"] != args.l1 or row["second_language"] != args.l2:
                continue
            ex = make(row, turn_sep=True) if a.get("turn_sep") else make(row)
            if ex is None:
                continue
            batch.extend(ds._windows(ex))
            if len(batch) < 32:
                continue
            b, batch = collate(batch[:32]), batch[32:]
            sw, _ = model(b["word_ids"].to(args.device), b["lang_ids"].to(args.device))
            m = b["sw_mask"]
            logits.append(sw[m].float().numpy())
            truth.append(b["y_sw"][m].numpy())
    if batch:
        b = collate(batch)
        with torch.no_grad():
            sw, _ = model(b["word_ids"].to(args.device), b["lang_ids"].to(args.device))
        m = b["sw_mask"]
        logits.append(sw[m].float().numpy())
        truth.append(b["y_sw"][m].numpy())

    prob = 1 / (1 + np.exp(-np.concatenate(logits)))
    y = np.concatenate(truth)
    s = prf(y, prob >= tau)
    print(f"{args.l1} -> {args.l2}  ({args.split}, tau={tau:.3f})")
    print(f"  n {s['n']:,}  base {s['base_rate']:.4f}  "
          f"P {s['precision']:.3f}  R {s['recall']:.3f}  F1 {s['f1']:.3f}")


if __name__ == "__main__":
    main()
