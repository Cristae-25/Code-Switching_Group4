#!/usr/bin/env python
"""Write kaggle/train_pretrained.ipynb: fine-tune a causal pretrained encoder on a Kaggle GPU.

    ./.venv/bin/python scripts/build_pretrained_notebook.py

One notebook for the three pretrained models (XLM-R, RoBERTa, mBERT): the MODEL
setting picks one. Each model reads its own tokenizer's pieces, so the notebook
builds that tokenizer's labels from our word labels and the original text
(about 5 minutes on CPU). It then runs exactly the command you would run
locally, and writes the same result JSON for scripts/compare_models.py. It
embeds the live src/ modules and scripts/train_anticipation.py verbatim, so
re-run this script after changing them.
"""
from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "kaggle", "train_pretrained.ipynb")
MODULES = ["__init__", "routing", "tokenizer", "dict_tagger", "unicode_tagger", "dispatch",
           "segments", "anticipation", "subword", "token_labels", "metrics", "evaluation",
           "streaming", "causal_model", "xlmr_model", "models"]
SCRIPTS = ["train_anticipation"]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.strip("\n")}


INTRO = """
# Causal pretrained encoders: anticipate code switches and their duration

Fine-tunes a **pretrained encoder with a causal attention mask** on sub-word labels. At every piece, the model sees only the pieces so far and predicts:
- **`y_sw`:** does the next piece switch language?
- **`y_dur`:** if so, is the new run Small (1–2), Medium (3–6) or Large (7+)?

| `MODEL` | Checkpoint | Pretrained on | Tokenizer |
|---|---|---|---|
| `"xlmr"` | xlm-roberta-base (278 M) | 100 languages | SentencePiece |
| `"roberta"` | roberta-base (125 M) | English only (the monolingual control) | byte-level BPE |
| `"mbert"` | bert-base-multilingual-cased (178 M) | 104 languages | WordPiece |

The protocol is the same as every other model in the project:

| Split | Share | Used for |
|---|---|---|
| train | 80% | fitting (`row_id % 10` in 0–7) |
| val | 10% | tuning the switch threshold (8) |
| test | 10% | the final numbers, scored once (9) |

The report (`src/evaluation.py`) covers switch P/R/F1 overall, per language pair, macro-F1 across pairs, per direction, and 3-class duration. It also includes **shared positions**, the last piece of each non-CJK word. Those are the same words for every tokenizer, so that score compares these models fairly. A causality check runs before training and must report `PREFIX-ONLY OK`.

## How to run

1. **Settings:** set the Accelerator to **GPU T4 ×2** (one GPU is used) and turn Internet **on**.
2. **Secret:** Add-ons → Secrets → `HF_TOKEN`, a read token with access to both `CodeSwitchGroup4/switchlingua-tagged-final` (our labels; a fine-grained token needs the `CodeSwitchGroup4` org added under its permissions) and the gated `Shelton1013/SwitchLingua_text` (the original text).
3. **Settings cell:** set `MODEL`, then **Save Version → Save & Run All**, so the run continues if you close the browser.
   - Use `SMOKE = True` first for a quick check.
   - A full run of 12,000 steps took about 72 minutes on a T4 for XLM-R; RoBERTa and mBERT are smaller.
4. **Collect the results:** from the version's Output tab, download `data/anticipation_<TAG>.json` (and `_zeroshot.json`, if run), plus `data/preds_<TAG>.npz`, into the repo's `data/`. `TAG` is the model name plus the tuning flags that are on (e.g. `xlmr_durw_prev`). Then run `./.venv/bin/python scripts/compare_models.py`. The saved predictions let thresholds be tuned locally with no GPU.
"""

CONFIG = """
MODEL = "xlmr"          # "xlmr" | "roberta" | "mbert"
SMOKE = False           # True: 60 steps + a small evaluation, to check everything runs
RUN_FULL = True         # train and score on all 15 pairs
ZERO_SHOT = False       # train on 10 pairs and score the 4 held-out languages
STEPS = 12_000          # ~1 epoch of the train split at batch 16
BATCH = 16
WINDOW, STRIDE = 384, 288   # these encoders read at most 512 positions
LR, HEAD_LR = 2e-5, 1e-3    # pretrained body vs. the two new heads
EVAL_BATCHES = 0        # 0 = score the whole val and test splits
# Tuning flags (all off = the baseline setup). See README "How models are evaluated".
SEED = 0                # reproducible initialisation and data order
DUR_WEIGHTS = "none"    # "balanced": class-weighted duration loss (weights from the train split)
DUR_HEAD = "softmax"    # "ordinal": cumulative cuts P(>Small), P(>Medium) (duration is ordered)
TURN_SEP = False        # separator piece between speaker turns
PREV_LANG = False       # input: the language of the last finished word
PREDICT_AT = "piece"    # "word_end": train and score only at the last piece of each word
CONTEXT = None          # 1: read the current piece only, no prefix (needs the three flags above off)
PAIRS = ""              # comma-separated first languages to train and score on ("" = all 15),
                        # e.g. "Cantonese,Chinese,Japanese,Korean,Russian,Arabic"
SAVE_CHECKPOINT = True  # 0.5-1.1 GB in /kaggle/working
LABELS_REPO = "CodeSwitchGroup4/switchlingua-tagged-final"
"""

SETUP = """
import os, subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "regex", "wordfreq"], check=True)
import torch
print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available(),
      [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
if not torch.cuda.is_available():
    print("WARNING: no GPU - enable Settings > Accelerator > GPU T4 x2")

WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.abspath("xlmr_out")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
for d in ("src", "scripts", "data", "models", "logs"):
    os.makedirs(os.path.join(WORK, d), exist_ok=True)
os.chdir(WORK)
sys.path.insert(0, WORK)

TOKEN = os.environ.get("HF_TOKEN")
if not TOKEN:
    try:
        from kaggle_secrets import UserSecretsClient
        TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass
print("working dir:", WORK, "| HF token:", "found" if TOKEN else "MISSING - add the HF_TOKEN secret")
"""

DATA = """
# Build this tokenizer's sub-word labels: our word labels + the original text.
import glob
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoTokenizer
from src.segments import parse_utterances
from src.subword import build_frame, write_parquet
from src.xlmr_model import BACKBONES

PATH = {"xlmr": "data/subword.parquet", "roberta": "data/subword_roberta.parquet",
        "mbert": "data/subword_mbert.parquet"}[MODEL]
tagged = pd.read_parquet(hf_hub_download(LABELS_REPO, "tagged.parquet", repo_type="dataset", token=TOKEN),
                         columns=["row_id", "first_language", "second_language", "quality",
                                  "tokens", "token_lang", "is_entity"])
snap = snapshot_download("Shelton1013/SwitchLingua_text", repo_type="dataset",
                         allow_patterns=["*.csv"], token=TOKEN)
files = sorted(glob.glob(os.path.join(snap, "*.csv")), key=os.path.basename)   # same order as locally
cells = pd.concat([pd.read_csv(f, usecols=["data_generation_result"]) for f in files],
                  ignore_index=True)["data_generation_result"]
assert len(cells) == 234_172, f"unexpected corpus size {len(cells):,}"
texts = {int(r): parse_utterances(cells.iloc[int(r)]) for r in tagged.row_id}
tok = AutoTokenizer.from_pretrained(BACKBONES[MODEL])
frame, stats = build_frame(tagged, texts, tok)
frame = frame.iloc[np.random.default_rng(0).permutation(len(frame))]   # no curriculum by language
write_parquet(frame, PATH)
print(f"{PATH}: {stats['rows']:,} conversations, {stats['pieces']:,} pieces, "
      f"{stats['failed']} alignment failures, {stats['mixed_pieces']} mixed pieces")
del tagged, cells, texts, frame
"""

TRAIN = """
steps, evalb = (60, 20) if SMOKE else (STEPS, EVAL_BATCHES)
common = (f"python scripts/train_anticipation.py --model {MODEL} --unit subword "
          f"--steps {steps} --batch-size {BATCH} --window {WINDOW} --stride {STRIDE} "
          f"--lr {LR} --head-lr {HEAD_LR} --eval-batches {evalb} --workers 2 "
          f"--seed {SEED} --dur-class-weights {DUR_WEIGHTS} --dur-head {DUR_HEAD} --predict-at {PREDICT_AT}"
          + (" --turn-sep" if TURN_SEP else "") + (" --prev-lang" if PREV_LANG else "")
          + (f" --context {CONTEXT}" if CONTEXT else ""))
TAG = MODEL + "".join(s for s, on in (("_durw", DUR_WEIGHTS != "none"), ("_ord", DUR_HEAD == "ordinal"),
                                      ("_sep", TURN_SEP),
                                      ("_prev", PREV_LANG), ("_wend", PREDICT_AT == "word_end"),
                                      (f"_ctx{CONTEXT}", bool(CONTEXT)),
                                      (f"_{len(PAIRS.split(','))}pairs", bool(PAIRS))) if on)
if RUN_FULL:
    cmd = (f"{common}" + (f" --train-langs {PAIRS}" if PAIRS else "") + f" --out data/anticipation_{TAG}.json --ckpt models/anticipation_{TAG}.pt "
           f"--save-preds data/preds_{TAG}.npz")
    print(cmd)
    get_ipython().system(f"{cmd} 2>&1 | tee logs/train_{TAG}.log")
else:
    print("RUN_FULL is off")
"""

ZS = """
if ZERO_SHOT:
    langs = "Cantonese,Arabic,Philippines,Chinese,Vietnamese,Malay,Japanese,Russian,Spanish,French"
    cmd = (f"{common} --train-langs {langs} "
           f"--out data/anticipation_{TAG}_zeroshot.json --ckpt models/anticipation_{TAG}_zeroshot.pt")
    print(cmd)
    get_ipython().system(f"{cmd} 2>&1 | tee logs/train_{TAG}_zeroshot.log")
else:
    print("ZERO_SHOT is off")
"""

RESULTS = """
import json
for f in (f"data/anticipation_{TAG}.json", f"data/anticipation_{TAG}_zeroshot.json"):
    if not os.path.exists(f):
        continue
    r = json.load(open(f))
    o, d = r["overall"], r["duration"]
    sh = r.get("shared", {}).get("overall", {}).get("f1", float("nan"))
    print(f"== {f}")
    print(f"  switch F1 {o['f1']:.3f} (P {o['precision']:.3f}, R {o['recall']:.3f}) | "
          f"macro-F1 {r['macro_f1']:.3f} | shared F1 {sh:.3f}")
    print("  by direction:", {k: round(v["f1"], 3) for k, v in r["per_direction"].items()})
    print(f"  duration: 3-class accuracy {d['accuracy']:.3f} | macro-F1 {d['macro_f1']:.3f}")
    if "zero_shot" in r:
        z = r["zero_shot"]
        print(f"  ZERO-SHOT held-out F1 {z['overall']['f1']:.3f} | macro {z['macro_f1']:.3f} | "
              + ", ".join(f"{k} {v['f1']:.3f}" for k, v in z["per_pair"].items()))
    display(pd.DataFrame({k: {"F1": v["f1"], "P": v["precision"], "R": v["recall"], "n": v["n"]}
                          for k, v in r["per_pair"].items()}).T.sort_values("F1", ascending=False).round(3))
    if "duration_tuned" in r:
        t = r["duration_tuned"]["report"]["duration"]
        print(f"  duration, offsets tuned on val {r['duration_tuned']['offsets']}: accuracy "
              f"{t['accuracy']:.3f} | macro-F1 {t['macro_f1']:.3f} | F1 small/medium/large "
              + " / ".join(f"{f:.3f}" for f in t["per_class_f1"]))
    if "duration_end_to_end" in r:
        e = r["duration_end_to_end"]
        print(f"  duration end to end (switch predicted AND class right): accuracy "
              f"{e['accuracy']:.3f} | macro-F1 {e['macro_f1']:.3f}")
    if "duration_per_pair" in r:
        display(pd.DataFrame({k: {"n": v["n"], "acc": v["accuracy"], "macro-F1": v["macro_f1"],
                                  "F1 small": v["per_class_f1"][0], "F1 medium": v["per_class_f1"][1],
                                  "F1 large": v["per_class_f1"][2],
                                  "e2e acc": v.get("end_to_end", {}).get("accuracy")}
                              for k, v in r["duration_per_pair"].items()}).T
                .sort_values("macro-F1", ascending=False).round(3))
"""

CLEANUP = """
if not SAVE_CHECKPOINT:
    for f in (f"models/anticipation_{TAG}.pt", f"models/anticipation_{TAG}_zeroshot.pt"):
        if os.path.exists(f):
            os.remove(f)
if os.path.exists(PATH):                      # derived from the private labels: don't publish it
    os.remove(PATH)
print(sorted(os.listdir("data")), sorted(os.listdir("models")))
"""


def main() -> int:
    cells = [md(INTRO), md("## Config"), code(CONFIG), md("## Setup"), code(SETUP),
             md("## Project code\nThe live `src/` modules and the training script, embedded verbatim.")]
    for name in MODULES:
        body = open(os.path.join(ROOT, "src", f"{name}.py")).read() or "# package marker\n"
        cells.append(code(f"%%writefile src/{name}.py\n{body}"))
    for name in SCRIPTS:
        body = open(os.path.join(ROOT, "scripts", f"{name}.py")).read()
        cells.append(code(f"%%writefile scripts/{name}.py\n{body}"))
    cells += [md("## Build the labels for this tokenizer"), code(DATA),
              md("## Train and evaluate (all 15 pairs)"), code(TRAIN),
              md("## Optional: zero-shot run (4 pairs held out)"), code(ZS),
              md("## Results"), code(RESULTS),
              md("## Clean up\nRemoves the derived training file (it comes from private labels). Result JSONs stay in `data/`."),
              code(CLEANUP)]
    nb = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}},
        "nbformat": 4, "nbformat_minor": 5}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(nb, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(f"wrote {OUT} ({len(cells)} cells, {len(MODULES)} modules, {len(SCRIPTS)} script)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
