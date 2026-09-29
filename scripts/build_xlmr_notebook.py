#!/usr/bin/env python
"""Write kaggle/train_xlmr.ipynb: fine-tune and evaluate causal XLM-R on a Kaggle GPU.

    ./.venv/bin/python scripts/build_xlmr_notebook.py

XLM-R-base trains at about 1.2 s/step on an M3 (7.7 h per epoch), so it runs on
Kaggle. The notebook embeds the live src/ modules and scripts/train_anticipation.py
verbatim, so re-run this script after changing them. It runs exactly the command
you would run locally, and writes the same result JSON for scripts/compare_models.py.
"""
from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "kaggle", "train_xlmr.ipynb")
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
# Causal XLM-R: anticipate code switches and their duration

Fine-tunes **XLM-R-base with a causal attention mask** on the sub-word labels. At every XLM-R piece, the model sees only the pieces so far and predicts:
- **`y_sw`:** does the next piece switch language?
- **`y_dur`:** if so, is the new run Small (1–2), Medium (3–6) or Large (7+)?

The protocol is the same as every other model in the project:

| Split | Share | Used for |
|---|---|---|
| train | 80% | fitting (`row_id % 10` in 0–7) |
| val | 10% | tuning the switch threshold (8) |
| test | 10% | the final numbers, scored once (9) |

The test report is the shared one (`src/evaluation.py`): switch P/R/F1 overall, **per language pair**, macro-F1 across pairs, **per direction** (into L2 / back into L1), and 3-class duration at true switch points. A causality check runs before training and must report `PREFIX-ONLY OK`.

## How to run

1. **Settings:** set the Accelerator to **GPU T4 ×2** (one GPU is used) and turn Internet **on**.
2. **Secret:** Add-ons → Secrets → `HF_TOKEN`, a read token whose account has access to `cristae/switchlingua-tagged-final`.
3. **Run All.**
   - Set `SMOKE = True` first for a 5-minute end-to-end check.
   - A full run of 12,000 steps (about one epoch at batch 16) should take roughly 1–2 hours on a T4. This hasn't been measured yet, and the step logs show the real speed.
4. **Collect the results.** From the Output tab, download `data/anticipation_xlmr.json` (and `data/anticipation_xlmr_zeroshot.json`, if run) into the repo's `data/`. Then run `./.venv/bin/python scripts/compare_models.py`.
"""

CONFIG = """
SMOKE = False           # True: 60 steps + a small evaluation, to check everything runs
STEPS = 12_000          # ~1 epoch of the train split at batch 16
BATCH = 16
WINDOW, STRIDE = 384, 288   # XLM-R reads at most 512 positions
LR, HEAD_LR = 2e-5, 1e-3    # pretrained body vs. the two new heads
EVAL_BATCHES = 0        # 0 = score the whole val and test splits
ZERO_SHOT = False       # also train on 10 pairs and score the 4 held out
SAVE_CHECKPOINT = True  # ~1.1 GB in /kaggle/working
LABELS_REPO = "cristae/switchlingua-tagged-final"
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
# The shared sub-word labels -> the training format (shuffled, row_id keeps the split).
import pandas as pd
from huggingface_hub import hf_hub_download
from src.token_labels import to_subword_frame

path = hf_hub_download(LABELS_REPO, "token_labels_subword.parquet", repo_type="dataset", token=TOKEN)
labels = pd.read_parquet(path, columns=["row_id", "first_language", "second_language", "quality",
                                        "piece_ids", "lang"])
frame = to_subword_frame(labels, seed=0)
frame.to_parquet("data/subword.parquet", index=False, row_group_size=8192)
print(f"data/subword.parquet: {len(frame):,} conversations, "
      f"{sum(map(len, frame.piece_ids)):,} pieces")
del labels, frame
"""

TRAIN = """
steps, evalb = (60, 20) if SMOKE else (STEPS, EVAL_BATCHES)
cmd = (f"python scripts/train_anticipation.py --model xlmr --unit subword "
       f"--steps {steps} --batch-size {BATCH} --window {WINDOW} --stride {STRIDE} "
       f"--lr {LR} --head-lr {HEAD_LR} --eval-batches {evalb} --workers 2 "
       f"--out data/anticipation_xlmr.json --ckpt models/anticipation_xlmr.pt")
print(cmd)
!{cmd} 2>&1 | tee logs/train_xlmr.log
"""

ZS = """
if ZERO_SHOT:
    langs = "Cantonese,Arabic,Philippines,Chinese,Vietnamese,Malay,Japanese,Russian,Spanish,French"
    cmd = (f"python scripts/train_anticipation.py --model xlmr --unit subword "
           f"--steps {60 if SMOKE else STEPS} --batch-size {BATCH} --window {WINDOW} --stride {STRIDE} "
           f"--lr {LR} --head-lr {HEAD_LR} --eval-batches {20 if SMOKE else EVAL_BATCHES} --workers 2 "
           f"--train-langs {langs} "
           f"--out data/anticipation_xlmr_zeroshot.json --ckpt models/anticipation_xlmr_zeroshot.pt")
    print(cmd)
    get_ipython().system(f"{cmd} 2>&1 | tee logs/train_xlmr_zeroshot.log")
else:
    print("ZERO_SHOT is off")
"""

RESULTS = """
import json
r = json.load(open("data/anticipation_xlmr.json"))
o = r["overall"]
print(f"switch F1 {o['f1']:.3f} (P {o['precision']:.3f}, R {o['recall']:.3f}) | "
      f"macro-F1 over pairs {r['macro_f1']:.3f}")
print("by direction:", {k: round(v["f1"], 3) for k, v in r["per_direction"].items()})
d = r["duration"]
print(f"duration: 3-class accuracy {d['accuracy']:.3f} | macro-F1 {d['macro_f1']:.3f}")
pd.DataFrame({k: {"F1": v["f1"], "P": v["precision"], "R": v["recall"], "n": v["n"]}
              for k, v in r["per_pair"].items()}).T.sort_values("F1", ascending=False).round(3)
"""

CLEANUP = """
if not SAVE_CHECKPOINT:
    for f in ("models/anticipation_xlmr.pt", "models/anticipation_xlmr_zeroshot.pt"):
        if os.path.exists(f):
            os.remove(f)
if os.path.exists("data/subword.parquet"):  # derived from the private labels: don't publish it
    os.remove("data/subword.parquet")
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
    cells += [md("## Download the labels"), code(DATA),
              md("## Train and evaluate XLM-R (all 15 pairs)"), code(TRAIN),
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
