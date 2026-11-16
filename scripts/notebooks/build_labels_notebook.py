#!/usr/bin/env python
"""Write kaggle/load_labels.ipynb: download our labels and pick the version you need.

    ./.venv/bin/python scripts/notebooks/build_labels_notebook.py

The notebook downloads the final word-level labels from the private Hugging
Face dataset and turns them into per-token switch and duration labels, at the
unit and names policy chosen in its first cell. It embeds the live ``src/``
label modules verbatim, so re-run this script after changing them. No GPU is
needed: nothing here runs a model.
"""
from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "kaggle", "load_labels.ipynb")

# Everything the label path imports (no torch, no models).
MODULES = ["__init__", "routing", "tokenizer", "dict_tagger", "unicode_tagger",
           "dispatch", "segments", "anticipation", "subword", "token_labels"]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.strip("\n")}


INTRO = """
# SwitchLingua — download our labels and choose a version

Every token of the SwitchLingua corpus (234,172 conversations, 15 language pairs) already has a language label. With the default settings, the result is identical to `token_labels_word.parquet` / `token_labels_subword.parquet` in the same Hugging Face dataset; use this notebook to pick a different names policy or a subset. This notebook downloads those labels from our **private** Hugging Face dataset and builds the prediction targets in the form you choose:

| Label | Meaning |
|---|---|
| `lang` | `L1` (first language), `L2` (second language), or `-` (punctuation, digits: skipped) |
| `y_sw` | `1` if the **next** token switches language, `0` if not, `-1` where undefined |
| `y_len` | at a switch, how many tokens the new run lasts (integer; 0 elsewhere) |
| `y_dur` | the class of `y_len`: `small` (1–2), `medium` (3–6), `large` (7+); empty elsewhere |
| `censored` | the run is still open when the text ends, so `y_len` is a lower bound |
| `dur_known` | the duration class is certain: the run is closed, or it is cut off but already 7+ |

Which model produced the underlying word labels, and how accurate each is, is in `docs/labelling_models.md` in the GitHub repo.

## How to run

1. **Access.** Your Hugging Face account needs access to `CodeSwitchGroup4/switchlingua-tagged-final` (ask Cristae to add you to the `CodeSwitchGroup4` org; a fine-grained token also needs that org under its permissions) and, for sub-word labels only, to the gated [`Shelton1013/SwitchLingua_text`](https://huggingface.co/datasets/Shelton1013/SwitchLingua_text).
2. **Token.** On Kaggle, add a secret named `HF_TOKEN` (Add-ons → Secrets) holding a read token. Locally, put `HF_TOKEN=...` in `.env`.
3. **Settings.** Turn Internet on. A CPU is enough.
4. **Choose** in the Config cell below, then *Run All*. The output files appear in `/kaggle/working` (Output tab).

Word labels take about a minute; sub-word labels about 5 (they tokenize the original text with XLM-R).
"""

CONFIG = """
# ---- Choose your labels -----------------------------------------------------
LEVEL = "word"           # "word":    one label per word (Chinese/Japanese/Korean: per character)
                         # "subword": one label per XLM-R piece (what an XLM-R model reads)
NAMES = "script_words"   # how names count:
                         #   "script_words" (default) a name takes the language its context implies
                         #   "transparent"            names are skipped, like punctuation
                         #   "script"                 every Latin-script name is English
PAIRS = None             # None = all 15, or e.g. [("Spanish", "English"), ("German", "French")]
SPLITS = None            # None = all, or e.g. ["train"]   (row_id % 10: 0-7 train, 8 val, 9 test)
LIMIT = None             # None = every row, or e.g. 200 rows per pair for a quick look
EXPORT = "parquet"       # "parquet", "csv" or "both"

LABELS_REPO = "CodeSwitchGroup4/switchlingua-tagged-final"
"""

SETUP = """
import os, subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "regex", "wordfreq"], check=True)

WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.abspath("labels_out")
os.makedirs(os.path.join(WORK, "src"), exist_ok=True)
os.chdir(WORK)
sys.path.insert(0, WORK)

TOKEN = os.environ.get("HF_TOKEN")
if not TOKEN:
    try:
        from kaggle_secrets import UserSecretsClient
        TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass
if not TOKEN:
    try:
        from dotenv import dotenv_values
        for env in (".env", "../.env"):
            TOKEN = TOKEN or dotenv_values(env).get("HF_TOKEN")
    except Exception:
        pass
assert LEVEL in ("word", "subword"), LEVEL
assert NAMES in ("script_words", "transparent", "script"), NAMES
assert EXPORT in ("parquet", "csv", "both"), EXPORT
print("working dir:", WORK, "| HF token:", "found" if TOKEN else "MISSING - add the HF_TOKEN secret")
"""

DOWNLOAD = """
import pandas as pd
from huggingface_hub import hf_hub_download

path = hf_hub_download(LABELS_REPO, "tagged.parquet", repo_type="dataset", token=TOKEN)
cols = ["row_id", "first_language", "second_language", "quality", "tokens", "token_lang", "is_entity"]
df = pd.read_parquet(path, columns=cols)
print(f"downloaded {len(df):,} rows")

split_of = lambda rid: "train" if rid % 10 < 8 else "val" if rid % 10 == 8 else "test"
df["split"] = df.row_id.map(split_of)
if PAIRS:
    keep = {tuple(p) for p in PAIRS}
    df = df[[(a, b) in keep for a, b in zip(df.first_language, df.second_language)]]
if SPLITS:
    df = df[df.split.isin(SPLITS)]
if LIMIT:
    df = df.groupby(["first_language", "second_language"], sort=False).head(LIMIT)
df = df.reset_index(drop=True)
print(f"selected {len(df):,} rows")
print(df.groupby(["first_language", "second_language"]).size().to_string())
"""

LABEL = """
import numpy as np
import src.anticipation as A
from src.token_labels import subword_labels, word_labels

A.NAMES = NAMES                                   # the names policy every label call uses
if LEVEL == "word":
    UNIT = "tokens"
    labels = word_labels(df)
else:
    UNIT = "pieces"
    import glob
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from src.segments import parse_utterances
    from src.subword import build_frame

    # The original text: same file order as the local pipeline, so row_id matches.
    snap = snapshot_download("Shelton1013/SwitchLingua_text", repo_type="dataset",
                             allow_patterns=["*.csv"], token=TOKEN)
    files = sorted(glob.glob(os.path.join(snap, "*.csv")), key=os.path.basename)
    corpus = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    assert len(corpus) == 234_172, f"unexpected corpus size {len(corpus):,}"
    texts = {int(i): parse_utterances(corpus.data_generation_result.iloc[int(i)]) for i in df.row_id}
    tok = AutoTokenizer.from_pretrained("xlm-roberta-base")
    sw, stats = build_frame(df, texts, tok)
    print(f"{stats['rows']:,} rows, {stats['pieces']:,} pieces, "
          f"{stats['failed']} alignment failures, {stats['mixed_pieces']} mixed pieces")
    labels = subword_labels(sw)
print(f"labelled {len(labels):,} rows at the {LEVEL} level, names policy {NAMES!r}")
"""

SUMMARY = """
def pair_summary(g):
    sw = np.concatenate([np.asarray(x) for x in g.y_sw])
    known = np.concatenate([np.asarray(x) for x in g.dur_known])
    cls = np.concatenate([np.asarray(x, dtype=object) for x in g.y_dur])[known]
    n = max(len(cls), 1)
    return pd.Series({"rows": len(g), "scored": int((sw >= 0).sum()),
                      "switch rate": f"{(sw == 1).sum() / max((sw >= 0).sum(), 1):.1%}",
                      "small": f"{(cls == 'small').sum() / n:.0%}",
                      "medium": f"{(cls == 'medium').sum() / n:.0%}",
                      "large": f"{(cls == 'large').sum() / n:.0%}"})

print(labels.groupby(["first_language", "second_language"]).apply(pair_summary).to_string())
"""

EXAMPLE = """
# One row, token by token. Change `i` to look at others.
i = 0
r = labels.iloc[i]
print(f"row {r.row_id} | {r.first_language} -> {r.second_language} | {r.split}")
pd.DataFrame({k: r[k] for k in [UNIT, "lang", "y_sw", "y_len", "y_dur", "censored"]}).head(40)
"""

SAVE = """
name = f"switchlingua_labels_{LEVEL}_{NAMES}"
out = []
if EXPORT in ("parquet", "both"):
    labels.to_parquet(os.path.join(WORK, name + ".parquet"), index=False)
    out.append(name + ".parquet")
if EXPORT in ("csv", "both"):
    # One line per conversation; each list column is space-separated, token-aligned.
    flat = labels.copy()
    for c in [UNIT, "lang", "y_sw", "y_len", "y_dur", "censored", "dur_known"]:
        flat[c] = [" ".join(str(int(v)) if isinstance(v, (bool, np.bool_)) else (str(v) or "_") for v in xs)
                   for xs in flat[c]]
    flat = flat.drop(columns=[c for c in ("piece_ids", "is_entity") if c in flat])
    flat.to_csv(os.path.join(WORK, name + ".csv"), index=False)
    out.append(name + ".csv")
for f in out:
    print(f"wrote {os.path.join(WORK, f)}  ({os.path.getsize(os.path.join(WORK, f)) / 1e6:.0f} MB)")
print("In the CSV, an empty y_dur is written as '_' so the columns stay aligned.")
"""


def main() -> int:
    cells = [md(INTRO), md("## Config"), code(CONFIG), md("## Setup"), code(SETUP),
             md("## Label code\nThe live `src/` label modules, embedded verbatim.")]
    for name in MODULES:
        with open(os.path.join(ROOT, "src", f"{name}.py")) as fh:
            body = fh.read() or "# package marker\n"   # %%writefile rejects an empty body
        cells.append(code(f"%%writefile src/{name}.py\n{body}"))
    cells += [md("## Download the word labels"), code(DOWNLOAD),
              md("## Build the labels you chose"), code(LABEL),
              md("## Per-pair summary"), code(SUMMARY),
              md("## Look at a row"), code(EXAMPLE),
              md("## Save"), code(SAVE)]
    nb = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "kaggle": {"accelerator": "none", "isInternetEnabled": True, "isGpuEnabled": False}},
        "nbformat": 4, "nbformat_minor": 5}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)
    print(f"wrote {OUT} ({len(cells)} cells, {len(MODULES)} embedded modules)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
