#!/usr/bin/env python
"""Build ``kaggle/label_switchlingua.ipynb`` - a self-contained labelling notebook.

The notebook embeds the current ``src/`` modules verbatim (one ``%%writefile``
cell each), so there is nothing to upload but the ``.ipynb`` itself, and it
can never drift from the local pipeline: re-run this script after changing
``src/`` and re-upload.

    ./.venv/bin/python scripts/build_kaggle_notebook.py

What the notebook does on a Kaggle GPU (T4 x2 recommended):
  1. loads the 15 SwitchLingua CSVs in the same order as ``run_tagging.py``,
     so ``row_id`` and the train/val/test split match the local parquet;
  2. works through the corpus one language pair at a time (big pairs in
     20k-row parts), sharding the model passes across the GPUs in fp16:
     AnE-LID on Latin->English rows, plus windows around Latin-script names in
     script-tagged rows; name positions come from the v1 labels (or AnE-NER);
  3. tags each part exactly as ``run_tagging.py`` does and uploads its CSV and
     parquet to a private Hugging Face dataset before starting the next, so a
     crash loses at most one part and a re-run skips everything uploaded;
  4. once every part is up, merges them into ``switchlingua_tagged_rows.csv``
     (one line per row, via ``persist.export_rows_csv``) and ``tagged.parquet``
     and uploads those too.
"""
from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "kaggle", "label_switchlingua.ipynb")

# Everything the tagging path imports. model_tagger (fastText) and the
# anticipation modules are not on it.
MODULES = ["__init__", "routing", "tokenizer", "unicode_tagger", "dict_tagger",
           "ane_tagger", "bert_tagger", "dispatch", "segments", "persist"]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.strip("\n")}


INTRO = """
# SwitchLingua — per-token language labels

This notebook labels every token of `Shelton1013/SwitchLingua_text` (234,172 rows) with its language:

| Rows | Tagger |
|---|---|
| Cantonese, Chinese, Japanese, Korean, Russian, Arabic, and Devanagari Hindi | Unicode script (exact) |
| Latin-script first language with English as L2, including romanized Hindi | AnE-LID (XLM-R-large) |
| German → French | wordfreq frequency comparison |
| Named entities, in every row | AnE-NER → a separate `entities` flag. Names keep a **language** label, and count as a switch only with visible evidence. A Latin-script name inside a non-Latin row is English. In Latin-script rows, the name's words decide when they're clearly one language's ("Severance Season 2" → English); otherwise it takes its clause's language ("Leo Messi" inside Spanish → Spanish). In non-Latin rows, Latin-script names get AnE-LID's label instead of "English by script". |

## How to run
1. **Settings:** set **Accelerator** to *GPU T4 x2* and turn **Internet** on.
2. **Token:** add a Kaggle secret named `HF_TOKEN` (*Add-ons → Secrets*) holding a Hugging Face token with **write** access, and accept the dataset terms at huggingface.co/datasets/Shelton1013/SwitchLingua_text. The token reads the corpus and uploads the results.
3. **Run it:** click **Save Version → Save & Run All (Commit)**. It runs with the browser closed.
4. **Watch progress** on Hugging Face at `huggingface.co/datasets/<you>/switchlingua-tagged-v2`, which is **private**. A pair's files appear there as soon as it finishes.
5. **If the run dies:** just commit again. The notebook checks the HF repo and skips every part already uploaded, so at most one part (≤ 20k rows) is redone.
6. **Smoke test first (optional):** set `LIMIT = 50` and `HF_REPO = "<you>/switchlingua-tagged-test"` so the test doesn't mix with the real output.

By default, name positions are read from the v1 labels (`ENTITY_SOURCE`), whose tokens are identical, so AnE-NER doesn't have to run again. Set `ENTITY_SOURCE = None` to recompute them.

## What ends up on Hugging Face
- **`switchlingua_tagged_rows.csv`**, uploaded last, once every pair is done. It has one line per row:
  - `tokens` and `labels` hold aligned, space-joined values, so `zip(tokens.split(" "), labels.split(" "))` pairs each token with its label.
  - The label codes are `L1`, `L2` and `O` (punctuation or digits). Names have a language code like every other word, resolved with the default (script_words) rule.
  - `entities` is an aligned `0`/`1` string, where `1` means the token is part of a named entity.
  - `l2_runs` holds the length in words of each second-language run, i.e. how long each switch lasts, under the default rule. Punctuation is skipped.
  - `l2_runs_names_transparent` skips names entirely, and `l2_runs_names_script` makes every Latin-script name English. These are the two ablations.
  - `split` is the row-level train/val/test assignment (`row_id % 10`).
- **`tagged.parquet`:** the same data in the local pipeline's format. Put it in `data/` for `baseline_anticipation.py`, `train_anticipation.py` and the rest.
- **`rows/<L1>_<L2>/part-NNN.csv` and `parquet/<L1>_<L2>/part-NNN.parquet`:** the per-pair pieces, uploaded as each finishes.

Download everything with `huggingface-cli download <you>/switchlingua-tagged --repo-type dataset --local-dir data/hf`.

*Generated by `scripts/build_kaggle_notebook.py`; do not edit by hand.*
"""

CONFIG = """
LIMIT = None          # e.g. 50 -> first 50 rows of each language pair, for a smoke test
BATCH = 64            # rows per GPU batch; lower to 32 if CUDA runs out of memory
HF_REPO = None        # None -> "<your-username>/switchlingua-tagged-v2" (private)
ENTITY_SOURCE = "cristae/switchlingua-tagged"   # v1 labels: name positions = its NE tokens.
                                                # None -> run AnE-NER again (slow)
WINDOW = 16           # context tokens either side of a Latin name in script-tagged rows
PART_ROWS = 20_000    # rows per upload; big pairs (Cantonese: 119k) go up in parts
UPLOAD = True         # False -> write to ./hf_mirror instead (local testing)
"""

SETUP = """
import os, subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "regex", "wordfreq", "pyarrow"],
               check=True)
import torch
print("torch", torch.__version__, "| CUDA devices:", torch.cuda.device_count(),
      [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
if not torch.cuda.is_available():
    print("WARNING: no GPU - enable Settings > Accelerator > GPU T4 x2, or this takes ~10 h.")

WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.abspath("kaggle_out")
os.makedirs(os.path.join(WORK, "src"), exist_ok=True)
os.chdir(WORK)
sys.path.insert(0, WORK)
print("working dir:", WORK)
"""

LOAD = """
import glob
import pandas as pd

NAMES = ["Arabic_eng.csv", "Can_Eng.csv", "Chinese_eng.csv", "French_eng.csv", "German_Eng.csv",
         "German_French.csv", "Hindi_eng.csv", "Italian_eng.csv", "Japanese_eng.csv",
         "Korean_eng.csv", "Malay_eng.csv", "Philippines_eng.csv", "Russian_eng.csv",
         "Spanish_eng.csv", "Vietnamese_eng.csv"]

TOKEN = os.environ.get("HF_TOKEN")
if not TOKEN:
    try:
        from kaggle_secrets import UserSecretsClient
        TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        TOKEN = None
print("HF token:", "found" if TOKEN else "MISSING - add the HF_TOKEN secret")

def find_csvs():
    # 1) an attached Kaggle dataset
    found = {os.path.basename(p): p for p in glob.glob("/kaggle/input/**/*.csv", recursive=True)}
    if all(n in found for n in NAMES):
        print("using attached Kaggle dataset")
        return [found[n] for n in NAMES]
    # 2) the Hugging Face Hub (gated: needs the HF_TOKEN secret)
    from huggingface_hub import snapshot_download
    snap = snapshot_download("Shelton1013/SwitchLingua_text", repo_type="dataset",
                             allow_patterns=["*.csv"], token=TOKEN)
    return [os.path.join(snap, n) for n in NAMES]

# Same order as run_tagging.load_corpus (sorted filenames), so row_id and the
# row_id % 10 split are identical to the local parquet.
files = sorted(find_csvs(), key=os.path.basename)
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
print(f"{len(df):,} rows")
if LIMIT:
    df = df.groupby(["first_language", "second_language"], sort=False).head(LIMIT)
    print(f"LIMIT={LIMIT}: {len(df):,} rows")
print(df.groupby(["first_language", "second_language"]).size().to_string())
"""

STORE = """
# Where finished parts go. Everything already there is skipped, which is what
# makes a re-run resume.
import shutil
if UPLOAD:
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=TOKEN)
    user = api.whoami()["name"]
    REPO = HF_REPO or f"{user}/switchlingua-tagged-v2"
    try:
        api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
    except Exception as exc:
        raise RuntimeError(f"cannot create {REPO} - does HF_TOKEN have WRITE access? {exc}")
    stored = set(api.list_repo_files(REPO, repo_type="dataset"))

    def put(local, path, msg):
        api.upload_file(path_or_fileobj=local, path_in_repo=path, repo_id=REPO,
                        repo_type="dataset", commit_message=msg)
        stored.add(path)

    def get(path):
        return hf_hub_download(REPO, path, repo_type="dataset", token=TOKEN)
    print(f"uploading to https://huggingface.co/datasets/{REPO} (private)")
else:
    REPO = os.path.join(WORK, "hf_mirror")
    os.makedirs(REPO, exist_ok=True)
    stored = {os.path.relpath(os.path.join(d, f), REPO)
              for d, _, fs in os.walk(REPO) for f in fs}

    def put(local, path, msg):
        os.makedirs(os.path.dirname(os.path.join(REPO, path)), exist_ok=True)
        shutil.copy(local, os.path.join(REPO, path))
        stored.add(path)

    def get(path):
        return os.path.join(REPO, path)
    print("UPLOAD=False: mirroring to", REPO)

CARD = '''__CARD__'''
if "README.md" not in stored:
    card = os.path.join(WORK, "README.md")
    with open(card, "w") as fh:
        fh.write(CARD)
    put(card, "README.md", "dataset card")      # also proves the token can write
print(f"{len(stored)} files already stored")
"""

CARD = '''---
license: other
pretty_name: SwitchLingua per-token language labels
---
# SwitchLingua per-token language labels

Per-token labels for [Shelton1013/SwitchLingua_text](https://huggingface.co/datasets/Shelton1013/SwitchLingua_text)
(234,172 rows). Derived from a gated dataset under its terms - keep this repo private.

* `switchlingua_tagged_rows.csv` - one line per row. `tokens` / `labels` / `entities` are
  aligned and space-joined; labels are `L1`, `L2`, `O` (punctuation/digits) - names carry
  their contextual language - and `entities` is 1 for named-entity tokens.
  `l2_runs` = length in words of each second-language run (names count as their language);
  `l2_runs_names_transparent` = the same with names transparent.
  `split` = `row_id % 10` (0-7 train, 8 val, 9 test).
* `tagged.parquet` - the same, in the local pipeline's format.
* `rows/` and `parquet/` - the per-pair parts the files above were merged from.

Taggers: Unicode script (non-Latin L1s, Devanagari Hindi), AnE-LID (Latin L1 with English
L2, romanized Hindi), wordfreq Zipf comparison (German-French), AnE-NER (entities, all rows).
'''

MODELS = """
import threading, time
from src import ane_tagger
from src.persist import (ane_labels, entity_masks, export_rows_csv, row_tokens,
                         save_tagged, tag_dataframe)

ENTS = None
if ENTITY_SOURCE:
    # v1 marked every name NE and has the same tokens, so its NE positions are
    # exactly the AnE-NER masks - no need to run the NER model again.
    from huggingface_hub import hf_hub_download
    v1 = pd.read_parquet(hf_hub_download(ENTITY_SOURCE, "tagged.parquet", repo_type="dataset",
                                         token=TOKEN), columns=["row_id", "tokens", "token_lang"])
    v1 = v1.set_index("row_id")
    ENTS = {int(r): [l == "NE" for l in v1.token_lang[r]] for r in df.index}
    bad = [r for r in df.index[:: max(1, len(df) // 2000)]
           if len(row_tokens(df.data_generation_result[r])) != len(ENTS[int(r)])]
    assert not bad, f"v1 tokens differ from this tokenizer on rows {bad[:5]} - set ENTITY_SOURCE = None"
    print(f"entity masks for {len(ENTS):,} rows from {ENTITY_SOURCE} "
          f"({sum(map(sum, ENTS.values())):,} entity tokens)")
    del v1

n_gpu = torch.cuda.device_count()
DEVICES = [f"cuda:{i}" for i in range(n_gpu)] or [None]
for dev in DEVICES:                       # models on every GPU, so any part can shard
    if ENTS is None:
        ane_tagger.load(ane_tagger.NER_REPO, device=dev)
    ane_tagger.load(ane_tagger.REPO, device=dev)
print("models loaded on", DEVICES)

# Work units: one per pair, big pairs cut into PART_ROWS-row parts; smallest first,
# so finished pairs start appearing on the Hub quickly.
units = []
for (l1, l2), g in sorted(df.groupby(["first_language", "second_language"]), key=lambda x: len(x[1])):
    for k, start in enumerate(range(0, len(g), PART_ROWS)):
        units.append((f"{l1}_{l2}", k, g.iloc[start:start + PART_ROWS]))
todo = [u for u in units if f"parquet/{u[0]}/part-{u[1]:03d}.parquet" not in stored]
print(f"{len(units)} parts, {len(units) - len(todo)} already stored, {len(todo)} to run")
"""

PASSES = """
def label_part(sub):
    # Both model passes, sharded across the GPUs, then tagging.
    shards = [sub.iloc[i::len(DEVICES)] for i in range(len(DEVICES))]
    out, errors = [None] * len(DEVICES), []

    def work(i):
        try:
            dev = DEVICES[i]
            if ENTS is not None:
                e = {int(r): ENTS[int(r)] for r in shards[i].index}
            else:
                e = entity_masks(shards[i], cache=f"data/entities.{i}.parquet", device=dev,
                                 batch=BATCH, progress=False)
            a = ane_labels(shards[i], cache=f"data/ane_lid.{i}.parquet", device=dev,
                           entities=e, window=WINDOW, batch=BATCH, progress=False)
            out[i] = (e, a)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(len(DEVICES))]
    [t.start() for t in threads]
    [t.join() for t in threads]
    if errors:
        raise errors[0]
    ents = {k: v for e, _ in out for k, v in e.items()}
    lid = {k: v for _, a in out for k, v in a.items()}
    return tag_dataframe(sub, progress=False, entities=ents, ane=lid)

os.makedirs("parts", exist_ok=True)
t_all, rows_done, rows_total = time.time(), 0, sum(len(u[2]) for u in todo)
for pair, k, sub in todo:
    t0 = time.time()
    tagged = label_part(sub)
    stem = f"{pair}/part-{k:03d}"
    pq_local = os.path.join("parts", stem.replace("/", "__") + ".parquet")
    csv_local = pq_local.replace(".parquet", ".csv")
    save_tagged(tagged, pq_local)
    export_rows_csv(tagged, csv_local)
    put(csv_local, f"rows/{stem}.csv", f"{pair} part {k}")
    put(pq_local, f"parquet/{stem}.parquet", f"{pair} part {k}")   # written last = part done
    rows_done += len(sub)
    rate = rows_done / (time.time() - t_all)
    print(f"{pair:24} part {k}: {len(sub):>6,} rows in {(time.time() - t0) / 60:5.1f} min | "
          f"{rows_done:,}/{rows_total:,} rows, eta {(rows_total - rows_done) / rate / 60:.0f} min",
          flush=True)
print("all parts stored")
"""

MERGE = """
# Merge every stored part - including ones uploaded by an earlier, crashed run -
# into the two final files, and upload them.
parts = sorted(p for p in stored if p.startswith("parquet/") and p.endswith(".parquet"))
expected = {f"parquet/{u[0]}/part-{u[1]:03d}.parquet" for u in units}
missing = expected - set(parts)
assert not missing, f"{len(missing)} parts missing - re-run the notebook: {sorted(missing)[:5]}"

full = pd.concat([pd.read_parquet(get(p)) for p in sorted(expected)], ignore_index=True)
full = full.sort_values("row_id").reset_index(drop=True)
save_tagged(full, os.path.join(WORK, "tagged.parquet"))
csv_path = export_rows_csv(full, os.path.join(WORK, "switchlingua_tagged_rows.csv"))
print(f"merged {len(full):,} rows; CSV {os.path.getsize(csv_path) / 1e6:.0f} MB")
put(os.path.join(WORK, "tagged.parquet"), "tagged.parquet", "merged parquet")
put(csv_path, "switchlingua_tagged_rows.csv", "merged CSV - all languages")
print("uploaded the merged CSV and parquet")
"""

CHECK = """
import numpy as np
rows = pd.read_csv(csv_path)

# tokens, labels and entity flags are aligned; labels use only L1/L2/O
bad = [r for r, t, l, e in zip(rows.row_id, rows.tokens.fillna(""), rows.labels.fillna(""),
                               rows.entities.fillna(""))
       if not len(t.split(" ")) == len(l.split(" ")) == len(e.split(" "))]
assert not bad, f"{len(bad)} rows with misaligned tokens/labels/entities, e.g. {bad[:5]}"
codes = pd.Series(" ".join(rows.labels.fillna("")).split()).value_counts()
assert set(codes.index) <= {"L1", "L2", "O"}, codes
assert len(rows) == len(df) and sorted(rows.row_id) == sorted(df.index)
print("checks passed: aligned tokens/labels/entities, 3 label codes, every row present\\n")
print("label counts:", codes.to_dict(), "\\n")

def runs(s):
    return [int(x) for x in str(s).split()] if isinstance(s, str) and s else []

summary = []
for (l1, l2), g in rows.groupby(["first_language", "second_language"]):
    r = [x for s in g.l2_runs for x in runs(s)]
    rt = [x for s in g.l2_runs_names_transparent for x in runs(s)]
    ra = [x for s in g.l2_runs_names_script for x in runs(s)]
    lab = " ".join(g.labels.fillna("")).split()
    ent = " ".join(g.entities.fillna("")).split()
    words = sum(1 for x in lab if x != "O")
    summary.append(dict(pair=f"{l1}->{l2}", rows=len(g),
                        name_share=f"{ent.count('1') / max(words, 1):.1%}",
                        switches=len(r), script=len(ra), transparent=len(rt),
                        median_run=float(np.median(r)) if r else 0,
                        median_transparent=float(np.median(rt)) if rt else 0,
                        single_word=f"{np.mean(np.array(r) == 1):.1%}" if r else "-"))
print(pd.DataFrame(summary).to_string(index=False))
"""

EXAMPLE = """
# One example row per language pair; names are marked with *.
for (l1, l2), g in rows.groupby(["first_language", "second_language"]):
    r = g.iloc[0]
    trip = list(zip(r.tokens.split(" "), r.labels.split(" "), r.entities.split(" ")))[:30]
    print(f"{l1} -> {l2}  (row {r.row_id}, l2_runs: {r.l2_runs})")
    print("   ", " ".join(f"{t}/{c}{'*' if e == '1' else ''}" for t, c, e in trip), "\\n")
"""


def main() -> int:
    cells = [md(INTRO), md("## Config"), code(CONFIG), md("## Setup"), code(SETUP),
             md("## Pipeline source\nThe live `src/` modules, embedded verbatim.")]
    for name in MODULES:
        with open(os.path.join(ROOT, "src", f"{name}.py")) as fh:
            body = fh.read() or "# package marker\n"   # %%writefile rejects an empty body
        cells.append(code(f"%%writefile src/{name}.py\n{body}"))
    cells += [md("## Load the corpus"), code(LOAD),
              md("## Result store (private Hugging Face dataset)"),
              code(STORE.replace("__CARD__", CARD)),
              md("## Load the models and plan the parts"), code(MODELS),
              md("## Label and upload, one part at a time\n"
                 "About 1–2 h on T4×2 (not yet measured). A pair's files appear on the Hub as soon as it finishes."),
              code(PASSES),
              md("## Merge all parts into the final CSV and parquet"), code(MERGE),
              md("## Checks and per-pair summary"), code(CHECK),
              md("## Examples"), code(EXAMPLE)]
    nb = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True,
                   "isGpuEnabled": True}},
        "nbformat": 4, "nbformat_minor": 5}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)
    print(f"wrote {OUT} ({len(cells)} cells, {len(MODULES)} embedded modules)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
