# Anticipatory Code-Switch Prediction

Can a model predict that a speaker is **about to switch language** before it happens, using only the words said so far? And can it predict how long the switch will last? This repo labels the [SwitchLingua](https://huggingface.co/datasets/Shelton1013/SwitchLingua_text) corpus and trains prefix-only (causal) models on it.

- **Corpus:** 234,172 code-switched conversations, 15 language pairs. 14 first languages are paired with English, plus German→French.
- **Task:** at each token `t`, with only tokens `1..t` visible, predict:
  - `y_sw`: will token `t+1` be in the other language?
  - `y_dur`: if so, how long does the new run last? **Small** (1–2 tokens), **Medium** (3–6), **Large** (7+).

## Where things stand

| Stage | Status |
|---|---|
| Per-word language labels for the whole corpus | ✅ done (AnE-LID / script / dictionary, with names resolved by context) |
| Sub-word (XLM-R token) labels: language, switch, duration | ✅ done (`data/subword.parquet`) |
| Word-level causal transformer + baselines | ✅ done: switch F1 **0.503** vs 0.295 for the best training-free baseline; **0.383** zero-shot on 4 held-out pairs |
| Pretrained backbone (XLM-R) on sub-word labels | ⬜ next (Week 5) |

The course schedule and pace are in [`docs/TIMELINE.md`](docs/TIMELINE.md).

## Setup

```bash
python3.13 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

Always run things as `./.venv/bin/python …` from the repo root. The scripts import `src.*` relative to the root.

**Access you need** (both are gated or private):

1. **SwitchLingua.** Request access on its [Hugging Face page](https://huggingface.co/datasets/Shelton1013/SwitchLingua_text).
2. **Our labels.** They live in the private dataset `cristae/switchlingua-tagged-final`; ask Cristae to add you.

Then create a Hugging Face token (read access is enough) and put it in `.env`. `.env` is gitignored, so never commit it:

```
HF_TOKEN=hf_...
```

## Get the data

Nothing under `data/` is in git: the corpus is gated, and the labelled files are large. Download the final word-level labels:

```bash
./.venv/bin/hf download cristae/switchlingua-tagged-final --repo-type dataset --local-dir data/final
cp data/final/tagged.parquet data/tagged.parquet
```

`data/final/switchlingua_tagged_rows.csv` holds the same labels as a one-row-per-conversation CSV, if you'd rather not use parquet.

The sub-word labels need the original SwitchLingua text. Download the CSVs into the Hugging Face cache, which is where the scripts look for them. Then build the labels (about 3 minutes on CPU):

```bash
./.venv/bin/hf download Shelton1013/SwitchLingua_text --repo-type dataset --token "$HF_TOKEN"
./.venv/bin/python scripts/build_subword.py        # -> data/subword.parquet
```

## What the labels look like

**Word level** (`data/tagged.parquet`): one row per conversation, with parallel lists `tokens`, `token_lang`, `is_entity`, plus switch statistics. CJK scripts are split per character; everything else is split per word.

**Sub-word level** (`data/subword.parquet`): one row per conversation, one entry per XLM-R piece.

```
piece          lang  y_sw  y_len  y_dur
▁Creo          L1    0
...
▁usar          L1    1     1      small      <- next piece switches; the new run is 1 piece long
▁AI            L2    1     5      medium
▁para          L1    0
...
▁medicamentos  L1    1     9      large*     <- * = run still open when the text ends
,              ·                             <- punctuation/digits: no language, skipped
▁especially    L2    0
```

- **Piece language.** Each piece takes the language of the word it sits in.
- **Names** get the language their context implies ("Game of Thrones" is English; "Messi" in a Spanish clause is Spanish).
- **Cut-off runs.** A run cut off by the end of the text counts for duration only once it is 7+, because only then is its class certain.
- **Seeing examples.** Open the parquet with pandas: `pd.read_parquet("data/subword.parquet").iloc[0]`.

## Common commands

```bash
./.venv/bin/python -m pytest tests/ -q                   # 229 tests
./.venv/bin/python scripts/shuffle_parquet.py            # -> data/tagged_shuffled.parquet (train on this)
./.venv/bin/python scripts/baseline_anticipation.py      # training-free baselines, ~4 min
./.venv/bin/python scripts/train_anticipation.py --build-vocab --path data/tagged_shuffled.parquet --steps 6000   # ~23 min on an M3
./.venv/bin/python scripts/eval_anticipation_pair.py --ckpt models/anticipation_zeroshot.pt --l1 German --l2 French
./.venv/bin/python scripts/eval_public.py                # taggers vs human-annotated LinCE / Denglisch
./.venv/bin/python scripts/review_server.py              # spot-check UI for the labels (report-only)
```

Re-labelling the corpus from scratch needs a GPU (the AnE-NER and AnE-LID passes). Use the Kaggle notebook in `kaggle/label_switchlingua.ipynb` (T4×2). Locally, `scripts/run_tagging.py` does the same thing but slowly.

## Repo layout

```
src/          library code
  tokenizer.py, routing.py, dispatch.py,           the labelling pipeline: tokenize -> pick a tagger
  unicode_tagger.py, ane_tagger.py, dict_tagger.py   -> tag each word
  segments.py                                      names policies, run-length encoding
  persist.py                                       parquet I/O, cached model passes
  subword.py                                       word labels -> XLM-R pieces
  anticipation.py                                  y_sw / y_len / y_dur label building
  streaming.py, causal_model.py, metrics.py        dataloader, causal transformer, metrics
scripts/      command-line entry points (see "Common commands")
tests/        pytest suite; fixtures/ holds frozen model outputs for the name edge cases
kaggle/       GPU labelling notebook (generated by scripts/build_kaggle_notebook.py)
notebooks/    exploratory analysis
docs/         write-ups (start here)
archive/      retired experiments behind the appendix's negative results
data/         local only: small result JSONs are tracked, everything else is gitignored
```

## Read next

1. [`docs/groupUpdate1.md`](docs/groupUpdate1.md): the anticipation task. It covers objectives, the held-out design, causal constraints, baselines, results, and next steps.
2. [`docs/labelling_pipeline.md`](docs/labelling_pipeline.md): how the labels were made. It covers every tagger tried, its accuracy, and why it was kept or rejected.
3. [`docs/design/`](docs/design/): the design and implementation plan for the sub-word labels.
4. [`CLAUDE.md`](CLAUDE.md): a condensed architecture guide, which is also what Claude Code reads when working in this repo.

## Ground rules

- **No hand labelling.** Label accuracy is measured automatically, against human-annotated public sets (`eval_public.py`) and by agreement between two independent taggers (`consensus_corpus.py`). The review tool is a spot-check: if it reveals a systematic error, fix it in code and re-tag. It never writes labels back.
- **Causality.** Nothing from positions `t+1…` may reach position `t`. `check_causality` enforces this, and every model change must keep it passing.
- **Splits are row-level** (`row_id % 10`: 0–7 train, 8 val, 9 test), so a conversation never straddles two splits.
- **Licences.**
  - SwitchLingua is gated: don't publish corpus text or derived labels, and keep derived Hugging Face datasets private.
  - AnE-LID and AnE-NER (`igorsterner/AnE-*`) are licensed for academic use only.
