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
| Baselines + from-scratch causal transformer, words and pieces | ✅ done: switch F1 **0.513** (words) / **0.453** (pieces), against 0.329 / 0.266 for the best training-free baseline (per-pair hazard). See [`docs/model_comparison.md`](docs/model_comparison.md) |
| XLM-R with a causal mask | ✅ trained on Kaggle: switch F1 **0.646** (pieces), macro-F1 0.676, into L2 0.536, duration accuracy 0.678, better than the from-scratch model on all 15 pairs; 4 pairs above 0.7. Zero-shot on 4 held-out languages: F1 **0.625**. mBERT: 0.634 (zero-shot 0.565). See [`docs/results/`](docs/results/) |

The course schedule and pace are in [`docs/TIMELINE.md`](docs/TIMELINE.md).

## Setup

```bash
python3.13 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

Always run things as `./.venv/bin/python …` from the repo root. The scripts import `src.*` relative to the root.

**Access you need** (both are gated or private):

1. **SwitchLingua.** Request access on its [Hugging Face page](https://huggingface.co/datasets/Shelton1013/SwitchLingua_text).
2. **Our labels.** They live in the private dataset `CodeSwitchGroup4/switchlingua-tagged-final`; ask Cristae to add you to the `CodeSwitchGroup4` org.

Then create a Hugging Face token (read access is enough) and put it in `.env`. `.env` is gitignored, so never commit it:

```
HF_TOKEN=hf_...
```

## Get the labels

The labels are ready-made: every token already has its switch and duration targets. They live in the private Hugging Face dataset `CodeSwitchGroup4/switchlingua-tagged-final` (never in git: the corpus is gated). With `HF_TOKEN` in `.env`:

```bash
./.venv/bin/python scripts/download_labels.py          # -> data/final/token_labels_{word,subword}.parquet
./.venv/bin/python scripts/download_labels.py --all    # also the word-language files + data/tagged.parquet
```

| File | One row per conversation; lists aligned to its… |
|---|---|
| `token_labels_word.parquet` (82 MB) | words (`tokens`); CJK per character |
| `token_labels_subword.parquet` (108 MB) | XLM-R pieces (`pieces`, `piece_ids`), what an XLM-R model reads |

Columns in both:

| Column | Meaning |
|---|---|
| `lang` | `L1`, `L2`, or `-` (punctuation and digits, which are skipped) |
| `y_sw` | **1 if the next token switches language**, 0 if not, -1 where undefined |
| `y_len` | at a switch: **how many tokens the new language lasts** |
| `y_dur` | the class of `y_len`: `small` (1–2), `medium` (3–6), `large` (7+) |
| `censored` / `dur_known` | the new run hits the end of the text / its class is certain anyway. Train duration only where `dur_known` |
| `split` | `train` / `val` / `test` (`row_id % 10`) |

```python
import pandas as pd
df = pd.read_parquet("data/final/token_labels_word.parquet")
row = df.iloc[0]
list(zip(row.tokens, row.lang, row.y_sw, row.y_dur))[:10]
```

**Other names policies or a subset: `kaggle/load_labels.ipynb`.** Upload it to Kaggle, add your token as a secret named `HF_TOKEN`, turn Internet on, and choose in its first cell:
- the level: `word` or `subword`
- the names policy: `script_words` (default), `transparent` or `script`
- the pairs, splits and row limit

With the defaults it reproduces the two files above. It needs no GPU, and it also runs locally in Jupyter.

**Rebuilding the labels yourself** (maintainers). Download the original SwitchLingua CSVs into the Hugging Face cache, then:

```bash
./.venv/bin/hf download Shelton1013/SwitchLingua_text --repo-type dataset --token "$HF_TOKEN"
./.venv/bin/python scripts/build_subword.py            # -> data/subword.parquet (~3 min CPU)
./.venv/bin/python scripts/export_token_labels.py      # -> data/final/token_labels_*.parquet
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
./.venv/bin/python -m pytest tests/ -q                   # 309 tests
./.venv/bin/python scripts/shuffle_parquet.py            # -> data/tagged_shuffled.parquet (train on this)
./.venv/bin/python scripts/baseline_anticipation.py      # training-free baselines, ~4 min
./.venv/bin/python scripts/train_anticipation.py --build-vocab --path data/tagged_shuffled.parquet --steps 6000   # ~23 min on an M3
./.venv/bin/python scripts/baseline_anticipation.py --unit subword                    # the same baselines on XLM-R pieces
./.venv/bin/python scripts/train_anticipation.py --unit subword --build-vocab --steps 6000   # pieces only, no language input (~35 min)
./.venv/bin/python scripts/train_anticipation.py --model xlmr --unit subword --steps 12000 --batch-size 16  # GPU: use kaggle/train_pretrained.ipynb
./.venv/bin/python scripts/compare_models.py                                         # -> docs/model_comparison.md
./.venv/bin/python scripts/eval_anticipation_pair.py --ckpt models/anticipation_zeroshot.pt --l1 German --l2 French
./.venv/bin/python scripts/eval_public.py                # taggers vs human-annotated LinCE / Denglisch
./.venv/bin/python scripts/review_server.py              # spot-check UI for the labels (report-only)
```

Re-labelling the corpus from scratch needs a GPU (the AnE-NER and AnE-LID passes). Use `kaggle/label_switchlingua.ipynb` (T4×2). Locally, `scripts/run_tagging.py` does the same thing but slowly. After changing `src/`, regenerate the notebooks with `scripts/build_kaggle_notebook.py` and `scripts/build_labels_notebook.py`.

## How models are evaluated

Every model is trained and scored the same way, so the results are directly comparable:

1. **Split.** Train on the train split (80% of conversations, by `row_id % 10`). Tune the switch threshold on val (10%). Score test (10%) once. A conversation never straddles two splits.
2. **Causal reading.** A model reads each conversation left to right. At every token it may use only the tokens so far; the causal mask enforces this, and `check_causality` must pass before any training starts.
3. **The report** (`src/evaluation.py`):
   - Switch precision, recall and F1, with 95% intervals. Overall, **per language pair** (German→French kept apart), macro-averaged across pairs, and **per direction** (into L2 vs back into L1).
   - Duration: 3-class accuracy, macro-F1 and a confusion matrix, at the true switch points whose class is certain.
4. **Compare.** `scripts/compare_models.py` puts every result file side by side, overall and pair by pair → [`docs/model_comparison.md`](docs/model_comparison.md).

| Baseline | What it knows |
|---|---|
| never / always / random at the base rate | nothing (sanity checks) |
| `naive_l2` | "switch iff the current token is L2" |
| hazard (pooled / **per pair**) | P(switch \| current language, how long it has lasted), fitted on train |
| from-scratch causal transformer | a small model (15 M parameters on words, 23 M on pieces) trained on our labels only |
| **XLM-R + causal mask** | a pretrained multilingual model, fine-tuned (the target) |

**Training the pretrained models (XLM-R, RoBERTa, mBERT) on Kaggle.**
1. Upload `kaggle/train_pretrained.ipynb`, set `MODEL` (`"xlmr"`, `"roberta"` or `"mbert"`), choose GPU T4 ×2, turn Internet on, and add the `HF_TOKEN` secret.
2. Run All. Try `SMOKE = True` first.
3. Download `data/anticipation_<MODEL>.json` from the Output tab into `data/`, then run `./.venv/bin/python scripts/compare_models.py`.

## Repo layout

```
src/          library code
  tokenizer.py, routing.py, dispatch.py,           the labelling pipeline: tokenize -> pick a tagger
  unicode_tagger.py, ane_tagger.py, dict_tagger.py   -> tag each word
  segments.py                                      names policies, run-length encoding
  persist.py                                       parquet I/O, cached model passes
  subword.py                                       word labels -> XLM-R pieces
  token_labels.py                                  the shared per-token label files (lang, y_sw, y_len, y_dur)
  anticipation.py                                  y_sw / y_len / y_dur label building
  streaming.py, causal_model.py, metrics.py        dataloader, causal transformer, metrics
  xlmr_model.py, models.py                         XLM-R with a causal mask; build either model
  evaluation.py                                    the one evaluation report every model uses
scripts/      command-line entry points (see "Common commands")
tests/        pytest suite; fixtures/ holds frozen model outputs for the name edge cases
kaggle/       load_labels.ipynb: download our labels, choose word/sub-word + names policy (CPU)
              train_pretrained.ipynb: fine-tune and evaluate causal XLM-R / RoBERTa / mBERT (GPU)
              label_switchlingua.ipynb: re-label the corpus from scratch (GPU, T4x2)
notebooks/    switch_eda.ipynb: when switches happen and how long they last, by pair and persona
docs/         write-ups (start here)
data/         local only: small result JSONs are tracked, everything else is gitignored
```

## Read next

**Working on the project?** Start with [`docs/TEAM_PLAN.md`](docs/TEAM_PLAN.md) (who owns which language pairs, and the milestones) and [`CONTRIBUTING.md`](CONTRIBUTING.md) (setup, git workflow, how to report a result). Each pair has a page in [`docs/pairs/`](docs/pairs/), and Kaggle runs are logged in [`docs/RUN_LOG.md`](docs/RUN_LOG.md).

1. [`docs/labelling_models.md`](docs/labelling_models.md): which model labelled which pair, its accuracy on human-annotated data, and how names are handled.
2. [`docs/groupUpdate1.md`](docs/groupUpdate1.md): the anticipation task. It covers objectives, the held-out design, causal constraints, baselines, results, and next steps.
3. [`docs/labelling_pipeline.md`](docs/labelling_pipeline.md): the full history of the labels. It covers every tagger tried, its accuracy, and why it was kept or rejected.
4. [`notebooks/switch_eda.ipynb`](notebooks/switch_eda.ipynb): exploratory analysis of the labels: switch timing (hazard curves), duration classes, and persona and style effects, all per pair. It renders on GitHub.
5. [`docs/TIMELINE.md`](docs/TIMELINE.md): course milestones and pace.

## Ground rules

- **No hand labelling.** Label accuracy is measured automatically, against human-annotated public sets (`eval_public.py`) and by agreement between two independent taggers (`consensus_corpus.py`). The review tool is a spot-check: if it reveals a systematic error, fix it in code and re-tag. It never writes labels back.
- **Causality.** Nothing from positions `t+1…` may reach position `t`. `check_causality` enforces this, and every model change must keep it passing.
- **Splits are row-level** (`row_id % 10`: 0–7 train, 8 val, 9 test), so a conversation never straddles two splits.
- **Licences.**
  - SwitchLingua is gated: don't publish corpus text or derived labels, and keep derived Hugging Face datasets private.
  - AnE-LID and AnE-NER (`igorsterner/AnE-*`) are licensed for academic use only.
