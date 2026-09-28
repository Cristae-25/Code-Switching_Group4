# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research project on the SwitchLingua corpus (`Shelton1013/SwitchLingua_text`, 234,172 rows, 14 first languages paired mostly with English, plus German→French). It has two stages:

1. **Labelling pipeline.** Tag every token with its language, then run-length encode the tags. Output: `data/tagged.parquet`.
2. **Anticipatory task.** From a prefix alone, predict whether token t+1 switches language (`y_sw`) and how long the new run lasts (`y_dur`). The labels are *derived* from stage 1's tags, so tagger quality sets the ceiling on label quality.

The long-form write-ups are the source of truth for design decisions and results. Read them before changing behaviour:
- `docs/groupUpdate1.md`: the anticipatory task (objectives, held-out design, causal constraints, baselines, results).
- `docs/labelling_pipeline.md`: every tagging technique tried, its accuracy, and why it was kept or rejected.
- `archive/data_pipeline_plan.md`: the original plan. It is **superseded** in several places (routing, tokenizer regex, the Arabic pair), so trust the code and the docstrings over it.

## Environment and commands

Always use `./.venv/bin/python`. The venv pins transformers 5.17 / torch 2.14, because transformers 4.47.1 segfaults under torch 2.10 (exit 139, no traceback). `requirements.txt` pins the versions installed in `.venv`. Use `fasttext-predict`, not `fasttext`, which fails to build on Python 3.13.

```bash
./.venv/bin/python -m pytest tests/ -q                                  # full suite
./.venv/bin/python -m pytest tests/test_segments.py::test_name -q       # single test
./.venv/bin/python scripts/run_tagging.py [--limit 2000] [--no-entities] # tag corpus -> data/tagged.parquet (NE pass cached)
./.venv/bin/python scripts/shuffle_parquet.py                           # -> data/tagged_shuffled.parquet
./.venv/bin/python scripts/build_subword.py                             # -> data/subword.parquet: XLM-R pieces + per-piece language, y_sw, y_len, y_dur (~3 min CPU)
./.venv/bin/python scripts/baseline_anticipation.py                     # training-free baselines (~4 min)
./.venv/bin/python scripts/train_anticipation.py --build-vocab --path data/tagged_shuffled.parquet --steps 6000
./.venv/bin/python scripts/eval_anticipation_pair.py --ckpt models/anticipation_zeroshot.pt --l1 German --l2 French
./.venv/bin/python scripts/build_kaggle_notebook.py                     # -> kaggle/label_switchlingua.ipynb (embeds src/; rebuild after src changes)
./.venv/bin/python scripts/review_server.py [--per-pair 15] [--report]  # spot-check UI on data/tagged.parquet; report-only
./.venv/bin/python scripts/apply_names_policy.py data/hf/tagged.parquet --from-v1 --out data/final/tagged.parquet  # final labels + CSV, no model
./.venv/bin/python scripts/compare_name_strategies.py                   # every names policy on the edge cases + corpus
./.venv/bin/python scripts/eval_public.py [--limit 64]                  # taggers vs human-annotated LinCE dev + Denglisch
./.venv/bin/python scripts/consensus_corpus.py [--rows-per-pair 500]    # production (AnE) vs dict-freq agreement on the corpus
```

Scripts add the repo root to `sys.path` and import `src.*`, so run them from the repo root. `run_tagging.py` reads the cached HF CSVs from `~/.cache/huggingface/hub/datasets--Shelton1013--SwitchLingua_text/` and only falls back to the Hub (HF token in `.env`) if they are missing. The parquet, sqlite and model files are gitignored.

## Architecture

### Labelling pipeline (`src/`)

`tokenizer.tokenize_mixed` → `dispatch.tag_tokens` → `segments.absorb_other` / run-length encoding → `persist` (parquet).

- **`tokenizer.py`** uses mixed granularity. Han, Kana and Hangul are split **per character**; Latin (with combining marks), Cyrillic, Arabic and Devanagari are split per word. Use `\p{Script=Latin}`, never `[A-Za-z]`, which splits accented words apart. Because of the per-character split, a CJK token is not the same unit as a Latin one, so compare run lengths within a script, not across scripts.
- **`routing.route(first_language, second_language)`** decides the tagger tier, and it is authoritative:
  - `unicode` for the script-disjoint languages (Cantonese, Chinese, Japanese, Korean, Russian, Arabic).
  - `ane` (AnE-LID, `src/ane_tagger.py`) for every other L1 with English as L2, including romanized Hindi.
  - `dict` (dict-freq) for German→French, because AnE only decides "English or not".

  Several tables in `routing.py` are historical measurement records rather than live routing: `MODEL_FOR`, `USES_DICT`, `V1_PASSES`, the fastText codes, and the comment tables. fastText and both mBERT tiers are **retired**; `bert` can still be dispatched for comparisons.
- **`dispatch.py`** is the single branch point. `row_strategy` re-routes Hindi **per row**: Devanagari-majority rows go to `unicode`. In romanized rows, any Devanagari token is forced to Hindi, overriding the model. Heavy imports (torch, the models) are lazy, so unicode and dict rows never load them.
- **The two model passes are batched and cached per `row_id`:**
  - AnE-NER → `data/entities.parquet`, every row, about 8 h.
  - AnE-LID → `data/ane_lid.parquet`, `ane` rows only.

  `run_tagging.py` computes both passes (resuming if interrupted), then tags each row using the cached labels. Delete a cache to force a recompute.
- **`unicode_tagger.py`** is exact by construction. **`dict_tagger.tag_frequency`** compares wordfreq Zipf scores across the two languages; a word unknown to both becomes `other`. **`bert_tagger.py`** maps the `ne` label to `other`, and caches models on the `sys` module on purpose, so that IPython `%autoreload` doesn't reload the 679 MB model on every row.
- **Named entities** are a flag, not a label. `token_lang` is always a language, and `is_entity` (from AnE-NER) marks name tokens.
  - **How names get their language:** AnE-LID assigns it from context (a title made of English words → English; an opaque name → its clause's language). In script-tagged rows, `dispatch._names_from_context` gives Latin-script names AnE-LID's label, computed on ±16-token windows (`persist.ane_labels(entities=...)`). Names in the row's own script stay L1.
  - **How names count is a policy** (`segments.resolve_names`; every edge case is pinned in `tests/test_name_strategies.py`). `repair_spans` first mends AnE-NER's fragmented names ("Severance [Season 2]", "Game [of] Thrones").
    - `"script_words"` (**default**, `DEFAULT_NAMES`): a name is a switch only with visible evidence. In non-Latin rows, a Latin-script name is English and a native-script name is L1. In Latin rows, the name's words decide when they're clearly one language's (by a Zipf margin of ≥ 1.0); otherwise it takes its neighbours' language and can't open a run of its own.
    - `"transparent"`: names are skipped (LinCE's convention).
    - `"script"`: every Latin-script name is English.
    - `"hybrid"` and `"language"` are the other two.

    The parquet stores `l2_runs` (default), `l2_runs_transparent` and `l2_runs_script`; the anticipation scripts take `--names`. `scripts/compare_name_strategies.py` prints every policy on the edge cases, plus a corpus estimate.
  - **Provenance of the final labels.** `data/tagged.parquet` (and `data/final/`) was built with `apply_names_policy.py --from-v1` from the v1 Kaggle labels (`cristae/switchlingua-tagged`, downloaded to `data/hf/`). Non-name tokens are exactly v1's. v1 recorded names as `NE`, so a name's `token_lang` is reconstructed (script or dict-freq); the default policy only uses it for a name sitting on an unpunctuated switch. The final labels are published privately as `cristae/switchlingua-tagged-final` on Hugging Face: CSV, parquet and a dataset card. Uploads use the write token `HF_FULL` in `.env`; `HF_TOKEN` is read-only. `data/tagged_pre_ane.parquet` holds the older dict-freq labels.
- **`segments.absorb_other`** folds `other` tokens (punctuation, digits, NEs) into the neighbouring run. It must never bridge two different languages; at a real boundary, the `other` token joins the *following* run. `recompute_runs` must agree exactly with `tag_row`, and a test enforces this.
- **`persist.py`** stores the ragged data as parallel primitive lists (`tokens`/`token_lang` and `segment_lang`/`segment_len`) because nested tuples don't round-trip through parquet. The source text is not stored; join on `row_id` to get it.
- `routing.tagger_quality` labels each row `exact` (script-disjoint) or `provisional`, and that label travels with every row.
- **Evaluation is fully automated, with no hand labelling.** This is a project requirement.
  - **Accuracy:** `scripts/eval_public.py` scores each tagger against human-annotated public sets, downloaded to `data/public/`: the LinCE spaeng/hineng dev splits and Denglisch. AnE-LID saw Denglisch in training, so its score there is an upper bound.
  - **Coverage:** `scripts/consensus_corpus.py` compares production (AnE-LID) with dict-freq on every `ane` row; the two taggers share no evidence. It writes `data/consensus.parquet`. On the public sets, tokens where the two agree were 99.3–100% correct.
  - `data/gold_eval.json` is saturated (every tagger is at about 99–100%) and its labels came from a model, so it can no longer rank taggers.
- `validate.py` checks tag ratios against the dataset's own `cs_ratio_result`. That signal is noisy: the exact-tagger noise band is |err| ≤ 0.134. `data/gold_eval.json` has 324 hand-labelled tokens and is the primary accuracy measure.

### Anticipatory task

- **`anticipation.py`** builds labels with three deliberate choices: punctuation is dropped from the stream (it neither breaks nor counts toward a run), the final token has no label (it is masked, not set to 0), and a run still open at the end of the text is right-censored and flagged. The language stream is binarised as L1/L2 relative to the row's own pair.
- **Duration labels** are stored as the integer `y_len`, with the classes Small 1–2 / Medium 3–6 / Large 7+ derived from it (`anticipation.dur_class`; the project requirement). A run cut off by the end of the text counts only once it is 7+ (`dur_known`), because only then is its class certain. Slot `NONE` (= `streaming.PAD_LANG`: punctuation, digits, a bare `▁`) is transparent: it neither breaks nor extends a run.
- **Sub-word labels** (`src/subword.py`, `scripts/build_subword.py`): the original CSV text is tokenized with the `xlm-roberta-base` tokenizer, and each piece takes the language of the word its characters fall in (character offsets; names resolved first, so a name's pieces keep its `script_words` language). In CJK, a merged piece such as 東京 is one unit. XLM-R emits a bare `▁` piece carrying the *next* character's offsets, so blank pieces are NONE. The models still train on word units; sub-word training (and dropping the language input there) is Week 5.
- **`streaming.py`** is an `IterableDataset` over `ParquetFile.iter_batches`; it never loads the whole corpus. Splits are **row-level**: `row_id % 10` gives 0–7 train, 8 val, 9 test. Long rows are cut into overlapping windows, and the first `window - stride` positions of each continuation window are used as context but masked out of the loss. Workers shard by batch index, and shuffling uses a bounded reservoir buffer.
- Train on `tagged_shuffled.parquet`. The source file is contiguous per language, so streaming it in order becomes an accidental curriculum by language.
- **`causal_model.py`**: nothing from positions t+1 onward may reach position t. The model uses an explicit `-inf` upper-triangular mask plus `is_causal=True`, only LayerNorm (BatchNorm over time would leak), and inputs shifted against labels. `check_causality` perturbs the future tokens and asserts the logits don't change. Keep it passing after any model change.
- **`metrics.py`**: switches are the rare class, so report positive-class P/R/F1, macro-F1 across pairs, and Wilson intervals, never accuracy.
- Zero-shot runs hold out Korean, Italian, Hindi and German (`--train-langs`). A holdout forces `--no-pair`, because the pair embedding for an unseen pair would be untrained. Evaluate German→French and German→English separately with `eval_anticipation_pair.py`; filtering on `first_language` alone would merge them.

## Schedule

`docs/TIMELINE.md` holds the course milestones, Weeks 3–15. Keep its status column and its "Pace check" section up to date when milestone work lands.

## Open state

The v1 labels come from the Kaggle notebook and live at `cristae/switchlingua-tagged` (a private HF dataset). They are downloaded to `data/hf/`; the read token in `.env` can access them. The final labels in `data/tagged.parquet` are derived from them locally (see "Provenance of the final labels" above). Label quality is measured by `eval_public.py` and `consensus_corpus.py`. `scripts/review_server.py` is a report-only **spot-check** for finding systematic errors by eye; it never writes labels back, and fixes go into code followed by a re-tag. `data/tagged_pre_ane.parquet` holds the labels from before the AnE-LID and NE re-tag, for diffing.

`archive/` holds the retired experiments behind the appendix's negative results (XLM-R and CANINE fine-tunes, and the dict-vs-fastText-vs-mBERT comparison in `eval_dict.py`). They still run from the repo root, but their model weights were deleted: XLM-R and CANINE must be retrained, and fastText's `lid.176.bin` re-downloaded (`src/model_tagger.py` prints the URL).
