# Contributing

How we work in this repo. What the project is is in [`README.md`](README.md).

## 1. One-time setup

1. **Python.**
   ```bash
   python3.13 -m venv .venv
   ./.venv/bin/pip install -r requirements.txt
   ./.venv/bin/python -m pytest tests/ -q          # everything should pass
   ```
   Always run things as `./.venv/bin/python …` from the repo root.
2. **Access** (ask Cristae for the first; request the second yourself):
   - **Our labels:** the private Hugging Face dataset `CodeSwitchGroup4/switchlingua-tagged-final`.
   - **SwitchLingua:** the gated original text, [`Shelton1013/SwitchLingua_text`](https://huggingface.co/datasets/Shelton1013/SwitchLingua_text). It is needed to rebuild sub-word labels and for the Kaggle notebook.
3. **Token.** Create a Hugging Face **read** token. Put `HF_TOKEN=hf_...` in `.env` (gitignored), and add the same token as a Kaggle secret named `HF_TOKEN`.
4. **Data.**
   ```bash
   ./.venv/bin/python scripts/labelling/download_labels.py --all     # labels -> data/final/, data/tagged.parquet
   ./.venv/bin/hf download Shelton1013/SwitchLingua_text --repo-type dataset --token "$HF_TOKEN"
   ./.venv/bin/python scripts/labelling/build_subword.py             # -> data/subword.parquet (XLM-R pieces, ~10 min)
   ```

## 2. Git workflow

- **Branch** from `chris_branch`, one branch per task: `<name>/<topic>`, e.g. `maya/german-french-homographs`.
- **Code changes:**
  - **Tests first:** write a failing test in `tests/` that shows the problem or the new behaviour, then make it pass.
  - Run the whole suite before you open a PR.
- **Open a pull request into `chris_branch`.** One other member reviews it, and the training lead merges.
- **Never commit:**
  - `.env` or any token;
  - parquet or CSV labels, or corpus text (SwitchLingua is gated, so derived labels stay in the private HF dataset);
  - model weights (`models/`).

  `.gitignore` already blocks `data/` (except result JSONs) and `models/`. Check `git status` before committing anyway.
- **Commit messages:** a short summary line, then what changed and why.

## 3. Rules that keep results comparable

These are what make our numbers trustworthy. Don't change them without a team decision recorded in `docs/results/`.

- **Splits** are by conversation: `row_id % 10`, where 0–7 is train, 8 val and 9 test. A conversation never straddles two splits.
- **Causality:** the model may only use tokens up to the current one. `check_causality` runs before every training run and must print `PREFIX-ONLY OK`. Any model change keeps it passing.
- **Thresholds** are tuned on **val**. **Test** is scored once.
- **Metrics:**
  - Switch prediction: F1 on the switch class (never accuracy), per pair, macro-F1 across pairs, and per direction.
  - Duration: 3-class macro-F1 at true switch points.
  - Across tokenizers, compare **shared F1** (the same words for every model).
- **Seeds:** use `--seed` (`SEED` in the notebook) for any run you report, and say which.
- **No hand labelling.** `review_server.py` is a spot-check. Fix systematic label errors in code and re-run the labelling.
- **Licences:** SwitchLingua is gated, so keep derived datasets private. AnE-LID and AnE-NER are for academic use only.

## 4. Running models

| Where | What | Command |
|---|---|---|
| Laptop | baselines | `./.venv/bin/python scripts/training/baseline_anticipation.py --unit subword` |
| Laptop | from-scratch model (~35 min on an M3) | `./.venv/bin/python scripts/training/train_anticipation.py --unit subword --steps 6000 --seed 0 --save-preds data/preds/<name>.npz --out data/<name>.json --ckpt models/<name>.pt` |
| Kaggle (GPU) | XLM-R or mBERT | `kaggle/train_pretrained.ipynb`: set `MODEL` and the tuning flags in its first cell; `SMOKE = True` first |

**Tuning flags** (all off by default): `--seed`, `--save-preds`, `--dur-class-weights balanced`, `--prev-lang`, `--predict-at word_end`, `--turn-sep`. What each does, and how much it helped, is in [`docs/results/2026-09-30_ablations.md`](docs/results/2026-09-30_ablations.md).

## 5. Reporting a result

1. **Results file:** put the run's JSON in `data/results/` (the Kaggle Output tab has `data/results/anticipation_<TAG>.json`) and its predictions in `data/preds/`.
2. **Dated record:** add `docs/results/YYYY-MM-DD_<topic>.md` with the settings (seed, steps, flags), the tables and what they mean. Add a row to `docs/results/README.md`.
3. **Scorecard:** run `./.venv/bin/python scripts/analysis/compare_models.py` to refresh `docs/model_comparison.md`. Add new result files to its `DEFAULT_SPECS` if they aren't picked up.
4. **Your pairs:** run `./.venv/bin/python scripts/analysis/pair_report.py --pair "<L1>-><L2>" --preds data/preds/<file>.npz --md docs/pairs/<L1>-<L2>.md`, then add your notes below the generated part.
5. **Commit the JSON and the docs** (not the `.npz` predictions or the checkpoint), and open a PR.

State numbers with their unit (words or which tokenizer's pieces) and the seed. A single run's ±0.005 isn't a finding; the Week 12 seeds give the intervals.

## 6. Useful tools

| Tool | What it's for |
|---|---|
| `scripts/analysis/pair_report.py` | one pair across all models, plus its worst errors as text |
| `scripts/analysis/tune_thresholds.py` | re-tune thresholds on saved predictions (no GPU) |
| `scripts/analysis/compare_models.py` | the full scorecard |
| `scripts/training/eval_anticipation_pair.py` | score a checkpoint on one exact pair |
| `scripts/labelling/review_server.py --lang <L1>` | spot-check labels in the browser (report-only) |
| `scripts/labelling/export_token_labels.py`, `kaggle/load_labels.ipynb` | per-token label files and other names policies |
| `notebooks/switch_eda.ipynb` | when switches happen and how long they last, per pair |
