# Project timeline

The committed schedule from the proposal. Week 3 is the week ending Sun 2026-09-27. Weeks are assumed to run Monday to Sunday; if that's wrong, shift every date.

Status key: ✅ done · 🟡 partial / done differently · ⬜ not started

| Week | Dates | Milestone | Status |
|---|---|---|---|
| 3 | Sep 21 – Sep 27 | Acquire SwitchLingua dataset, review language-pair selection | ✅ all 15 pairs, 4 held out (`groupUpdate1.md` §1) |
| 4 | Sep 28 – Oct 4 | Sub-word tokenization & LID mapping, generate switch and duration labels, build streaming dataloaders | ✅ LID mapping, final per-token labels (AnE-LID, `script_words` names; private HF `cristae/switchlingua-tagged-final`), **sub-word labels** (XLM-R pieces with per-piece language, switch, integer duration and the 3 required classes), and **streaming dataloaders for both units** (`src/streaming.py`, `--unit subword`: 384-piece windows, no language input). Piece-level baselines in `docs/groupUpdate1.md` §4.6 |
| 5 | Oct 5 – Oct 11 | Multilingual backbone setup (XLM-R / mBERT), implement causal attention mask | 🟡 **XLM-R backbone with a causal mask built** (`src/xlmr_model.py`, `is_decoder=True`, `check_causality` passes at 0.00 drift on the real model) and **trained on Kaggle (2026-09-29): switch F1 0.646 on pieces, macro-F1 0.676, 4 pairs above 0.7** (`docs/results/2026-09-29_xlmr_causal.md`). mBERT dropped: its tokenizer doesn't match the sub-word labels and gives 2.8% UNK on Cantonese |
| 6 | Oct 12 – Oct 18 | Build dual-task heads (switch & duration) | ✅ `head_sw` / `head_dur` in `src/causal_model.py` |
| 7 | Oct 19 – Oct 25 | Implement multitask loss, verify gradient flow & end-to-end unit testing | 🟡 Multitask loss (`--dur-weight`) and end-to-end tests are done. There is no explicit gradient-flow test yet |
| 8 | Oct 26 – Nov 1 | Train baseline models on 1–2 language pairs | ✅ exceeded: baselines run on all pairs; rerun on the final labels 2026-09-27 |
| 9 | Nov 2 – Nov 8 | Extend training to all pairs and tune | 🟡 Trained on all pairs, plus a zero-shot run, both rerun on the final labels (F1 0.503; zero-shot 0.383). No tuning sweep yet |
| 10 | Nov 9 – Nov 15 | Complete all training runs and checkpoint models | ⬜ Checkpoints exist for the from-scratch model only; the final (backbone) runs are still to come |
| 11 | Nov 16 – Nov 22 | Anticipatory F1 and duration accuracy | 🟡 Metrics implemented and reported for the current model |
| 12 | Nov 23 – Nov 29 | Macro-average accuracy, variance analysis | 🟡 Macro-F1 and Wilson CIs are done. There is no multi-seed variance analysis yet |
| 13 | Nov 30 – Dec 6 | Pair-specific breakdown, identify best "universal predictor" model | 🟡 A per-pair breakdown exists; there is only one model to compare so far |
| 14 | Dec 7 – Dec 13 | Research paper & streaming demo | ⬜ |
| 15 | Dec 14 – Dec 20 | Final GitHub repository and presentation | ⬜ |

## Pace check — 2026-09-27 (end of Week 3)

**Ahead of schedule.** Week 3 is complete. Most of Weeks 4–9 already exists for a from-scratch model, and the numbers are in `groupUpdate1.md` §4.

**Gaps against the named milestones.** These are the items that could become late:
1. **Sub-word tokenization** (Week 4): the tokenization is word-level today (`groupUpdate1.md` §5.3 item 4).
2. **XLM-R / mBERT as the anticipation backbone** (Week 5): not built yet (§5.3 item 6). Weeks 10–13 depend on it.
3. **Gradient-flow verification** (Week 7): no test yet.
4. **Multi-seed variance analysis** (Week 12).

**Work not on the timeline that blocks the claims:** a second non-English L2 (§5.3 item 2). Human review has been replaced by automated evaluation: the taggers are scored on human-annotated LinCE and Denglisch data, and production is checked against dict-freq across the whole corpus.

## Update — 2026-09-27

- **Labelling:** production for Latin-script rows with English as L2 switched to AnE-LID, which scores 99.4% on LinCE Spanish–English against 94.1% for dict-freq. Proper nouns now get a transparent `NE` label, and Devanagari tokens in romanized Hindi rows are fixed.
- **Re-tag done:** the final labels were built locally from the Kaggle v1 run and published privately to Hugging Face (`cristae/switchlingua-tagged-final`). Names now follow the `script_words` policy rather than being skipped.

## Update — 2026-09-27 (final labels)

- **All anticipation numbers regenerated** on the final labels: shuffle, training-free baselines, headline model, zero-shot model, the German pair splits and the curriculum ablation. `groupUpdate1.md` is updated throughout.
- **Headline:** causal transformer F1 0.503 (was 0.509) against 0.295 for the best training-free baseline. Duration bucket accuracy 0.533 against 0.428. Zero-shot on 4 held-out pairs: 0.383 (was 0.361).
- **Label noise:** Spanish, French, Italian and Vietnamese now fall inside the exact-pair noise band (1–4% of switches spent on single-token islands, down from 7–15%). Hindi, Tagalog and Malay stay high, most likely genuine insertion style.
- **Compute:** a 6,000-step run takes about 23 minutes on the local M3 (46 for the zero-shot run), so this model size doesn't need Kaggle. A pretrained backbone (Week 5) probably will.
- **Still open against the named milestones:** sub-word tokenization (Week 4), a pretrained backbone (Week 5), a gradient-flow test (Week 7), a multi-seed variance analysis (Week 12), and a second non-English L2.

## Update — 2026-09-27 (sub-word labels)

- **Sub-word token labels built** (`scripts/build_subword.py` → `data/subword.parquet`, 2.7 min on CPU). All 234,172 rows are covered, with 0 alignment failures. There are 21,740,404 XLM-R pieces; 155 of them (0.0007%) mix two languages.
- **Pieces:** 56.9% L1, 27.9% L2, 15.2% no language (punctuation and digits).
- **Switch labels:** 18,211,198 scored pieces and 1,896,074 switches (10.41%). That is within 4 of the word-level switch count, so the language boundaries carried over intact.
- **Duration labels:** now the professor's 3 classes, stored as integers. Among runs whose class is known: Small 26.8%, Medium 22.3%, Large 50.9%. That includes 145,049 cut-off runs of 7+; counting closed runs only, the split is 29.1% / 24.3% / 46.6%.
- **Next (Week 5):** stream the sub-word labels into a model with the XLM-R backbone.

## Update — 2026-09-28 (Week 4 complete)

- **Sub-word loader:** `SwitchWindowDataset(unit="subword")` streams `data/subword.parquet` in 384-piece windows (XLM-R's limit is 512). The model gets no language input on pieces (`CausalSwitchModel(use_lang=False)`). The scripts take `--unit subword`, and `--no-lang` for a fair word-level comparison.
- **Duration** is scored in the 3 required classes: accuracy, macro-F1 and a confusion matrix.
- **Baselines** (test split, best training-free):
  - Switch F1 is **0.295** on words, unchanged, and **0.235** on pieces.
  - The mode-per-direction duration baseline gets 0.585 accuracy on words and 0.509 on pieces.
- **Smoke test:** a 400-step sub-word model trains and passes the causality check. Its switch F1 of 0.259 is on only 7% of the test split, so it is not yet comparable with the baselines.
- **Next (Week 5):** the XLM-R backbone with a causal mask, trained on a Kaggle GPU.

## Update — 2026-09-28 (evaluation and the XLM-R backbone)

- **One evaluation for every model** (`src/evaluation.py`): switch F1 per pair and per direction, macro-F1 across pairs, and 3-class duration. The comparison lives in `docs/model_comparison.md`.
- **Per-pair hazard baseline:** switch F1 0.329 on words and 0.266 on pieces, against 0.295 / 0.235 for the pooled hazard. Fitted per pair, it no longer scores 0 on switches into L2 (0.227 on words).
- **From-scratch transformer,** re-scored:
  - Switch F1 0.513 on words and 0.453 on pieces; best on every pair.
  - Duration 3-class accuracy 0.712 on words; Medium is the hardest class.
- **XLM-R:** runs at 1.19 s/step on the M3 (7.7 h/epoch), so the full run goes to Kaggle. This is the Week 5 milestone in progress.

## Update — 2026-09-29 (XLM-R trained)

- **XLM-R + causal mask**, about 1 epoch on a Kaggle T4:
  - Switch F1 **0.646** on pieces and macro-F1 **0.676**, against 0.453 for the from-scratch model and 0.266 for the per-pair hazard. Into L2: 0.536. Duration accuracy 0.678, macro-F1 0.575.
  - Better on all 15 pairs (+0.15 to +0.27). Korean, German→English, Japanese and Hindi are above 0.7.
  - Archived in `docs/results/`.
- **Week 5 milestone met** (a multilingual backbone with a causal mask). Weeks 6–8 are covered too: dual heads, multitask loss, baselines.
- **Still to run:**
  1. ~~Download `data/anticipation_xlmr.json`~~ Done: duration and per-direction are in the results. Two independent runs agree within ±0.004 per pair.
  2. ~~The XLM-R zero-shot run.~~ Done: held-out F1 0.625, 12.6% below trained-on. German→French transfers (0.450 against always-switch 0.167).
  3. The from-scratch zero-shot rerun with the new report, as its comparison.
  4. Seeds for the variance analysis (Week 12).
