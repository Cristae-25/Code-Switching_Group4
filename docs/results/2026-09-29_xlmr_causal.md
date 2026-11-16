# 2026-09-29: XLM-R with a causal mask, all 15 pairs (sub-word)

**Run:** `kaggle/train_xlmr.ipynb` (now `kaggle/train_pretrained.ipynb` with `MODEL = "xlmr"`) on a Kaggle T4. Model `xlm-roberta-base` with `is_decoder=True` (causal attention), 277.5 M parameters, plus a switch head and a 3-class duration head. The inputs are XLM-R pieces only, with no language input. `check_causality` gave 0.00 drift over 31 prefixes before training.

**Settings,** from the results file: 12,000 steps, batch 16, window 384 / stride 288, learning rate 2e-05 for the body and 0.001 for the heads, duration loss weight 0.3, trained on all pairs.

**Protocol:** the same as every model (`src/evaluation.py`). Train on the train split (`row_id % 10` in 0–7), tune the threshold on val (8; τ = 0.337, val F1 0.647), score test (9) once.

**Files:**
- [`2026-09-29_xlmr_causal.json`](2026-09-29_xlmr_causal.json) — the full results file, also at `data/results/anticipation_xlmr.json`.
- [`2026-09-29_xlmr_causal_run2_per_pair.csv`](2026-09-29_xlmr_causal_run2_per_pair.csv) — the per-pair table of a second, independent training with the same settings, as the notebook printed it. The zero-shot Run All re-trains the full model first, so there were two runs.

## Switch prediction (`y_sw`), test split

The per-pair test sizes are identical to the from-scratch sub-word model's (for example, Cantonese 1,061,851), so the comparison is on exactly the same positions.

| Model (pieces) | Switch F1 (P / R) | Macro-F1 over pairs | F1 into L2 | F1 back into L1 |
|---|---|---:|---:|---:|
| Hazard, per pair (best training-free) | 0.266 (0.181 / 0.499) | 0.259 | 0.217 | 0.299 |
| From-scratch causal transformer (23 M params) | 0.453 (0.398 / 0.526) | 0.453 | 0.356 | 0.560 |
| **XLM-R + causal mask** | **0.646** (0.625 / 0.669) | **0.676** | **0.536** | **0.759** |

| Pair | **XLM-R** F1 | P | R | Run 2 F1 | From-scratch | Gain | Hazard, per pair |
|---|---:|---:|---:|---:|---:|---:|---:|
| Korean | **0.808** | 0.775 | 0.844 | 0.809 | 0.611 | +0.197 | 0.265 |
| Japanese | **0.736** | 0.694 | 0.783 | 0.734 | 0.527 | +0.208 | 0.243 |
| German | **0.733** | 0.695 | 0.776 | 0.735 | 0.469 | +0.265 | 0.268 |
| Hindi | **0.715** | 0.639 | 0.811 | 0.717 | 0.541 | +0.174 | 0.362 |
| Chinese | 0.699 | 0.647 | 0.761 | 0.698 | 0.429 | +0.270 | 0.242 |
| French | 0.695 | 0.644 | 0.755 | 0.698 | 0.453 | +0.242 | 0.208 |
| Spanish | 0.673 | 0.626 | 0.727 | 0.671 | 0.425 | +0.248 | 0.239 |
| Italian | 0.671 | 0.632 | 0.716 | 0.675 | 0.406 | +0.266 | 0.242 |
| Arabic | 0.664 | 0.617 | 0.717 | 0.667 | 0.412 | +0.251 | 0.241 |
| Russian | 0.659 | 0.631 | 0.690 | 0.661 | 0.387 | +0.272 | 0.243 |
| German→French | 0.658 | 0.647 | 0.669 | 0.661 | 0.387 | +0.271 | 0.223 |
| Tagalog (Philippines) | 0.638 | 0.559 | 0.745 | 0.639 | 0.492 | +0.147 | 0.323 |
| Vietnamese | 0.622 | 0.603 | 0.642 | 0.622 | 0.411 | +0.211 | 0.248 |
| Cantonese | 0.619 | 0.621 | 0.618 | 0.621 | 0.442 | +0.177 | 0.252 |
| Malay | 0.548 | 0.530 | 0.567 | 0.549 | 0.397 | +0.151 | 0.281 |

**Run to run:** the two independent XLM-R trainings agree within **±0.004 F1 on every pair** (run 2's overall F1, rebuilt from its per-pair P/R, is 0.647). That is a first, informal variance check; the Week 12 analysis needs seeded runs.

## Duration (`y_dur`), at true switch points whose class is certain (n = 180,990)

| Model (pieces) | 3-class accuracy | Macro-F1 | Within one class | F1 Small / Medium / Large |
|---|---:|---:|---:|---|
| Mode per direction (training-free) | 0.509 | 0.225 | 0.732 | — |
| From-scratch transformer | 0.636 | 0.482 | 0.863 | 0.591 / 0.079 / 0.776 |
| **XLM-R + causal mask** | **0.678** | **0.575** | **0.910** | 0.636 / 0.278 / 0.813 |

Confusion matrix (rows = truth, columns = prediction):

| | pred small | pred medium | pred large |
|---|---:|---:|---:|
| **small (1–2)** | 31,664 | 5,965 | 10,860 |
| **medium (3–6)** | 14,015 | 8,078 | 18,216 |
| **large (7+)** | 5,461 | 3,740 | 82,991 |

## Reading

- **Pretraining is worth about +0.2 F1 everywhere.** XLM-R beats the from-scratch model on all 15 pairs, by +0.147 to +0.272 (mean +0.223), and beats the best training-free baseline by more than 2×. Four pairs pass F1 0.7: Korean, Japanese, German→English and Hindi.
- **Both directions improve by about +0.2, including the hard one.** Switching back into L1 goes from 0.560 to 0.759. Switching *into* the second language goes from 0.356 to **0.536**. The into-L2 direction is where run length carries little signal and a model has to anticipate from what is being said (report §4.3), and it is the one that matters for the research question.
- **The biggest per-pair gains (about +0.27) are on Russian, German→French, Chinese and Italian;** Arabic gains +0.25. Russian, Italian and Arabic are the pairs whose words were most often out of vocabulary for the word-level model (Russian 12.6%, Arabic 11.3%, Italian 6.6%, report §5.1).
- **Macro-F1 is above micro-F1 (0.676 vs 0.646).** Cantonese, half the test positions, is one of the weaker pairs (0.619). The model is better on the typical pair than on the largest one.
- **The hardest pairs are the ones that switch mid-clause:** Malay, Cantonese, Vietnamese and Tagalog. The EDA (§2c) shows these rarely switch at punctuation.
- **Duration improves, and Medium is still the problem.** Medium F1 rises from 0.079 to 0.278, but most Medium runs are still predicted Large (18,216) or Small (14,015). Class-weighted duration loss is the obvious next experiment; the EDA computed the weights.
- **German→French reaches 0.658** (+0.271), far above its trivial always-switch level (about 0.17). It was trained on in this run, so this is **not** evidence about the zero-shot question (report §4.5). The zero-shot run answers that.
