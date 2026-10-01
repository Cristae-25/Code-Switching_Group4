# 2026-09-30: Tuning ablations (local, from-scratch model on XLM-R pieces)

**Why:** to find out which training changes are worth spending Kaggle GPU hours on XLM-R. Each change was tested on the cheap local model first.

**Setup:**
- **Model:** the from-scratch causal transformer (23 M params) on XLM-R pieces.
- **Run:** 6,000 steps on an M3, about 35 minutes each.
- **Seed:** every run uses `--seed 0`, so the initialisation and data order are identical. Only the flag differs.
- **Evaluation:** the threshold is tuned on val and test is scored once.
- **Predictions:** saved with `--save-preds` (in `data/preds/`, gitignored).
- **Fair switch metric: shared F1.** It is scored only at the last piece of each non-CJK word, the same 747 K test words in every run. `--predict-at word_end` changes which positions are scored at all, so the raw F1 is not comparable across runs.

## Results (test split)

| Flag (one change vs. the baseline) | Shared switch F1 | Shared macro-F1 | Shared F1 into L2 / into L1 | Duration accuracy | Duration macro-F1 | Duration F1 Small / Medium / Large |
|---|---:|---:|---:|---:|---:|---|
| baseline (no flags) | 0.547 | 0.498 | 0.423 / 0.602 | 0.637 | 0.481 | 0.592 / 0.075 / 0.777 |
| `--dur-class-weights balanced` | 0.547 | 0.496 | 0.422 / 0.602 | 0.602 | **0.540** | 0.583 / **0.295** / 0.743 |
| `--turn-sep` | 0.546 | 0.496 | 0.423 / 0.601 | 0.637 | 0.482 | 0.593 / 0.077 / 0.777 |
| `--prev-lang` | **0.556** | **0.510** | 0.428 / 0.610 | 0.640 | 0.490 | 0.596 / 0.096 / 0.780 |
| `--predict-at word_end` | 0.557 | 0.510 | 0.433 / 0.615 | 0.638 | 0.483 | 0.594 / 0.077 / 0.778 |
| **all three adopted flags** (`--dur-class-weights balanced --prev-lang --predict-at word_end`) | **0.570** | **0.528** | **0.445 / 0.627** | 0.608 | **0.554** | 0.592 / **0.322** / 0.747 |

Raw switch F1 on each run's own positions: 0.450, 0.452, 0.450, 0.461, and 0.488 for word_end, which scores 1.62 M positions instead of 1.83 M. The duration positions (180,990 switches with a certain class) are the same in every run.

**Balanced duration weights,** from the train split's class counts (387,080 / 323,108 / 735,639): Small 1.10, Medium 1.32, Large 0.58 (mean 1). They are the same with `--predict-at word_end`, because the duration targets sit at the same positions.

## Offline thresholds (no GPU)

`scripts/tune_thresholds.py` re-tunes the switch threshold on the saved val predictions, per language pair or per direction, and applies it to test:

| Run | Global threshold: F1 / macro-F1 | Per-pair: F1 / macro-F1 | Per-direction: F1 / macro-F1 |
|---|---|---|---|
| baseline | 0.450 / 0.450 | 0.449 / 0.452 | 0.445 / 0.447 |
| `--prev-lang` | 0.461 / 0.462 | 0.460 / 0.465 | 0.454 / 0.457 |
| `--predict-at word_end` | 0.488 / 0.491 | 0.486 / 0.492 | 0.481 / 0.487 |

## Reading

- **Class-weighted duration loss: adopt.** Medium F1 rises 4× (0.075 → 0.295) and duration macro-F1 by +0.059, with switch prediction unchanged. Accuracy falls 3.5 points because the model stops defaulting to Large. Macro-F1 is the metric we report for an imbalanced 3-class task, so this is the right trade.
- **Previous-word language: adopt.** It adds +0.009 shared F1 and +0.012 macro-F1, with small gains on both directions and on duration. The input is causal: at each piece it is the language of the last word that had *finished*.
- **Word-end prediction: adopt.** It adds +0.010 shared F1 and +0.013 macro-F1, and is the best on switching *into* L2 (+0.010). Training only on the decisions that matter, the last piece of each word, helps, and the model still reads every piece.
- **Turn separators: drop.** No effect (−0.001).
- **Per-pair thresholds: no gain** (at most +0.003 macro-F1). The model is already calibrated across pairs, so one global threshold is enough.
- **Combined, the three adopted flags add up:** +0.023 shared F1, +0.030 macro-F1, +0.022 into L2, and +0.073 duration macro-F1, against the baseline with the same seed. That is a little more than the two switch gains summed, and 4–5× the run-to-run noise. This is the configuration to carry to XLM-R.
- **Noise.** Unseeded reruns of the same setup varied by about ±0.005 switch F1. The +0.01 gains are about twice that, so they are suggestive, not conclusive. The class-weight effect on duration is far beyond it. The Week 12 seed analysis will put proper intervals on these.
