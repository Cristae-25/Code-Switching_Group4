# 2026-10-12: XLM-R trained on the 6 script-disjoint pairs, plain vs class-weighted duration

**Runs:** `kaggle/train_pretrained.ipynb` on a Kaggle T4, two runs with identical settings except the duration loss.

| TAG | Duration loss | Training time |
|---|---|---:|
| `xlmr_6pairs` | plain cross-entropy | 69.6 min |
| `xlmr_durw_6pairs` | class-weighted: Small 1.197, Medium 1.266, Large 0.537, from the train split's counts (240,740 / 227,656 / 536,640) | 78.0 min |

**Settings:** XLM-R base with a causal mask, 12,000 steps, `SEED = 0`, `DUR_HEAD = "softmax"`. Trained on Cantonese, Chinese, Japanese, Korean, Russian and Arabic → English only; the other 9 pairs are scored as held out. Both runs passed `check_causality` (drift 0.0) and score the same 1,344,905 test positions, so every comparison below is paired.

**Protocol:**
- **Switch threshold:** τ = 0.377 in both runs, tuned on val.
- **Duration:** scored at the 125,675 true switch points whose class is certain. The raw argmax and the val-tuned decision (`duration_tuned`, offsets fit on val) are both reported.
- **Intervals:** 95% intervals come from a row-level paired bootstrap, 1,000 resamples (`scripts/bootstrap_preds.py`).

**Files:**
- results JSONs: [`2026-10-12_xlmr_6pairs.json`](2026-10-12_xlmr_6pairs.json) and [`2026-10-12_xlmr_durw_6pairs.json`](2026-10-12_xlmr_durw_6pairs.json);
- bootstrap intervals: [`2026-10-12_bootstrap_6pairs_vs_durw.json`](2026-10-12_bootstrap_6pairs_vs_durw.json);
- predictions: `data/preds_xlmr_6pairs.npz` and `data/preds_xlmr_durw_6pairs.npz`, gitignored.

## 1. Switch prediction: weighting changes nothing

| Pair | Plain F1 (P / R) | Weighted F1 | Weighted − plain (95% CI) | All-15-pairs model (2026-09-29) |
|---|---|---:|---|---:|
| Korean | **0.813** (0.790 / 0.837) | 0.814 | +0.001 [−0.002, +0.003] | 0.808 |
| Japanese | **0.741** (0.713 / 0.771) | 0.742 | +0.000 [−0.003, +0.004] | 0.736 |
| Chinese | **0.702** (0.658 / 0.752) | 0.702 | −0.000 [−0.003, +0.003] | 0.699 |
| Arabic | **0.669** (0.647 / 0.692) | 0.668 | −0.001 [−0.004, +0.003] | 0.664 |
| Russian | **0.666** (0.656 / 0.675) | 0.665 | −0.000 [−0.004, +0.003] | 0.659 |
| Cantonese | **0.625** (0.647 / 0.605) | 0.625 | +0.000 [−0.001, +0.001] | 0.619 |
| **Pooled** | **0.645** (0.657 / 0.634) | 0.645 | +0.000 [−0.001, +0.001] | — |
| **Macro over 6** | **0.703** | 0.703 | | 0.698 |

- **By direction** (both runs): into L2 0.499, back into L1 0.782.
- **Weighting has no effect on switch prediction.** The difference is +0.000 pooled, and every per-pair interval contains 0. That is expected: the duration loss is weighted 0.3 and only applies at switch points, so it barely moves the shared encoder.
- **Training on only these 6 pairs is about the same as training on all 15**, at +0.003 to +0.007 per pair (macro 0.703 vs 0.698). That is within the earlier run-to-run spread (±0.004), so the other 9 pairs neither help nor hurt these 6.
- **The 9 held-out pairs** (the Latin-script pairs, including German→French) reach F1 0.514 (macro 0.520) with no training on them, against 0.659 (macro 0.662) for the all-pairs model that trained on them. A model trained only on non-Latin scripts keeps about 78% of its switch F1 on Latin-script text.

## 2. Duration: weighting helps the raw decision, and val tuning closes the gap

Pooled over the 6 pairs (n = 125,675):

| | Accuracy | Macro-F1 | F1 Small / Medium / Large | End-to-end accuracy |
|---|---:|---:|---|---:|
| Plain, argmax | **0.686** | 0.581 | 0.608 / 0.313 / **0.822** | **0.461** |
| Weighted, argmax | 0.649 | 0.600 | 0.601 / 0.414 / 0.785 | 0.434 |
| Weighted − plain (95% CI) | | **+0.019** [+0.017, +0.022] | Medium **+0.101** [+0.095, +0.106] | −0.027 [−0.029, −0.025] |
| Plain, **val-tuned** | 0.663 | **0.602** | 0.599 / 0.406 / 0.802 | 0.445 |
| Weighted, val-tuned | 0.662 | **0.602** | 0.597 / 0.409 / 0.799 | 0.444 |

The tuned offsets (Small, Medium, Large): plain [0, +0.09, −0.59], weighted [0, +0.04, +0.21]. Tuning pushes each model back toward the same decision boundaries from opposite sides.

Per pair, macro-F1:

| Pair | Duration points | Plain, argmax | Weighted, argmax | Plain, val-tuned | Weighted, val-tuned | Medium F1, tuned (plain / weighted) | End-to-end accuracy, tuned |
|---|---:|---:|---:|---:|---:|---|---:|
| Japanese | 5,377 | 0.579 | 0.619 | **0.623** | 0.617 | 0.321 / 0.314 | 0.620 |
| Chinese | 5,623 | 0.571 | 0.607 | **0.607** | 0.605 | 0.401 / 0.397 | 0.571 |
| Cantonese | 100,834 | 0.580 | 0.595 | **0.598** | 0.598 | 0.414 / 0.417 | 0.411 |
| Korean | 4,916 | 0.561 | 0.601 | 0.597 | **0.599** | 0.272 / 0.284 | 0.655 |
| Arabic | 4,562 | 0.570 | 0.604 | 0.596 | **0.599** | 0.370 / 0.379 | 0.528 |
| Russian | 4,363 | 0.549 | 0.585 | 0.593 | 0.586 | 0.351 / 0.347 | 0.512 |

The training-free bars for the same points are piece lookup macro-F1 0.396–0.479 and most-common-class 0.225–0.271 ([baselines](../../data/baseline_anticipation_subword_6pairs.json)).

**Reading.**
- **On the raw argmax, weighting is a real duration gain on every pair:** macro-F1 +0.015 to +0.040, Medium F1 +0.10 to +0.15, all intervals excluding 0. The price is accuracy (−0.037) and end-to-end accuracy (−0.027), because the weighted model stops defaulting to Large, the commonest and easiest class.
- **After val tuning the two models are the same** (0.602 vs 0.602 pooled; per pair within ±0.007). This matches the local ablation ([2026-10-11](2026-10-11_duration_decision.md)): weighting and tuning fix the same lean toward Large. One of them is needed; both add nothing.
- **Medium is still the hard class,** at 0.27–0.42 F1. It is weakest on Korean (0.27–0.28) and Japanese (0.31–0.32), where Medium is rarest (12.5% and 13.5% of switches), and best on Cantonese (0.41–0.42).
- **End to end** (switch predicted *and* class right), the model gets 41–66% of true switches fully right per pair. It is lowest on Cantonese, the pair with the weakest switch recall.
- **Compare duration within a script only.** CJK runs count characters (or merged pieces); Russian and Arabic count words.

## Decision

- **Report:** the plain XLM-R with val-tuned duration as the 6-pair headline. Switch macro-F1 0.703; duration macro-F1 0.602, Medium 0.406.
- **Weighting:** class weighting is equivalent to tuning, so either is fine. Keep `DUR_WEIGHTS="none"` plus tuning as the default: it keeps the raw model's higher accuracy and lets the decision be re-tuned without retraining.
- **Next for Medium:** the run-length signal is already used, so the remaining gap is information the prefix doesn't have. More training data on the rarest Medium pairs, or the censored-run extension of the ordinal head, are the levers left. A larger loss weight is not one: tuning already sets the boundary.
