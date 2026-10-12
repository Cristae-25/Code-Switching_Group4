# 2026-10-11: Duration decision tuning and an ordered (ordinal) duration head

**Question:** Medium (3–6 pieces) is the weak duration class, and the model hedges toward Large. Two fixes were tried on the cheap local model before spending Kaggle hours on XLM-R:

1. **Tune the decision on val, after training.** Add one log-offset per class to the duration head's probabilities, and pick the offsets that maximise macro-F1 on val. Apply them unchanged to test. No retraining (`src/metrics.tune_class_offsets`).
2. **Treat duration as ordered.** A cumulative-link head (`--dur-head ordinal`, `src/causal_model.OrdinalHead`) predicts P(> Small) and P(> Medium) from one shared projection with two ordered cut points, in place of a 3-way softmax.

**Setup:**
- **Model and data:** the same as the [2026-09-30 ablations](2026-09-30_ablations.md): the from-scratch causal transformer on XLM-R pieces, all 15 pairs.
- **Run:** 6,000 steps, `--seed 0`, on an M3.
- **Evaluation:** the threshold is tuned on val and test is scored once. Duration is scored at true switch points whose class is certain (n = 180,990).
- **Earlier runs:** their offsets were fit offline from the saved predictions with `scripts/tune_duration.py`. New runs fit them during training and write `duration_tuned` into the results JSON.

## Results (test split)

| Duration head | Class weights | Macro-F1, argmax | **Macro-F1, val-tuned** | F1 Small / Medium / Large (tuned) | Within one class (tuned) | Small↔Large errors (tuned) | Shared switch F1 |
|---|---|---:|---:|---|---:|---:|---:|
| softmax | none | 0.481 | **0.548** | 0.558 / 0.334 / 0.752 | 0.909 | 9.1% | 0.547 |
| softmax | balanced | 0.540 | 0.547 | 0.557 / 0.336 / 0.747 | 0.908 | 9.2% | 0.547 |
| ordinal | none | 0.458 | 0.536 | 0.560 / 0.306 / 0.742 | **0.935** | **6.5%** | 0.542 |
| ordinal | balanced | 0.496 | 0.536 | 0.563 / 0.305 / 0.740 | 0.932 | 6.8% | 0.548 |
| softmax + `--prev-lang` + `--predict-at word_end` | balanced | 0.554 | 0.558 | 0.555 / 0.362 / 0.757 | — | — | 0.570 |

The tuned offsets for the plain softmax were Small 0, Medium +0.33, Large −0.45: a push away from Large, toward Medium.

## Reading

- **Val tuning is the cheapest fix and the best one.** It lifts the unweighted softmax from 0.481 to **0.548** macro-F1 (Medium 0.075 → 0.334) with no retraining, matching class weighting (0.540). On top of weighting it adds little (+0.007, +0.004 on the combined flags), because both correct the same lean toward Large. Every run now reports it as `duration_tuned`.
- **The ordinal head does not win on macro-F1.** After tuning it is 0.012 below the softmax (0.536 vs 0.548), and Medium F1 is lower (0.306 vs 0.334). Untuned, it predicts no Medium at all (F1 0.000): with two cuts close together, the Medium interval is narrow at the default boundaries.
- **What it does buy is fewer far-off errors.** Small-for-Large and Large-for-Small mistakes fall from 9.1% to 6.5% of switch points (−29%), and within-one-class accuracy rises from 0.909 to 0.935. That is the ordering doing its job: errors land in the neighbouring class.
- **Switch prediction is unaffected** by either change (shared F1 0.542–0.548, inside the ±0.005 run-to-run noise).

## Decision

- **Adopt val-tuned offsets** as the duration number we report alongside the raw argmax. It costs nothing and is on by default (`duration_tuned` in every results JSON).
- **Don't spend GPU on `xlmr_ord_6pairs` for now.** Macro-F1 is the project's duration metric, and the ordinal head loses on it. Revisit if the write-up starts to care about error severity (within-one accuracy or Small↔Large confusions), where ordinal is clearly better.
- **Not tried, and the natural ordinal follow-up:** training on right-censored runs. A run cut off by the end of the text at 3–6 pieces is known to be > Small, so the first cut could learn from it, while its class stays unscored. That would add training signal only the ordinal head can use.
