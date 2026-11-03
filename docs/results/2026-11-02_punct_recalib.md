# 2026-11-02: Would deciding at the punctuation help? (local, no GPU)

**Why:** a switch label sits on the last language-bearing piece *before* any punctuation, so the model decides at `home` in `home, 然后` without having read the comma ([punctuation breakdown](2026-11-01_punct_breakdown.md)). Before spending a Kaggle run on moving the label, this measures a training-free lower bound on the gain.

**How:** `./.venv/bin/python scripts/punct_recalib.py xlmr_prev` (13 s).
- **Inputs:** the saved `xlmr_prev` logits, plus the kind of gap that follows each position (`adjacent` / `comma` / `sentence_end` / `other_gap` / `turn`, the buckets from `punct_breakdown.py`). The gap kind is exactly what a model deciding at the punctuation would also know.
- **Model:** a logistic recalibration `[1, logit, bucket one-hot, bucket × logit]`.
- **Fit and score:** fitted on **val**, τ re-chosen on val, scored on **test**.
- **Check:** the raw row reproduces the run json (pooled F1 0.6461).
- **Output:** `data/punct_recalib_xlmr_prev.json`.

## Results (test, 15 pairs)

| Variant | τ | Switch F1 | Macro-F1 | adjacent | comma | sentence_end | other_gap | turn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw `xlmr_prev` | 0.332 | 0.6461 | 0.6765 | 0.467 | 0.667 | 0.840 | 0.625 | 0.869 |
| + gap offset | 0.365 | 0.6772 | 0.7201 | 0.441 | 0.671 | 0.892 | 0.639 | 0.866 |
| + gap offset × logit | 0.365 | **0.6775** | **0.7195** | 0.443 | 0.670 | 0.889 | 0.639 | 0.869 |

- **Per pair:** 14 of 15 pairs gain, from +0.010 (Hindi) to +0.069 (French→English, Italian→English, Arabic→English). Philippines loses 0.015, because 66% of its switches are mid-clause.

## Reading

- **About +0.031 pooled and +0.043 macro.** That is roughly 6× the ±0.005 seed noise, and it comes from post-hoc information alone. A model trained to read the punctuation should reach at least this, so `--decide-at gap` is worth a run.
- **Most of the gain is at sentence ends** (0.840 → 0.89): knowing a `.` follows lifts the prior to 0.78.
- **Mid-clause F1 drops** (0.467 → 0.443). That is the shared τ moving up, not a worse model. Mid-clause (L1→L2 insertion starts) is still the bottleneck, and punctuation cannot help it.
- **Comparability:** part of the gain is the task getting easier (deciding after the comma), so report the `gap` run next to the word-decision runs, never in place of them.

## Next

`--decide-at gap` (notebook `DECIDE_AT = "gap"`, TAG suffix `_gap`). It moves every label to the last non-blank piece of the gap (`anticipation.gap_landing`), and never onto a bare `▁`, which would leak the next character's script. Positives, durations and scored-position counts are unchanged; on 3000 real rows 12.9% of labels move, mostly onto `,` `。` `.` `?`. Compare with `bootstrap_preds.py xlmr_prev xlmr_gap`, and with `punct_breakdown.py xlmr_gap`, which re-buckets the landing positions.
