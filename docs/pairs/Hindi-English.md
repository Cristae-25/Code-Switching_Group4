# Hindi->English

## All models (test split)

Switch F1 is on each model's own pieces (tokenizers differ). Shared F1 is on the same words for every model, the fair cross-model number.

| Model | Switch F1 | P | R | Test positions | Shared F1 |
|---|---:|---:|---:|---:|---:|
| hazard, pooled (words) | 0.381 | 0.363 | 0.400 | 45,754 | 0.381 |
| hazard, per pair (words) | 0.442 | 0.289 | 0.937 | 45,754 | 0.442 |
| hazard, pooled (pieces) | 0.329 | 0.283 | 0.393 | 57,063 | 0.373 |
| hazard, per pair (pieces) | 0.362 | 0.224 | 0.948 | 57,063 | 0.434 |
| hazard, per pair (RoBERTa pieces) | 0.355 | 0.284 | 0.473 | 124,399 | 0.401 |
| hazard, per pair (mBERT pieces) | 0.307 | 0.184 | 0.947 | 68,010 | 0.433 |
| scratch (word) | 0.616 | 0.503 | 0.795 | 45,754 | 0.616 |
| scratch (subword) | 0.544 | 0.421 | 0.769 | 57,063 | 0.586 |
| xlmr (subword) | 0.715 | 0.639 | 0.811 | 57,063 |  |
| xlmr (subword, trained on 10 pairs, held-out pairs) | 0.595 | 0.634 | 0.561 | 57,063 |  |
| mbert (subword) | 0.706 | 0.622 | 0.817 | 68,010 | 0.718 |
| scratch (subword, trained on 10 pairs, held-out pairs) | 0.372 | 0.400 | 0.347 | 57,063 |  |

## Owner's notes

_Findings from the per-pair checklist: EDA, label spot-check, error types, the experiment. Regenerate the tables above with `scripts/analysis/pair_report.py --pair "Hindi->English" --preds <preds.npz> --md docs/pairs/Hindi-English.md`, then paste your notes back below._
