# Arabic->English

## All models (test split)

Switch F1 is on each model's own pieces (tokenizers differ). Shared F1 is on the same words for every model, the fair cross-model number.

| Model | Switch F1 | P | R | Test positions | Shared F1 |
|---|---:|---:|---:|---:|---:|
| hazard, pooled (words) | 0.239 | 0.185 | 0.337 | 37,052 | 0.239 |
| hazard, per pair (words) | 0.322 | 0.212 | 0.674 | 37,052 | 0.322 |
| hazard, pooled (pieces) | 0.197 | 0.142 | 0.325 | 51,395 | 0.229 |
| hazard, per pair (pieces) | 0.241 | 0.168 | 0.428 | 51,395 | 0.311 |
| hazard, per pair (RoBERTa pieces) | 0.224 | 0.183 | 0.290 | 140,221 | 0.235 |
| hazard, per pair (mBERT pieces) | 0.176 | 0.154 | 0.205 | 64,030 | 0.197 |
| scratch (word) | 0.482 | 0.428 | 0.551 | 37,052 | 0.482 |
| scratch (subword) | 0.400 | 0.408 | 0.392 | 51,395 | 0.431 |
| xlmr (subword) | 0.664 | 0.617 | 0.717 | 51,395 |  |
| xlmr (subword, trained on 10 pairs) | 0.666 | 0.638 | 0.698 | 51,395 |  |
| mbert (subword) | 0.649 | 0.606 | 0.698 | 64,030 | 0.663 |
| scratch (subword, trained on 10 pairs) | 0.406 | 0.383 | 0.431 | 51,395 |  |

## Owner's notes

_Findings from the per-pair checklist: EDA, label spot-check, error types, the experiment. Regenerate the tables above with `scripts/analysis/pair_report.py --pair "Arabic->English" --preds <preds.npz> --md docs/pairs/Arabic-English.md`, then paste your notes back below._
