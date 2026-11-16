# Malay->English

## All models (test split)

Switch F1 is on each model's own pieces (tokenizers differ). Shared F1 is on the same words for every model, the fair cross-model number.

| Model | Switch F1 | P | R | Test positions | Shared F1 |
|---|---:|---:|---:|---:|---:|
| hazard, pooled (words) | 0.277 | 0.232 | 0.344 | 44,558 | 0.277 |
| hazard, per pair (words) | 0.328 | 0.237 | 0.532 | 44,558 | 0.328 |
| hazard, pooled (pieces) | 0.241 | 0.185 | 0.346 | 51,673 | 0.275 |
| hazard, per pair (pieces) | 0.281 | 0.184 | 0.592 | 51,673 | 0.321 |
| hazard, per pair (RoBERTa pieces) | 0.232 | 0.180 | 0.326 | 84,583 | 0.265 |
| hazard, per pair (mBERT pieces) | 0.254 | 0.165 | 0.548 | 60,428 | 0.314 |
| scratch (word) | 0.459 | 0.399 | 0.541 | 44,558 | 0.459 |
| scratch (subword) | 0.397 | 0.336 | 0.484 | 51,673 | 0.419 |
| xlmr (subword) | 0.548 | 0.530 | 0.567 | 51,673 |  |
| xlmr (subword, trained on 10 pairs) | 0.545 | 0.542 | 0.548 | 51,673 |  |
| mbert (subword) | 0.531 | 0.520 | 0.542 | 60,428 | 0.537 |
| scratch (subword, trained on 10 pairs) | 0.396 | 0.335 | 0.485 | 51,673 |  |

## Owner's notes

_Findings from the per-pair checklist: EDA, label spot-check, error types, the experiment. Regenerate the tables above with `scripts/analysis/pair_report.py --pair "Malay->English" --preds <preds.npz> --md docs/pairs/Malay-English.md`, then paste your notes back below._
