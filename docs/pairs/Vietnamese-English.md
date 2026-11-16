# Vietnamese->English

## All models (test split)

Switch F1 is on each model's own pieces (tokenizers differ). Shared F1 is on the same words for every model, the fair cross-model number.

| Model | Switch F1 | P | R | Test positions | Shared F1 |
|---|---:|---:|---:|---:|---:|
| hazard, pooled (words) | 0.267 | 0.219 | 0.343 | 55,973 | 0.267 |
| hazard, per pair (words) | 0.270 | 0.239 | 0.312 | 55,973 | 0.271 |
| hazard, pooled (pieces) | 0.222 | 0.167 | 0.332 | 60,766 | 0.258 |
| hazard, per pair (pieces) | 0.248 | 0.155 | 0.616 | 60,766 | 0.270 |
| hazard, per pair (RoBERTa pieces) | 0.232 | 0.194 | 0.289 | 177,772 | 0.254 |
| hazard, per pair (mBERT pieces) | 0.221 | 0.166 | 0.332 | 63,861 | 0.257 |
| scratch (word) | 0.453 | 0.469 | 0.439 | 55,973 | 0.453 |
| scratch (subword) | 0.403 | 0.422 | 0.385 | 60,766 | 0.413 |
| xlmr (subword) | 0.622 | 0.603 | 0.642 | 60,766 |  |
| xlmr (subword, trained on 10 pairs) | 0.623 | 0.620 | 0.626 | 60,766 |  |
| mbert (subword) | 0.608 | 0.592 | 0.626 | 63,861 | 0.613 |
| scratch (subword, trained on 10 pairs) | 0.410 | 0.401 | 0.419 | 60,766 |  |

## Owner's notes

_Findings from the per-pair checklist: EDA, label spot-check, error types, the experiment. Regenerate the tables above with `scripts/analysis/pair_report.py --pair "Vietnamese->English" --preds <preds.npz> --md docs/pairs/Vietnamese-English.md`, then paste your notes back below._
