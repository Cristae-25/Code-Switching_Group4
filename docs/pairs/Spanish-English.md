# Spanish->English

## All models (test split)

Switch F1 is on each model's own pieces (tokenizers differ). Shared F1 is on the same words for every model, the fair cross-model number.

| Model | Switch F1 | P | R | Test positions | Shared F1 |
|---|---:|---:|---:|---:|---:|
| hazard, pooled (words) | 0.232 | 0.183 | 0.317 | 45,175 | 0.232 |
| hazard, per pair (words) | 0.278 | 0.185 | 0.561 | 45,175 | 0.278 |
| hazard, pooled (pieces) | 0.191 | 0.139 | 0.305 | 54,699 | 0.222 |
| hazard, per pair (pieces) | 0.239 | 0.146 | 0.654 | 54,699 | 0.281 |
| hazard, per pair (RoBERTa pieces) | 0.212 | 0.175 | 0.268 | 76,905 | 0.225 |
| hazard, per pair (mBERT pieces) | 0.223 | 0.137 | 0.603 | 57,481 | 0.270 |
| scratch (word) | 0.489 | 0.452 | 0.534 | 45,175 | 0.489 |
| scratch (subword) | 0.425 | 0.396 | 0.458 | 54,699 | 0.457 |
| xlmr (subword) | 0.673 | 0.626 | 0.727 | 54,699 |  |
| xlmr (subword, trained on 10 pairs) | 0.669 | 0.632 | 0.710 | 54,699 |  |
| mbert (subword) | 0.660 | 0.610 | 0.719 | 57,481 | 0.666 |
| scratch (subword, trained on 10 pairs) | 0.422 | 0.400 | 0.446 | 54,699 |  |

## Owner's notes

_Findings from the per-pair checklist: EDA, label spot-check, error types, the experiment. Regenerate the tables above with `scripts/analysis/pair_report.py --pair "Spanish->English" --preds <preds.npz> --md docs/pairs/Spanish-English.md`, then paste your notes back below._
