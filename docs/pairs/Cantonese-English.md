# Cantonese->English

## All models (test split)

Switch F1 is on each model's own pieces (tokenizers differ). Shared F1 is on the same words for every model, the fair cross-model number.

| Model | Switch F1 | P | R | Test positions | Shared F1 |
|---|---:|---:|---:|---:|---:|
| hazard, pooled (words) | 0.327 | 0.303 | 0.356 | 1,224,548 | 0.434 |
| hazard, per pair (words) | 0.328 | 0.290 | 0.378 | 1,224,548 | 0.427 |
| hazard, pooled (pieces) | 0.249 | 0.198 | 0.336 | 1,061,851 | 0.414 |
| hazard, per pair (pieces) | 0.252 | 0.186 | 0.389 | 1,061,851 | 0.400 |
| hazard, per pair (RoBERTa pieces) | 0.277 | 0.206 | 0.424 | 2,559,549 | 0.384 |
| hazard, per pair (mBERT pieces) | 0.265 | 0.203 | 0.382 | 1,289,718 | 0.406 |
| scratch (word) | 0.498 | 0.494 | 0.502 | 1,224,548 | 0.675 |
| scratch (subword) | 0.439 | 0.399 | 0.488 | 1,061,851 | 0.622 |
| xlmr (subword) | 0.619 | 0.621 | 0.618 | 1,061,851 |  |
| xlmr (subword, trained on 10 pairs) | 0.623 | 0.636 | 0.610 | 1,061,851 |  |
| mbert (subword) | 0.609 | 0.607 | 0.612 | 1,289,718 | 0.791 |
| scratch (subword, trained on 10 pairs) | 0.449 | 0.405 | 0.505 | 1,061,851 |  |

## Owner's notes

_Findings from the per-pair checklist (docs/TEAM_PLAN.md): EDA, label spot-check, error types, the experiment. Regenerate the tables above with `scripts/pair_report.py --pair "Cantonese->English" --preds <preds.npz> --md docs/pairs/Cantonese-English.md`, then paste your notes back below._
