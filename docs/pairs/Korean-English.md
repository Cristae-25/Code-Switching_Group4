# Korean->English

## All models (test split)

Switch F1 is on each model's own pieces (tokenizers differ). Shared F1 is on the same words for every model, the fair cross-model number.

| Model | Switch F1 | P | R | Test positions | Shared F1 |
|---|---:|---:|---:|---:|---:|
| hazard, pooled (words) | 0.247 | 0.192 | 0.348 | 72,637 | 0.311 |
| hazard, per pair (words) | 0.252 | 0.213 | 0.309 | 72,637 | 0.329 |
| hazard, pooled (pieces) | 0.207 | 0.149 | 0.339 | 55,454 | 0.305 |
| hazard, per pair (pieces) | 0.265 | 0.165 | 0.677 | 55,454 | 0.318 |
| hazard, per pair (RoBERTa pieces) | 0.232 | 0.189 | 0.301 | 176,994 | 0.322 |
| hazard, per pair (mBERT pieces) | 0.208 | 0.158 | 0.303 | 65,761 | 0.313 |
| scratch (word) | 0.636 | 0.566 | 0.727 | 72,637 | 0.663 |
| scratch (subword) | 0.626 | 0.611 | 0.642 | 55,454 | 0.584 |
| xlmr (subword) | 0.808 | 0.775 | 0.844 | 55,454 |  |
| xlmr (subword, trained on 10 pairs, held-out pairs) | 0.775 | 0.768 | 0.782 | 55,454 |  |
| mbert (subword) | 0.793 | 0.751 | 0.841 | 65,761 | 0.803 |
| scratch (subword, trained on 10 pairs, held-out pairs) | 0.318 | 0.272 | 0.383 | 55,454 |  |

## Owner's notes

_Findings from the per-pair checklist (docs/TEAM_PLAN.md): EDA, label spot-check, error types, the experiment. Regenerate the tables above with `scripts/pair_report.py --pair "Korean->English" --preds <preds.npz> --md docs/pairs/Korean-English.md`, then paste your notes back below._
