# German->French

## All models (test split)

Switch F1 is on each model's own pieces (tokenizers differ). Shared F1 is on the same words for every model, the fair cross-model number.

| Model | Switch F1 | P | R | Test positions | Shared F1 |
|---|---:|---:|---:|---:|---:|
| hazard, pooled (words) | 0.198 | 0.151 | 0.285 | 39,912 | 0.198 |
| hazard, per pair (words) | 0.280 | 0.194 | 0.504 | 39,912 | 0.280 |
| hazard, pooled (pieces) | 0.150 | 0.102 | 0.281 | 52,046 | 0.190 |
| hazard, per pair (pieces) | 0.223 | 0.156 | 0.391 | 52,046 | 0.272 |
| hazard, per pair (RoBERTa pieces) | 0.000 | 0.000 | 0.000 | 90,948 | 0.000 |
| hazard, per pair (mBERT pieces) | 0.202 | 0.140 | 0.363 | 57,318 | 0.267 |
| scratch (word) | 0.433 | 0.419 | 0.448 | 39,912 | 0.433 |
| scratch (subword) | 0.368 | 0.371 | 0.366 | 52,046 | 0.403 |
| xlmr (subword) | 0.658 | 0.647 | 0.669 | 52,046 |  |
| xlmr (subword, trained on 10 pairs, held-out pairs) | 0.450 | 0.497 | 0.411 | 52,046 |  |
| mbert (subword) | 0.663 | 0.645 | 0.683 | 57,318 | 0.673 |
| scratch (subword, trained on 10 pairs, held-out pairs) | 0.194 | 0.176 | 0.215 | 52,046 |  |

## Owner's notes

_Findings from the per-pair checklist (docs/TEAM_PLAN.md): EDA, label spot-check, error types, the experiment. Regenerate the tables above with `scripts/pair_report.py --pair "German->French" --preds <preds.npz> --md docs/pairs/German-French.md`, then paste your notes back below._
