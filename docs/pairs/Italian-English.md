# Italian->English

## All models (test split)

Switch F1 is on each model's own pieces (tokenizers differ). Shared F1 is on the same words for every model, the fair cross-model number.

| Model | Switch F1 | P | R | Test positions | Shared F1 |
|---|---:|---:|---:|---:|---:|
| hazard, pooled (words) | 0.227 | 0.177 | 0.317 | 41,459 | 0.227 |
| hazard, per pair (words) | 0.285 | 0.192 | 0.547 | 41,459 | 0.285 |
| hazard, pooled (pieces) | 0.192 | 0.140 | 0.308 | 51,977 | 0.220 |
| hazard, per pair (pieces) | 0.242 | 0.148 | 0.680 | 51,977 | 0.292 |
| hazard, per pair (RoBERTa pieces) | 0.214 | 0.176 | 0.272 | 73,665 | 0.223 |
| hazard, per pair (mBERT pieces) | 0.193 | 0.149 | 0.276 | 55,900 | 0.221 |
| scratch (word) | 0.467 | 0.443 | 0.494 | 41,459 | 0.467 |
| scratch (subword) | 0.414 | 0.434 | 0.397 | 51,977 | 0.446 |
| xlmr (subword) | 0.671 | 0.632 | 0.716 | 51,977 |  |
| xlmr (subword, trained on 10 pairs, held-out pairs) | 0.661 | 0.638 | 0.686 | 51,977 |  |
| mbert (subword) | 0.658 | 0.616 | 0.706 | 55,900 | 0.664 |
| scratch (subword, trained on 10 pairs, held-out pairs) | 0.366 | 0.330 | 0.411 | 51,977 |  |

## Owner's notes

_Findings from the per-pair checklist (docs/TEAM_PLAN.md): EDA, label spot-check, error types, the experiment. Regenerate the tables above with `scripts/pair_report.py --pair "Italian->English" --preds <preds.npz> --md docs/pairs/Italian-English.md`, then paste your notes back below._
