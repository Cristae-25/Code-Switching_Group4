# Philippines->English

## All models (test split)

Switch F1 is on each model's own pieces (tokenizers differ). Shared F1 is on the same words for every model, the fair cross-model number.

| Model | Switch F1 | P | R | Test positions | Shared F1 |
|---|---:|---:|---:|---:|---:|
| hazard, pooled (words) | 0.313 | 0.274 | 0.366 | 46,130 | 0.313 |
| hazard, per pair (words) | 0.394 | 0.255 | 0.867 | 46,130 | 0.394 |
| hazard, pooled (pieces) | 0.264 | 0.207 | 0.365 | 58,024 | 0.310 |
| hazard, per pair (pieces) | 0.323 | 0.198 | 0.863 | 58,024 | 0.391 |
| hazard, per pair (RoBERTa pieces) | 0.290 | 0.215 | 0.443 | 71,957 | 0.342 |
| hazard, per pair (mBERT pieces) | 0.287 | 0.174 | 0.825 | 65,327 | 0.393 |
| scratch (word) | 0.548 | 0.439 | 0.728 | 46,130 | 0.548 |
| scratch (subword) | 0.494 | 0.387 | 0.684 | 58,024 | 0.528 |
| xlmr (subword) | 0.638 | 0.559 | 0.745 | 58,024 |  |
| xlmr (subword, trained on 10 pairs) | 0.639 | 0.576 | 0.718 | 58,024 |  |
| mbert (subword) | 0.627 | 0.551 | 0.727 | 65,327 | 0.637 |
| scratch (subword, trained on 10 pairs) | 0.498 | 0.391 | 0.686 | 58,024 |  |

## Owner's notes

_Findings from the per-pair checklist (docs/TEAM_PLAN.md): EDA, label spot-check, error types, the experiment. Regenerate the tables above with `scripts/pair_report.py --pair "Philippines->English" --preds <preds.npz> --md docs/pairs/Philippines-English.md`, then paste your notes back below._
