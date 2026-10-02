# Russian->English

## All models (test split)

Switch F1 is on each model's own pieces (tokenizers differ). Shared F1 is on the same words for every model, the fair cross-model number.

| Model | Switch F1 | P | R | Test positions | Shared F1 |
|---|---:|---:|---:|---:|---:|
| hazard, pooled (words) | 0.236 | 0.185 | 0.325 | 37,302 | 0.236 |
| hazard, per pair (words) | 0.311 | 0.204 | 0.650 | 37,302 | 0.311 |
| hazard, pooled (pieces) | 0.199 | 0.145 | 0.317 | 49,679 | 0.228 |
| hazard, per pair (pieces) | 0.243 | 0.158 | 0.525 | 49,679 | 0.307 |
| hazard, per pair (RoBERTa pieces) | 0.220 | 0.181 | 0.281 | 177,548 | 0.231 |
| hazard, per pair (mBERT pieces) | 0.188 | 0.146 | 0.265 | 60,659 | 0.214 |
| scratch (word) | 0.470 | 0.428 | 0.521 | 37,302 | 0.470 |
| scratch (subword) | 0.372 | 0.390 | 0.356 | 49,679 | 0.408 |
| xlmr (subword) | 0.659 | 0.631 | 0.690 | 49,679 |  |
| xlmr (subword, trained on 10 pairs) | 0.667 | 0.649 | 0.685 | 49,679 |  |
| mbert (subword) | 0.645 | 0.616 | 0.676 | 60,659 | 0.653 |
| scratch (subword, trained on 10 pairs) | 0.395 | 0.393 | 0.398 | 49,679 |  |

## Owner's notes

_Findings from the per-pair checklist (docs/TEAM_PLAN.md): EDA, label spot-check, error types, the experiment. Regenerate the tables above with `scripts/pair_report.py --pair "Russian->English" --preds <preds.npz> --md docs/pairs/Russian-English.md`, then paste your notes back below._
