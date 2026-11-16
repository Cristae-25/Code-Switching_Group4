# French->English

## All models (test split)

Switch F1 is on each model's own pieces (tokenizers differ). Shared F1 is on the same words for every model, the fair cross-model number.

| Model | Switch F1 | P | R | Test positions | Shared F1 |
|---|---:|---:|---:|---:|---:|
| hazard, pooled (words) | 0.233 | 0.182 | 0.326 | 41,448 | 0.233 |
| hazard, per pair (words) | 0.302 | 0.204 | 0.585 | 41,448 | 0.302 |
| hazard, pooled (pieces) | 0.195 | 0.141 | 0.314 | 54,022 | 0.223 |
| hazard, per pair (pieces) | 0.208 | 0.160 | 0.299 | 54,022 | 0.238 |
| hazard, per pair (RoBERTa pieces) | 0.225 | 0.185 | 0.286 | 69,334 | 0.235 |
| hazard, per pair (mBERT pieces) | 0.195 | 0.150 | 0.281 | 55,603 | 0.224 |
| scratch (word) | 0.518 | 0.470 | 0.576 | 41,448 | 0.518 |
| scratch (subword) | 0.458 | 0.432 | 0.488 | 54,022 | 0.494 |
| xlmr (subword) | 0.695 | 0.644 | 0.755 | 54,022 |  |
| xlmr (subword, trained on 10 pairs) | 0.694 | 0.654 | 0.740 | 54,022 |  |
| mbert (subword) | 0.689 | 0.638 | 0.748 | 55,603 | 0.696 |
| scratch (subword, trained on 10 pairs) | 0.430 | 0.394 | 0.474 | 54,022 |  |

## Owner's notes

_Findings from the per-pair checklist: EDA, label spot-check, error types, the experiment. Regenerate the tables above with `scripts/analysis/pair_report.py --pair "French->English" --preds <preds.npz> --md docs/pairs/French-English.md`, then paste your notes back below._
