# Japanese->English

## All models (test split)

Switch F1 is on each model's own pieces (tokenizers differ). Shared F1 is on the same words for every model, the fair cross-model number.

| Model | Switch F1 | P | R | Test positions | Shared F1 |
|---|---:|---:|---:|---:|---:|
| hazard, pooled (words) | 0.232 | 0.168 | 0.376 | 94,997 | 0.281 |
| hazard, per pair (words) | 0.244 | 0.191 | 0.337 | 94,997 | 0.306 |
| hazard, pooled (pieces) | 0.190 | 0.129 | 0.365 | 64,510 | 0.277 |
| hazard, per pair (pieces) | 0.243 | 0.159 | 0.516 | 64,510 | 0.273 |
| hazard, per pair (RoBERTa pieces) | 0.218 | 0.163 | 0.329 | 131,811 | 0.296 |
| hazard, per pair (mBERT pieces) | 0.182 | 0.141 | 0.258 | 84,141 | 0.267 |
| scratch (word) | 0.537 | 0.495 | 0.586 | 94,997 | 0.616 |
| scratch (subword) | 0.522 | 0.506 | 0.538 | 64,510 | 0.560 |
| xlmr (subword) | 0.736 | 0.694 | 0.783 | 64,510 |  |
| xlmr (subword, trained on 10 pairs) | 0.737 | 0.698 | 0.781 | 64,510 |  |
| mbert (subword) | 0.700 | 0.650 | 0.759 | 84,141 | 0.770 |
| scratch (subword, trained on 10 pairs) | 0.516 | 0.465 | 0.580 | 64,510 |  |

## Owner's notes

_Findings from the per-pair checklist (docs/TEAM_PLAN.md): EDA, label spot-check, error types, the experiment. Regenerate the tables above with `scripts/pair_report.py --pair "Japanese->English" --preds <preds.npz> --md docs/pairs/Japanese-English.md`, then paste your notes back below._
