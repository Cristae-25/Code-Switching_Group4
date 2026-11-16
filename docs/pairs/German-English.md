# German->English

## All models (test split)

Switch F1 is on each model's own pieces (tokenizers differ). Shared F1 is on the same words for every model, the fair cross-model number.

| Model | Switch F1 | P | R | Test positions | Shared F1 |
|---|---:|---:|---:|---:|---:|
| hazard, pooled (words) | 0.248 | 0.198 | 0.330 | 32,290 | 0.248 |
| hazard, per pair (words) | 0.318 | 0.211 | 0.639 | 32,290 | 0.318 |
| hazard, pooled (pieces) | 0.210 | 0.156 | 0.321 | 39,836 | 0.241 |
| hazard, per pair (pieces) | 0.268 | 0.170 | 0.638 | 39,836 | 0.322 |
| hazard, per pair (RoBERTa pieces) | 0.236 | 0.191 | 0.309 | 61,810 | 0.248 |
| hazard, per pair (mBERT pieces) | 0.248 | 0.152 | 0.677 | 43,108 | 0.319 |
| scratch (word) | 0.516 | 0.488 | 0.549 | 32,290 | 0.516 |
| scratch (subword) | 0.464 | 0.454 | 0.474 | 39,836 | 0.500 |
| xlmr (subword) | 0.733 | 0.695 | 0.776 | 39,836 |  |
| xlmr (subword, trained on 10 pairs, held-out pairs) | 0.653 | 0.638 | 0.670 | 39,836 |  |
| mbert (subword) | 0.718 | 0.674 | 0.768 | 43,108 | 0.728 |
| scratch (subword, trained on 10 pairs, held-out pairs) | 0.359 | 0.315 | 0.417 | 39,836 |  |

## Owner's notes

_Findings from the per-pair checklist: EDA, label spot-check, error types, the experiment. Regenerate the tables above with `scripts/analysis/pair_report.py --pair "German->English" --preds <preds.npz> --md docs/pairs/German-English.md`, then paste your notes back below._
