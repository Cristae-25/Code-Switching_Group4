# Chinese->English

## All models (test split)

Switch F1 is on each model's own pieces (tokenizers differ). Shared F1 is on the same words for every model, the fair cross-model number.

| Model | Switch F1 | P | R | Test positions | Shared F1 |
|---|---:|---:|---:|---:|---:|
| hazard, pooled (words) | 0.221 | 0.163 | 0.345 | 81,504 | 0.269 |
| hazard, per pair (words) | 0.225 | 0.179 | 0.304 | 81,504 | 0.284 |
| hazard, pooled (pieces) | 0.179 | 0.122 | 0.331 | 62,016 | 0.260 |
| hazard, per pair (pieces) | 0.242 | 0.171 | 0.413 | 62,016 | 0.189 |
| hazard, per pair (RoBERTa pieces) | 0.199 | 0.152 | 0.289 | 152,713 | 0.270 |
| hazard, per pair (mBERT pieces) | 0.142 | 0.141 | 0.143 | 86,007 | 0.235 |
| scratch (word) | 0.448 | 0.475 | 0.424 | 81,504 | 0.608 |
| scratch (subword) | 0.435 | 0.405 | 0.469 | 62,016 | 0.577 |
| xlmr (subword) | 0.699 | 0.647 | 0.761 | 62,016 |  |
| xlmr (subword, trained on 10 pairs) | 0.697 | 0.650 | 0.752 | 62,016 |  |
| mbert (subword) | 0.676 | 0.624 | 0.737 | 86,007 | 0.773 |
| scratch (subword, trained on 10 pairs) | 0.423 | 0.383 | 0.474 | 62,016 |  |

## Owner's notes

_Findings from the per-pair checklist: EDA, label spot-check, error types, the experiment. Regenerate the tables above with `scripts/analysis/pair_report.py --pair "Chinese->English" --preds <preds.npz> --md docs/pairs/Chinese-English.md`, then paste your notes back below._
