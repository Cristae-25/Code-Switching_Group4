# 2026-11-02: Previous-word language and pair embedding on XLM-R (Kaggle T4)

**Runs:** `kaggle/train_pretrained.ipynb`. Both use seed 0, 12,000 steps (1.02 epochs), batch 16, window 384 / stride 288, lr 2e-5 for the body and 1e-3 for the heads, all 15 pairs, and one flag each. Both causality checks passed.

| TAG | Flag | Train time |
|---|---|---:|
| `xlmr_prev` | `--prev-lang`: embedding of the last finished word's language | 71 min |
| `xlmr_pair` | `--pair-emb`: embedding of the row's pair (15 pairs) | 80 min |

There is no seed-0 no-flag baseline with saved predictions. The reference is the unseeded 2026-09-29 `xlmr` run, which has the same settings and no flags.

## Results (test)

| Run | Switch F1 | Macro-F1 (pairs) | Shared F1 | Duration macro-F1 | τ |
|---|---:|---:|---:|---:|---:|
| `xlmr` (09-29, no flags) | 0.6459 | 0.6760 | n/a | 0.575 | 0.337 |
| `xlmr_prev` | 0.6461 | 0.6765 | 0.7259 | 0.574 | 0.332 |
| `xlmr_pair` | 0.6460 | 0.6766 | 0.7266 | 0.573 | 0.371 |

Paired row bootstrap, `xlmr_pair − xlmr_prev` (`scripts/bootstrap_preds.py xlmr_prev xlmr_pair`):
- **switch F1:** −0.000 [−0.001, +0.001]
- **duration macro-F1:** −0.001 [−0.002, +0.001]

Per pair, every switch-F1 difference is within ±0.006. The end-to-end duration accuracy is 0.017 lower for `pair` in every pair. That is a threshold artifact: its higher τ (0.371) fires fewer switches.

## Reading

- **Neither input helps XLM-R.** Both runs are within 0.0003 of the no-flag run, far inside the ±0.005 seed noise. The from-scratch ablation's +0.009 for previous-word language does not carry over.
- **Two explanations, not yet separated:**
  1. **Redundant:** XLM-R already reads each word's language from its text (script, spelling), so a tag adds nothing.
  2. **Under-trained:** the new embeddings start at zero and train at the body's lr (2e-5, one-cycle). Adam moves each dimension at most about lr per step, so at most about 0.12 over the run and usually much less. XLM-R's token embeddings have a per-dimension std of 0.21, so the added vectors may have stayed too small to matter. The from-scratch model trained them at 3e-4.
- **Separating them:** read `lang_emb.weight` / `pair_emb.weight` norms from `models/anticipation_xlmr_{prev,pair}.pt` and compare them with the token-embedding row norm (5.9). If they are tiny, retry with the new embeddings at the head lr. If they are comparable, the inputs are redundant; drop them. Under LoRA (`xlmr_lora16`) they train at 2e-4, which tests explanation 2 indirectly.

## Breakdowns on 15 pairs (`xlmr_prev`)

- **Punctuation** (`scripts/punct_breakdown.py xlmr_prev`): the same as on the 6 pairs. Mid-clause F1 0.467, and 37.4 k of about 62.8 k misses. Comma 0.667, sentence end 0.840, turn 0.869. Latin pairs are weakest mid-clause (German→French 0.17, French→English 0.20). Hindi (0.66) and Philippines (0.61) are strongest, because most of their switches are mid-clause.
- **Grammar** (`scripts/grammar_breakdown.py xlmr_prev`): the same conclusion. In every group the model's mean probability tracks the base rate, or sits above it. 82% of mid-clause misses are switches out of L1 (F1 0.31). In the European Latin pairs the model *over*-predicts after mid-frequency content words (e.g. Spanish 0.040 actual vs 0.107 predicted), so function words are not a missed cue there. The only under-predicted groups are rare words in Malay (0.43 vs 0.23) and Vietnamese (0.41 vs 0.24). They are about 550 misses, about 1% of all, and plausibly tagger noise (an English word tagged L1).
