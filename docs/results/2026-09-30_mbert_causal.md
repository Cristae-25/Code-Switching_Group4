# 2026-09-30: mBERT with a causal mask, compared with XLM-R

**Runs:** `kaggle/train_pretrained.ipynb` with `MODEL = "mbert"`, `RUN_FULL = True` and `ZERO_SHOT = True`, on a Kaggle T4.
- **Model:** `bert-base-multilingual-cased` with `is_decoder=True` (causal attention), 177 M parameters, plus a switch head and a 3-class duration head. It reads pieces only, with no language input.
- **Causality check:** 0.00 drift before both runs.
- **Settings:** identical to the XLM-R runs: 12,000 steps, batch 16, window 384 / stride 288, learning rate 2e-05 for the body and 0.001 for the heads.
- **Files:** [`2026-09-30_mbert_causal.json`](2026-09-30_mbert_causal.json) (all 15 pairs) and [`2026-09-30_mbert_causal_zeroshot.json`](2026-09-30_mbert_causal_zeroshot.json) (Korean, Italian, Hindi and German held out).
- **XLM-R comparison:** the [all-pairs](2026-09-29_xlmr_causal.md) and [zero-shot](2026-09-29_xlmr_causal_zeroshot.md) records.

**RoBERTa (English-only) was not run.** The team decided the monolingual control was not needed; the code supports it (`MODEL = "roberta"`).

## What is comparable, and how

The two models use different tokenizers, so they are scored on different sets of test positions.

| Model | Test positions | CJK text split into |
|---|---:|---|
| XLM-R | 1,825,011 | multi-character pieces |
| mBERT | 2,177,352 | single characters |

The **shared positions**, the last piece of each non-CJK word, are the same about 747 K test words for every tokenizer. mBERT reports them, but the XLM-R runs predate that report, so **there is no exact same-positions XLM-R vs mBERT number yet.** The comparisons below are:
1. **XLM-R vs mBERT per pair,** each on its own pieces. The switch rate per unit differs (10.4% vs 8.7%), so each model is also shown against the per-pair hazard fitted on *its own* pieces. On non-CJK pairs the two tokenizers split words similarly, so the raw F1 is close to comparable there; on CJK it is not.
2. **mBERT vs the from-scratch models on the shared positions:** identical words, exactly comparable.

## Switch prediction, all 15 pairs (test split)

| Model (own pieces) | Switch F1 (P / R) | Macro-F1 | F1 into L2 | F1 into L1 | Per-pair hazard on the same pieces |
|---|---|---:|---:|---:|---:|
| **XLM-R** | **0.646** (0.625 / 0.669) | **0.676** | **0.536** | 0.759 | 0.266 |
| **mBERT** | 0.634 (0.610 / 0.660) | 0.662 | 0.512 | 0.759 | 0.254 |

| Pair | XLM-R | mBERT | mBERT − XLM-R | Hazard (XLM-R pieces) | Hazard (mBERT pieces) |
|---|---:|---:|---:|---:|---:|
| Korean | **0.808** | 0.793 | -0.015 | 0.265 | 0.208 |
| Japanese | **0.736** | 0.700 | -0.036 | 0.243 | 0.182 |
| German | **0.733** | 0.718 | -0.015 | 0.268 | 0.248 |
| Hindi | **0.715** | 0.706 | -0.009 | 0.362 | 0.307 |
| Chinese | **0.699** | 0.676 | -0.023 | 0.242 | 0.142 |
| French | **0.695** | 0.689 | -0.006 | 0.208 | 0.195 |
| Spanish | **0.673** | 0.660 | -0.013 | 0.239 | 0.223 |
| Italian | **0.671** | 0.658 | -0.013 | 0.242 | 0.193 |
| Arabic | **0.664** | 0.649 | -0.015 | 0.241 | 0.176 |
| Russian | **0.659** | 0.645 | -0.014 | 0.243 | 0.188 |
| German→French | 0.658 | **0.663** | +0.006 | 0.223 | 0.202 |
| Tagalog | **0.638** | 0.627 | -0.011 | 0.323 | 0.287 |
| Vietnamese | **0.622** | 0.608 | -0.014 | 0.248 | 0.221 |
| Cantonese | **0.619** | 0.609 | -0.010 | 0.252 | 0.265 |
| Malay | **0.548** | 0.531 | -0.017 | 0.281 | 0.254 |

## Shared positions: mBERT vs the from-scratch models, on identical words

Scored only at the last piece of each non-CJK word (in CJK rows, the English words). The word model also receives the current word's language as an input, which the piece models do not, so its row flatters it.

| Model | Shared F1 | Macro-F1 | F1 into L2 | F1 into L1 |
|---|---:|---:|---:|---:|
| Per-pair hazard (mBERT pieces) | 0.357 | 0.289 | 0.304 | 0.376 |
| From-scratch, XLM-R pieces | 0.546 | 0.495 | 0.419 | 0.601 |
| From-scratch, words (with language input) | 0.588 | 0.534 | 0.422 | 0.652 |
| **mBERT** | **0.719** | **0.692** | **0.590** | **0.773** |

| Pair | **mBERT** | From-scratch, words | From-scratch, pieces | Shared test words |
|---|---:|---:|---:|---:|
| Korean | **0.803** | 0.663 | 0.584 | 16,453 |
| Cantonese | **0.791** | 0.675 | 0.622 | 220,068 |
| Chinese | **0.773** | 0.608 | 0.577 | 21,368 |
| Japanese | **0.770** | 0.616 | 0.560 | 22,228 |
| German | **0.728** | 0.516 | 0.500 | 32,290 |
| Hindi | **0.718** | 0.616 | 0.586 | 45,752 |
| French | **0.696** | 0.518 | 0.494 | 41,448 |
| German→French | **0.673** | 0.433 | 0.403 | 39,912 |
| Spanish | **0.666** | 0.489 | 0.457 | 45,175 |
| Italian | **0.664** | 0.467 | 0.446 | 41,459 |
| Arabic | **0.663** | 0.482 | 0.431 | 37,044 |
| Russian | **0.653** | 0.470 | 0.408 | 37,302 |
| Tagalog | **0.637** | 0.548 | 0.528 | 46,130 |
| Vietnamese | **0.613** | 0.453 | 0.413 | 55,971 |
| Malay | **0.537** | 0.459 | 0.419 | 44,558 |

## Duration, at true switch points whose class is certain

| Model (own pieces) | n | 3-class accuracy | Macro-F1 | F1 Small / Medium / Large |
|---|---:|---:|---:|---|
| XLM-R | 180,990 | 0.678 | 0.575 | 0.636 / 0.278 / 0.813 |
| mBERT | 181,475 | 0.713 | 0.576 | 0.672 / 0.221 / 0.836 |

Durations are counted in each model's own pieces, and mBERT counts CJK runs in characters, so more of its runs are Large. Accuracy is therefore not comparable across the two, but macro-F1 is close (0.575 vs 0.576). mBERT's confusion matrix, rows = truth, columns small / medium / large: small [33427, 3610, 10781], medium [12843, 5121, 16899], large [5356, 2652, 90786]. Medium is again the hardest class.

## Zero-shot: 4 languages held out (the universality test)

Both models were trained on the same 10 first languages; the threshold was tuned on the trained pairs' validation split only.

| Held-out pair | **XLM-R zero-shot** | mBERT zero-shot | mBERT − XLM-R | XLM-R trained on it | mBERT trained on it | mBERT zero-shot, shared | Always switch (mBERT pieces) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Korean | **0.775** | 0.663 | -0.111 | 0.808 | 0.793 | 0.802 | 0.148 |
| Italian | **0.661** | 0.634 | -0.027 | 0.671 | 0.658 | 0.643 | 0.152 |
| German | **0.653** | 0.620 | -0.033 | 0.733 | 0.718 | 0.631 | 0.165 |
| Hindi | **0.595** | 0.528 | -0.067 | 0.715 | 0.706 | 0.549 | 0.297 |
| German→French | **0.450** | 0.417 | -0.033 | 0.658 | 0.663 | 0.423 | 0.153 |
| **All held out** | **0.625** (macro 0.627) | 0.565 (macro 0.573) | -0.054 | 0.717 (mean) | 0.708 (mean) | 0.582 | |

| Zero-shot, held-out pairs | XLM-R | mBERT |
|---|---:|---:|
| Relative drop vs trained-on | **12.6%** | 19.1% |
| F1 into L2 / back into L1 | **0.529** / 0.730 | 0.424 / 0.720 |
| Duration accuracy / macro-F1 | 0.660 / 0.562 | 0.646 / 0.539 |

## Reading

- **Trained on all pairs, the two are close, with XLM-R slightly ahead.**
  - XLM-R's switch F1 is higher on 14 of 15 pairs, by 0.006–0.036 (about 0.011 on non-CJK pairs, 0.021 on CJK). mBERT is ahead only on German→French (+0.006).
  - The gaps are small but mostly beyond the ±0.004 that two XLM-R runs differed by.
  - Measured against each model's own-tokenizer baseline, mBERT gains more than XLM-R on 14 of 15 pairs (all but Cantonese). Part of the raw gap is the tokenizer, not the model.
- **Zero-shot, XLM-R is clearly better, and that is the project's question.**
  - XLM-R is ahead on all five held-out pairs, and loses 12.6% relative when a language is held out against mBERT's 19.1%.
  - The largest gap is Korean (0.775 vs 0.663): mBERT's trained-on Korean score is nearly as high (0.793), so the difference is in transfer, not capacity.
  - On switching *into* L2 zero-shot, XLM-R scores 0.529 against 0.424.
  - A plausible reason: XLM-R's pretraining corpus (CC-100: web text, including informal and romanized writing) is closer to code-switched conversation than mBERT's Wikipedia.
- **German→French transfers for both** (XLM-R 0.450, mBERT 0.417, against always-switching 0.153–0.167). Transfer to an unseen non-English second language is a property of multilingual pretraining, not of one model.
- **On identical words, mBERT beats both from-scratch models on all 15 pairs.** Its shared F1 is 0.719, against 0.546 on pieces and 0.588 for the word model, which even gets the language as an input. Pretraining is worth about +0.15 on the same positions.
- **Duration:** mBERT and XLM-R have almost the same macro-F1 (0.576 vs 0.575). Medium is the weakest class for both.

## To make XLM-R vs mBERT exact

Re-run XLM-R with the current notebook (`MODEL = "xlmr"`, `RUN_FULL = True`, `ZERO_SHOT = True`, about 2.5 h on Kaggle). It will then report the shared positions, and the two models can be compared on identical words, in and out of distribution.
