# Causal mBERT on 6 Latin-script pairs (2026-10-13)

**Pairs:** Hindi, Spanish, French, Italian, German and Tagalog, each → English.
**Notebook:** [`kaggle/train_mbert_6pairs.ipynb`](../../kaggle/train_mbert_6pairs.ipynb), Kaggle version 2 ("Full Run"), GPU T4, 1,682 s (28 min).
**Author:** Sumitra Chhetri

## Summary

- Causal mBERT reaches a mean **anticipatory F1 of 0.670** (validation) across the 6 pairs, against **0.497** for the naive current-word baseline (test).
- It is also more consistent across pairs: **σ = 0.041** vs 0.080 for the baseline.
- The gain is largest for French, German, Italian and Spanish (+0.21 to +0.24) and smallest for Hindi and Tagalog (+0.07 to +0.08), where the current word alone already predicts much of the switching.

## Setup

| | |
|---|---|
| Model | `bert-base-multilingual-cased`, `is_decoder=True` (causal), 177 M parameters, plus a switch head (2 classes) and a duration head (3 classes) on every piece |
| Loss | λ₁·CE(switch) + λ₂·CE(duration), λ₁ = λ₂ = 1, `ignore_index=-1` for both |
| Training | 3 epochs over all 38,318 train conversations, batch 16 (7,185 steps), AdamW lr 2e-5, weight decay 0.01, 10% linear warmup, fp16, grad clip 1.0, no random seed set |
| Data | `token_labels_word.parquet` from `CodeSwitchGroup4/switchlingua-tagged-final`; split by `row_id % 10` (0–7 train, 8 val, 9 test) |
| Scored | Validation split, 4,789 conversations; decision = argmax (no threshold tuning) |

### Labels
- **Unit:** the word. Punctuation and digits (`-`) are skipped: `y_sw` is −1 there and they neither break nor extend a run.
- **Switch:** the group's `y_sw` as is: 1 if the next word is in the other language.
- **Duration:** `small`/`medium`/`large` → 0/1/2 at switches. Set to −1 where `dur_known` is false (the burst is cut off by the end of the text and could still be any class): 3.5–8.8% of switches per pair.
- **Mapping to mBERT pieces:** each word's labels go on its **last** piece; every other piece and `[CLS]`/`[SEP]` are −1. The prediction is made once word *t* has been fully read. A check confirms that no word-level label is lost.
- **Normalization:** curly apostrophes, curly quotes and en/em dashes are converted to ASCII before tokenizing. mBERT's vocabulary lacks them, so `J’ai`, `it’s` and `don’t` became `[UNK]`. After the fix the `[UNK]` rate is ≤ 0.01% per pair.

### Causal check
Replacing every piece after position *k* with random pieces and comparing the outputs at positions 0..*k*:

| Model | Largest change in outputs 0..*k* |
|---|---:|
| Causal mBERT | **0.00** |
| Bidirectional mBERT (no mask) | 3.57 |

The check ran on one validation conversation (89 pieces, *k* = 44). On the test run it was 0.00 vs 3.71 on another (197 pieces).

## Results

**Switch (anticipatory F1, SWITCH class) and duration accuracy, after epoch 3:**

| Pair | mBERT F1 (val) | Naive F1 (test) | Gain | Duration acc. (val) |
|---|---:|---:|---:|---:|
| German–English | **0.717** | 0.481 | +0.236 | 0.730 |
| Hindi–English | **0.716** | 0.634 | +0.082 | 0.692 |
| French–English | **0.678** | 0.455 | +0.223 | 0.755 |
| Italian–English | **0.647** | 0.429 | +0.218 | 0.752 |
| Spanish–English | **0.643** | 0.432 | +0.211 | 0.730 |
| Tagalog–English | **0.616** | 0.549 | +0.067 | 0.624 |
| **Mean** | **0.670** | 0.497 | +0.173 | 0.714 |
| **σ across pairs** | **0.041** | 0.080 | | 0.049 |

- **Naive baseline:** P(switch | current word, lowercased), fitted on train, with the cutoff tuned on val and scored on test. Two other naive rules score lower: "always switch" 0.264 mean, and "current word's language" 0.0, because neither language switches more than half the time.
- **Duration accuracy** is counted at true switches with a known class. It beats always predicting each pair's most common class: about 0.52 for Tagalog, 0.55 for German and 0.59–0.60 for the others.

**By epoch (validation):**

| Epoch | Mean F1 | σ F1 | Mean duration acc. | Train loss (switch / duration) at epoch end |
|---|---:|---:|---:|---|
| 1 | 0.637 | 0.046 | 0.704 | 0.253 / 0.713 |
| 2 | 0.662 | 0.043 | 0.713 | 0.228 / 0.658 |
| 3 | 0.670 | 0.041 | 0.714 | 0.215 / 0.627 |

Gains are shrinking (+0.025, then +0.008). Validation F1 kept rising, so there is no sign of overfitting by epoch 3.

**The pairs fall into two groups.** Hindi and Tagalog switch about twice as often as the others (22–26% of words vs about 11–12%), and mostly in short 1–2 word bursts (49–57% of switches). French, German, Italian and Spanish switch less often, and about half their switches are 7+ words. The current word is a strong cue for short insertions, so the baseline is already high for Hindi and Tagalog and mBERT adds less there. Clause-level switches benefit more from context.

## Caveats

- **Splits differ:** mBERT is scored on **validation** and the naive baseline on **test**. A test-split score for mBERT is still to do.
- **One run, no fixed seed:** differences of about 0.01 between pairs, or in σ, may be noise.
- **No threshold tuning:** a val-tuned switch threshold would likely raise F1.
- **Label noise:** labels for these Latin-script pairs come from AnE-LID, not human annotation, so some errors are the tagger's, not the model's.

## Next steps

1. Score the trained model once on the **test** split.
2. Tune the switch threshold on validation.
3. Run 3 seeds to put intervals on F1 and σ.
4. Add duration macro-F1 and the per-switch-type breakdown (`cs_type`).
5. Error analysis, especially for Tagalog (lowest F1 and duration accuracy).

## Reproducing

1. Upload `kaggle/train_mbert_6pairs.ipynb` to Kaggle.
2. Turn on a GPU and Internet, and add `HF_TOKEN` under Add-ons → Secrets.
3. Set `SMOKE = False` in the DataLoader cell.
4. Run **Save Version → Save & Run All**.

Outputs are written to `/kaggle/working/`: `causal_mbert.pt` and `loss_log.csv`.
