# 2026-09-29: XLM-R with a causal mask, zero-shot (4 languages held out)

**Run:** `kaggle/train_xlmr.ipynb` with `ZERO_SHOT = True`, on a Kaggle T4. The settings match the [all-pairs run](2026-09-29_xlmr_causal.md): 12,000 steps, batch 16, window 384 / stride 288, learning rate 2e-5 for the body and 1e-3 for the heads, pieces only. The causality check passed.

**Design** (report §1.4):
- **Trained on 10 first languages:** Cantonese, Arabic, Tagalog (Philippines), Chinese, Vietnamese, Malay, Japanese, Russian, Spanish, French.
- **Held out entirely:** Korean, Italian, Hindi and German, which gives five pairs, since German has both English and French as its second language.
- **Threshold:** tuned on the validation split of the trained pairs only (τ = 0.358), then applied unchanged to the held-out test pairs.
- **Results file:** [`2026-09-29_xlmr_causal_zeroshot.json`](2026-09-29_xlmr_causal_zeroshot.json), also at `data/anticipation_xlmr_zeroshot.json`.

## Held-out pairs (never seen in training), test split

| Held-out pair | **Zero-shot F1** | P | R | Same pair, trained on (all-pairs XLM-R) | Drop | From-scratch, trained on (pieces) | Always switch |
|---|---:|---:|---:|---:|---:|---:|---:|
| Korean→English | **0.775** | 0.768 | 0.782 | 0.808 | −0.033 | 0.611 | 0.173 |
| Italian→English | **0.661** | 0.638 | 0.686 | 0.671 | −0.010 | 0.406 | 0.162 |
| German→English | **0.653** | 0.638 | 0.670 | 0.733 | −0.080 | 0.469 | 0.177 |
| Hindi→English | **0.595** | 0.634 | 0.561 | 0.715 | −0.120 | 0.541 | 0.344 |
| German→French | **0.450** | 0.497 | 0.411 | 0.658 | −0.208 | 0.387 | 0.167 |
| **All held out** (n = 256,376) | **0.625** (macro 0.627) | 0.641 | 0.608 | 0.717 (mean) | **−12.6% relative** | | |

| Held-out pairs, other metrics | Zero-shot XLM-R |
|---|---:|
| F1 switching into L2 / back into L1 | 0.529 / 0.730 |
| Duration: 3-class accuracy / macro-F1 | 0.660 / 0.562 |
| Duration F1: Small / Medium / Large | 0.627 / 0.268 / 0.792 |

The same model on its 10 **trained** pairs scores F1 0.634 (macro 0.656).

## Reading

- **Universality holds far better than before.** Held-out F1 is 0.625, against 0.717 on the same pairs when they are trained on: a 12.6% relative drop. The from-scratch word model lost 27% zero-shot (report §4.5). Zero-shot XLM-R beats the from-scratch model **that was trained on those pairs** on all five, and beats the per-pair hazard fitted with access to them.
- **Near transfer is almost free, and distance matters less than script.**
  - **Italian** loses only 0.010, with Spanish and French in training.
  - **Korean** loses only 0.033, with no Koreanic language in training. Its script is disjoint from English, so the switch cue is visible.
  - **Hindi** loses the most among the English pairs (−0.120). Most Hindi rows are romanized, so the model must tell romanized Hindi from English with no Hindi in training.
- **German→French, the one non-English second language, now transfers:** F1 0.450 against 0.167 for always-switching.
  - The from-scratch word model scored 0.243 there, against 0.212 for always-switching, which is essentially no signal.
  - So the report's conclusion that the model learned "anticipate English" rather than "anticipate a switch" (§4.5) is **partly overturned** by a pretrained multilingual model.
  - It is still the largest drop (−0.208, and 0.203 below German→English from the same held-out first language), so the English bias is weaker but still present.
- **Caveat.** The from-scratch zero-shot result (held-out F1 0.383) was scored on words with the earlier report, with the two German pairs merged. The like-for-like comparison is a piece-level from-scratch zero-shot run with the current report, which has not been run yet.
