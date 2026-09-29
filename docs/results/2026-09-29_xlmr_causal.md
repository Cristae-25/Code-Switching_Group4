# 2026-09-29: XLM-R with a causal mask, all 15 pairs (sub-word)

**Run:** `kaggle/train_xlmr.ipynb` on a Kaggle T4. Model `xlm-roberta-base` with `is_decoder=True` (causal attention), plus a switch head and a 3-class duration head. The inputs are XLM-R pieces only, with no language input.

**Settings:** the notebook defaults. We assume these were not changed; confirm against `args` in the results JSON.

| Setting | Value |
|---|---|
| Steps | 12,000 (about 1 epoch) |
| Batch | 16 |
| Window / stride | 384 / 288 |
| Learning rate, body / heads | 2e-5 / 1e-3 |
| Duration loss weight | 0.3 |

**Protocol:** the same as every model (`src/evaluation.py`). Train on the train split (`row_id % 10` in 0–7), tune the threshold on val (8), score test (9) once.

**Raw per-pair table:** [`2026-09-29_xlmr_causal_per_pair.csv`](2026-09-29_xlmr_causal_per_pair.csv), exactly as reported by the notebook.

## Switch prediction (`y_sw`), test split

The per-pair test sizes are identical to the from-scratch sub-word model's (for example, Cantonese 1,061,851), so the comparison is on exactly the same positions.

| Model (pieces) | Switch F1 | Macro-F1 over pairs |
|---|---:|---:|
| Hazard, per pair (best training-free) | 0.266 | 0.259 |
| From-scratch causal transformer (23 M params) | 0.453 | 0.453 |
| **XLM-R + causal mask (278 M params)** | **0.647** † | **0.677** |

† **Overall P / R / F1 = 0.626 / 0.670 / 0.647, reconstructed** from the per-pair P and R. Each pair's true switch count comes from our labels on the same test positions (189,756 switches in total). The error is under ±0.002, from the 3-decimal rounding. Replace it with the value from `data/anticipation_xlmr.json` once that file is downloaded.

| Pair | **XLM-R** F1 | P | R | From-scratch (pieces) | Gain | Hazard, per pair |
|---|---:|---:|---:|---:|---:|---:|
| Korean→English | **0.809** | 0.778 | 0.843 | 0.611 | +0.198 | 0.265 |
| German→English | **0.735** | 0.699 | 0.776 | 0.469 | +0.266 | 0.268 |
| Japanese→English | **0.734** | 0.692 | 0.781 | 0.527 | +0.207 | 0.243 |
| Hindi→English | **0.717** | 0.640 | 0.813 | 0.541 | +0.176 | 0.362 |
| French→English | 0.698 | 0.647 | 0.757 | 0.453 | +0.245 | 0.208 |
| Chinese→English | 0.698 | 0.643 | 0.763 | 0.429 | +0.269 | 0.242 |
| Italian→English | 0.675 | 0.635 | 0.720 | 0.406 | +0.269 | 0.242 |
| Spanish→English | 0.671 | 0.622 | 0.727 | 0.425 | +0.246 | 0.239 |
| Arabic→English | 0.667 | 0.625 | 0.716 | 0.412 | +0.255 | 0.241 |
| Russian→English | 0.661 | 0.633 | 0.692 | 0.387 | +0.274 | 0.243 |
| German→French | 0.661 | 0.648 | 0.674 | 0.387 | +0.274 | 0.223 |
| Tagalog (Philippines)→English | 0.639 | 0.560 | 0.743 | 0.492 | +0.147 | 0.323 |
| Vietnamese→English | 0.622 | 0.605 | 0.640 | 0.411 | +0.211 | 0.248 |
| Cantonese→English | 0.621 | 0.622 | 0.620 | 0.442 | +0.179 | 0.252 |
| Malay→English | 0.549 | 0.534 | 0.565 | 0.397 | +0.152 | 0.281 |

## Reading

- **Pretraining is worth about +0.2 F1 everywhere.** XLM-R beats the from-scratch model on all 15 pairs, by +0.15 to +0.27 (mean +0.23), and beats the best training-free baseline by more than 2×. Four pairs pass F1 0.7: Korean, German→English, Japanese and Hindi.
- **The biggest gains are on the Latin-script, morphologically rich pairs** (Russian, Italian, Arabic, German→French: +0.26 to +0.27). These are the pairs whose words were most often out of vocabulary for the word-level model (Russian 12.6%, Arabic 11.3%, Italian 6.6%, report §5.1). This is where a pretrained multilingual vocabulary should help most.
- **German→French improves as much as any pair (+0.274).** It is 11th of 15, far above the trivial "always switch" level (F1 about 0.17 at its 9.1% switch rate). The from-scratch zero-shot model sat at that trivial level. It was trained on in this run, so it is **not** evidence about the zero-shot question in report §4.5. That needs the zero-shot run.
- **The weakest pairs are the ones that switch mid-clause:** Malay (0.549), Cantonese (0.621), Vietnamese (0.622) and Tagalog (0.639). Cantonese is half the corpus, and its F1 dominates the pooled 0.647. That is why macro-F1 (0.677) sits above micro-F1 (0.647): the model is *better* on the typical pair than on the largest one.
- **The ranking of pairs is stable.** Korean, Japanese and Hindi were already among the easiest for the from-scratch model; Malay was already among the hardest.

## Still missing from this run

These are in `data/anticipation_xlmr.json` in the notebook's Output tab; download it into the repo's `data/`:
- the exact overall P / R / F1;
- **per direction** (into L2 vs back into L1);
- **duration** (3-class accuracy, macro-F1, confusion matrix);
- the tuned threshold and training settings.

Then run `./.venv/bin/python scripts/compare_models.py` to add XLM-R to `docs/model_comparison.md`.
