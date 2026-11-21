# 2026-11-20: Where do the final models miss, per language? (local, no GPU)

**Why:** to find, per language pair, which kind of switch the two final models get wrong: intra-sentential, inter-sentential or extra-sentential (tag). And to test the idea that **Korean scores best because its sentence-final endings mark the end of a clause**.

**Models:** `xlmr_prev_gap_36k_rsv2` (τ = 0.413) and `mbert_prev_gap_36k_rsv2` (τ = 0.430). Both use `decide_at=gap` and `--prev-lang`, 36k steps, seed 0. Units are each model's own pieces. Scored on the test split (`row_id % 10 == 9`) at each run's own τ.

**How:**
```
./.venv/bin/python scripts/analysis/error_breakdown.py xlmr_prev_gap_36k_rsv2
./.venv/bin/python scripts/analysis/error_breakdown.py mbert_prev_gap_36k_rsv2 --path data/subword_mbert.parquet
```
Each run takes about 15 s and writes `data/results/error_breakdown_<tag>.json`. Its pooled F1 reproduces the run JSON exactly (0.7610 / 0.7623), so the gap relocation and the joins are right. Every position is sliced four ways:

- **`cs_type`**: SwitchLingua's own label (Intrasentential / Intersentential / Extra-sentential tag, about ⅓ of rows each), joined from `data/tagged.parquet`. It is the type the generator was *asked* for, so it is validated against the observed switch positions first.
- **Gap bucket** (`punct_breakdown`): what lies between the decision word and the next language-bearing piece. `adjacent` = mid-clause, `comma`, `sentence_end`, `turn`.
- **Direction**: into L2 (out of L1) vs back into L1.
- **L1 ending** (Korean, Japanese, Chinese, Cantonese): whether the L1 text read so far ends in a sentence-final form, a connective, a particle or an informal -어/-아 ending. These are suffix heuristics (`ENDINGS`), and they are causal: they use pieces ≤ t only.

**Mix-standardised F1** reweights each pair's positions to the pooled share of every (bucket × label) cell. It answers whether a pair is better *within* each kind of switch, or just has an easier mix.

XLM-R and mBERT agree to ±0.01 on almost every cell below, so the tables show XLM-R and the findings hold for both.

## 1. Per pair: headline F1 vs mix-standardised F1

| Pair | F1 | Std F1 | Mid-clause share of switches | F1 adjacent | F1 comma | F1 sent. end | F1 turn | Share of misses that are mid-clause |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Korean→English | **0.899** | 0.790 | 0.20 | 0.59 | **0.95** | 0.98 | 0.97 | 0.89 |
| Japanese→English | 0.878 | **0.806** | 0.20 | 0.61 | 0.86 | 0.97 | 0.98 | 0.70 |
| French→English | 0.867 | 0.742 | 0.18 | 0.37 | 0.88 | 0.98 | 0.99 | 0.84 |
| German→English | 0.847 | 0.739 | 0.19 | 0.37 | 0.87 | 0.98 | 0.99 | 0.78 |
| Chinese→English | 0.843 | 0.766 | 0.21 | 0.48 | 0.81 | 0.97 | 0.99 | 0.71 |
| Italian→English | 0.840 | 0.735 | 0.22 | 0.38 | 0.86 | 0.97 | 0.98 | 0.83 |
| Spanish→English | 0.839 | 0.742 | 0.22 | 0.41 | 0.86 | 0.96 | 0.98 | 0.82 |
| Arabic→English | 0.836 | 0.770 | 0.28 | 0.45 | 0.89 | 0.98 | 0.98 | 0.89 |
| German→French | 0.793 | 0.715 | 0.27 | 0.32 | 0.84 | 0.97 | 0.97 | 0.85 |
| Russian→English | 0.792 | 0.748 | 0.30 | 0.41 | 0.83 | 0.99 | 0.98 | 0.84 |
| Vietnamese→English | 0.792 | 0.746 | 0.36 | 0.50 | 0.86 | 0.97 | 0.98 | 0.88 |
| Hindi→English | 0.775 | 0.663 | **0.67** | **0.71** | 0.89 | 0.95 | 0.91 | 0.92 |
| Malay→English | 0.733 | 0.709 | 0.47 | 0.50 | 0.82 | 0.94 | 0.97 | 0.88 |
| Philippines→English | 0.723 | 0.656 | **0.65** | 0.66 | 0.72 | 0.86 | 0.95 | 0.83 |
| Cantonese→English | 0.723 | 0.739 | 0.44 | 0.56 | **0.67** | 0.90 | 0.90 | 0.71 |
| **All (pooled)** | 0.761 | 0.761 | 0.41 | 0.57 | 0.77 | 0.93 | 0.93 | 0.76 |

- **The ranking is mostly the switch mix.** Pairs whose switches sit behind punctuation score high, and pairs whose switches are mid-clause score low. Korean, Japanese and the European pairs have only 18–22% mid-clause switches. Hindi and Philippines have 65–67%, and Malay 47%.
- **Korean's lead shrinks after standardisation, and it loses first place to Japanese.** It drops from 0.899 to 0.790, behind Japanese (0.806). Its lead over the mean pair goes from 0.087 (0.899 vs 0.812) to 0.052 (0.790 vs 0.738), so about 40% of the lead is its easy mix. What is left comes mostly from the **comma bucket**: 0.95, the best of any pair (the median is 0.86). Section 3 explains why.
- **Hindi and Philippines are hard twice over.** They have the most mid-clause switches and the lowest standardised F1 (0.66). They are actually the *best* pairs on mid-clause switches (0.71 / 0.66), because those switches are common there (base rate 0.13–0.15, against about 0.02 elsewhere). The bigger loss is that they underperform the other pairs on every boundary bucket.
- **Cantonese is the one pair that standardisation helps** (0.723 → 0.739). Its weak spot is commas (0.67) and sentence ends (0.90). After a comma, Cantonese switches only 24% of the time (the pooled rate is 32% and Korean's is 68%), so a comma there is a weak cue.

## 2. `cs_type` (intra / inter / extra) — the type label adds nothing once position is known

Validation: the share of each type's switches by observed bucket (pooled):

| cs_type | adjacent | comma | sentence end | turn |
|---|---:|---:|---:|---:|
| intra | **0.53** | 0.18 | 0.09 | 0.18 |
| inter | 0.32 | 0.07 | **0.30** | **0.29** |
| extra (tag) | 0.37 | **0.28** | 0.07 | 0.26 |

The label points the right way: intra rows put their switches mid-clause, inter rows at sentence ends, and tag rows at commas. But it is loose. A third of the switches in "inter" rows are mid-clause, because the generator does not follow its instruction strictly.

| Pooled | F1 | Base rate | Share of misses | Share of false alarms | Duration macro-F1 |
|---|---:|---:|---:|---:|---:|
| intra | **0.702** | 0.108 | **0.44** | 0.44 | **0.588** |
| inter | 0.808 | 0.085 | 0.24 | 0.24 | 0.628 |
| extra | 0.781 | 0.121 | 0.33 | 0.32 | 0.627 |

F1 by cs_type × bucket:

| cs_type | adjacent | comma | sentence end | turn |
|---|---:|---:|---:|---:|
| intra | 0.56 | 0.73 | 0.89 | 0.92 |
| inter | 0.58 | 0.58 | 0.96 | 0.93 |
| extra | 0.57 | 0.84 | 0.89 | 0.94 |

- **Intra-sentential rows cause the most errors** (44% of misses and of false alarms, F1 0.70 against 0.81 for inter), and every pair shows this (intra is the lowest column for all 15).
- **But within a bucket, the type hardly matters.** Mid-clause F1 is 0.56 / 0.58 / 0.57 for the three types. The intra gap is the same thing as section 1: intra rows have more mid-clause switches. Where in the sentence a switch sits explains the errors, and the generator's label adds nothing on top.
- The one cell that differs is **commas in tag-switching rows: 0.84**, against 0.58 in inter rows. A tag ("you know,", "right,") is a short, recognisable phrase followed by a comma.
- **Duration is also hardest in intra rows** (0.588 against 0.63): short insertions inside a clause are Small/Medium, and those are the confusable classes.
- **Direction is still the biggest single split.** Switches out of L1 have F1 0.650 and cause 83% of misses. Switches back into L1 have F1 0.878. This matches the 2026-11-01 grammar breakdown on the older model.

## 3. Korean: do sentence-final endings mark the switch?

Korean L1 positions, by the ending of the word read so far:

| Ending | n | Switch rate | Mean P(switch) | F1 | Share of misses |
|---|---:|---:|---:|---:|---:|
| final (다, 요, 니다, 까, 죠, 거든…) | 1,978 | **0.545** | 0.542 | **0.977** | 0.03 |
| informal (-어/-아, 해, 야, 지) | 2,570 | 0.447 | 0.453 | 0.965 | 0.04 |
| connective (는데, 고, 서, 면, 지만…) | 1,518 | 0.122 | 0.137 | 0.774 | 0.05 |
| particle (은/는, 이/가, 을/를, 에, 에서…) | 9,429 | 0.033 | 0.037 | 0.257 | **0.43** |
| other (mid-word, nouns) | 19,825 | 0.013 | 0.014 | 0.551 | 0.25 |

The same table, with the ending crossed with what follows the word:

| Korean | n | Switch rate | F1 |
|---|---:|---:|---:|
| final + **no punctuation** | 805 | **0.010** | 0.67 |
| final + comma | 445 | **0.942** | 0.987 |
| informal + comma | 713 | 0.958 | 0.988 |
| connective + comma | 285 | 0.530 | 0.814 |
| other + comma | 112 | 0.598 | 0.895 |
| final + sentence end | 649 | 0.960 | 0.984 |

The hypothesis is **half right**:

- **An ending on its own is not a switch cue.** With no punctuation after it, a final ending switches 1% of the time, no more than any other word. Without punctuation it is usually not the end of a clause at all (for example, an adnominal form before a noun).
- **An ending together with punctuation is a strong cue, and the model uses it.** After a comma, a final or informal ending switches 94–96% of the time; after a connective, 53%. The ending tells the model whether a comma closes a sentence (which then switches) or joins two clauses. This is why Korean's comma F1 is 0.95 against 0.86 for the median pair. Because the run decides at the gap, the model reads both the ending and the comma. The mean P(switch) tracks the true rate within 0.02 in every row, so the model already knows the cue; it is not a missed signal.
- **Korean's remaining misses are English insertions after particles** (43% of misses, F1 0.26, base rate 3%). An example is "전에 VPN 앱을": the English noun follows a phrase boundary that the particle marks, and whether it does is a lexical choice. Nothing in the prefix reveals it.
- **Japanese shows the same mechanism** (SOV, sentence-final forms): a comma after a final form switches 94% of the time, against 24% after a particle. It has the same small mid-clause share. This is why Japanese and Korean are #1 and #2 after standardisation.
- **The Chinese control shows a much weaker effect** (comma after 了/吗/呢/吧: 0.62, against 0.45 otherwise). **Cantonese is reversed**: a comma after a sentence-final particle (啦/喎/嘅…) switches only 8% of the time, against 21% otherwise. Cantonese uses these particles everywhere, including mid-utterance, so they don't mark a boundary. This fits Cantonese's weak comma F1.

**Conclusion:** Korean is best for two reasons. 80% of its switches sit at punctuation or turn boundaries, which every model finds easy. And verb-final morphology disambiguates its commas. Its hard core is the same as every pair's: an English noun inserted mid-clause.

## Caveats

- The suffix lists are heuristics. A noun can end in a "particle" syllable (플레이 → 이; pinned in `tests/test_error_breakdown.py`), and -어/-아 is both an informal ending and a connective, so it has its own class. The particle class is therefore noisy, but the final and comma effects are 0.5–0.9 differences in switch rate, far above any plausible noise.
- `cs_type` is the generation prompt's label, not an annotation.
- These are single seeds. The bucket-level F1 differences between XLM-R and mBERT are ≤ 0.02, smaller than any effect discussed here.
- The Japanese, Chinese and Cantonese "words" are runs of L1 characters (there are no spaces), so the ending is the suffix of the L1 text read so far.
