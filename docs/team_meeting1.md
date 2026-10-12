# Team meeting 1: XLM-R on the 6 script-disjoint pairs

Results of XLM-R on the six language pairs whose scripts don't overlap with English: Cantonese, Chinese, Japanese, Korean, Russian and Arabic → English. Because the scripts differ, every word's language label is exact, so these errors are the model's, not the tagger's.

**Run:** `xlmr_6pairs`, Kaggle T4, 2026-10-12.
- **Model:** XLM-R base (277 M parameters) with a causal mask.
- **Training:** 12,000 steps, seed 0, on these 6 pairs only.
- **Data split:** train on 80% of conversations, tune on 10% (val), report on the other 10% (test), scored once.
- **Full record:** [`results/2026-10-12_xlmr_6pairs_duration.md`](results/2026-10-12_xlmr_6pairs_duration.md), with the bootstrap intervals and the class-weighted comparison run.

## How the model makes its predictions

The model reads a conversation one sub-word piece at a time, as an XLM-R tokenizer splits it. It sees only the pieces themselves: no language tags, and not which language pair it is reading. At every piece it makes two predictions about the *next* piece: will it switch language (**switch**), and if so, how long will the new-language stretch last: Small (1–2 pieces), Medium (3–6) or Large (7+) (**duration**). The whole conversation (up to 384 pieces) goes through the model in one pass, but a causal attention mask means the prediction at piece t can only draw on pieces 1…t. That is the same as feeding it each prefix separately. Before training, an automated check replaces every later piece with random ones and confirms the earlier predictions don't change at all (drift 0.0). The labels come from the next piece, so the model can never see the answer it is asked for.

**Switch:** the model outputs a probability, and we call a switch when it is at least 0.377. That threshold was chosen on the val split to maximise F1, then applied unchanged to test. Punctuation, digits and the final piece of each conversation are never scored. **Duration** is scored only where a switch really happens and its length class is certain (a stretch cut off by the end of the text counts only once it has reached 7+ pieces). The model's three class probabilities are adjusted by one offset per class, also fitted on val, which stops it defaulting to Large, the majority class. So the duration numbers answer "given that a switch is coming, how long will it last?" (the switch is known in advance), not the end-to-end question.

## Summary

**Switch prediction** (all 6 pairs: F1 0.645, precision 0.657, recall 0.634; average over the pairs 0.703):

| Pair | Test positions | True switches | F1 | Precision | Recall | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Korean | 55,454 | 5,240 | **0.813** | 0.790 | 0.837 | 0.964 |
| Japanese | 64,510 | 5,658 | **0.741** | 0.713 | 0.771 | 0.953 |
| Chinese | 62,016 | 5,942 | **0.702** | 0.658 | 0.752 | 0.939 |
| Arabic | 51,395 | 4,908 | **0.669** | 0.647 | 0.692 | 0.934 |
| Russian | 49,679 | 4,677 | **0.666** | 0.656 | 0.675 | 0.936 |
| Cantonese | 1,061,851 | 104,881 | **0.625** | 0.647 | 0.605 | 0.928 |

**Accuracy is not the headline number.** About 90% of positions are "no switch", so always predicting "no switch" already scores about 0.90. F1, precision and recall on the switch class are the measures that matter.

**Duration prediction** (all 6 pairs: accuracy 0.663, macro-F1 0.602; n = 125,675 true switch points):

| Pair | Switch points | Accuracy | Macro-F1 | Within one class |
|---|---:|---:|---:|---:|
| Korean | 4,916 | 0.763 | **0.597** | 0.912 |
| Japanese | 5,377 | 0.778 | **0.623** | 0.941 |
| Chinese | 5,623 | 0.729 | **0.607** | 0.946 |
| Arabic | 4,562 | 0.705 | **0.596** | 0.927 |
| Russian | 4,363 | 0.701 | **0.593** | 0.920 |
| Cantonese | 100,834 | 0.645 | **0.598** | 0.939 |

- **Macro-F1** averages the three classes equally, so a model that ignores Medium can't score well on it.
- **Within one class** counts a prediction as close if it is off by at most one class (Small↔Medium or Medium↔Large).
- **Compare duration within a script only:** Chinese, Japanese, Korean and Cantonese runs count characters, while Russian and Arabic count words.

## Per-pair detail

Each pair has two confusion matrices: switch on all test positions, and duration (Small / Medium / Large) at true switch points. Precision and recall in the duration table are per class.

### Korean → English

**Switch** (F1 0.813, precision 0.790, recall 0.837, accuracy 0.964):

| | Predicted switch | Predicted no switch |
|---|---:|---:|
| **Actual switch** | 4,388 | 852 |
| **Actual no switch** | 1,164 | 49,050 |

**Duration** (accuracy 0.763, macro-F1 0.597):

| Actual \ predicted | Small | Medium | Large | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| **Small (1–2)** | 695 | 111 | 287 | 0.651 | 0.636 | 0.643 |
| **Medium (3–6)** | 227 | 137 | 250 | 0.349 | 0.223 | 0.272 |
| **Large (7+)** | 146 | 145 | 2,918 | 0.845 | 0.909 | 0.876 |

### Japanese → English

**Switch** (F1 0.741, precision 0.713, recall 0.771, accuracy 0.953):

| | Predicted switch | Predicted no switch |
|---|---:|---:|
| **Actual switch** | 4,365 | 1,293 |
| **Actual no switch** | 1,755 | 57,097 |

**Duration** (accuracy 0.778, macro-F1 0.623):

| Actual \ predicted | Small | Medium | Large | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| **Small (1–2)** | 657 | 143 | 178 | 0.653 | 0.672 | 0.662 |
| **Medium (3–6)** | 209 | 207 | 308 | 0.365 | 0.286 | 0.321 |
| **Large (7+)** | 140 | 217 | 3,318 | 0.872 | 0.903 | 0.887 |

### Chinese → English

**Switch** (F1 0.702, precision 0.658, recall 0.752, accuracy 0.939):

| | Predicted switch | Predicted no switch |
|---|---:|---:|
| **Actual switch** | 4,470 | 1,472 |
| **Actual no switch** | 2,326 | 53,748 |

**Duration** (accuracy 0.729, macro-F1 0.607):

| Actual \ predicted | Small | Medium | Large | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| **Small (1–2)** | 499 | 255 | 206 | 0.612 | 0.520 | 0.562 |
| **Medium (3–6)** | 222 | 410 | 416 | 0.411 | 0.391 | 0.401 |
| **Large (7+)** | 95 | 332 | 3,188 | 0.837 | 0.882 | 0.859 |

### Arabic → English

**Switch** (F1 0.669, precision 0.647, recall 0.692, accuracy 0.934):

| | Predicted switch | Predicted no switch |
|---|---:|---:|
| **Actual switch** | 3,397 | 1,511 |
| **Actual no switch** | 1,857 | 44,630 |

**Duration** (accuracy 0.705, macro-F1 0.596):

| Actual \ predicted | Small | Medium | Large | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| **Small (1–2)** | 526 | 248 | 181 | 0.614 | 0.551 | 0.581 |
| **Medium (3–6)** | 177 | 297 | 321 | 0.366 | 0.374 | 0.370 |
| **Large (7+)** | 154 | 266 | 2,392 | 0.827 | 0.851 | 0.838 |

### Russian → English

**Switch** (F1 0.666, precision 0.656, recall 0.675, accuracy 0.936):

| | Predicted switch | Predicted no switch |
|---|---:|---:|
| **Actual switch** | 3,157 | 1,520 |
| **Actual no switch** | 1,653 | 43,349 |

**Duration** (accuracy 0.701, macro-F1 0.593):

| Actual \ predicted | Small | Medium | Large | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| **Small (1–2)** | 554 | 205 | 164 | 0.583 | 0.600 | 0.591 |
| **Medium (3–6)** | 210 | 258 | 278 | 0.357 | 0.346 | 0.351 |
| **Large (7+)** | 187 | 259 | 2,248 | 0.836 | 0.834 | 0.835 |

### Cantonese → English

**Switch** (F1 0.625, precision 0.647, recall 0.605, accuracy 0.928):

| | Predicted switch | Predicted no switch |
|---|---:|---:|
| **Actual switch** | 63,439 | 41,442 |
| **Actual no switch** | 34,682 | 922,288 |

**Duration** (accuracy 0.645, macro-F1 0.598):

| Actual \ predicted | Small | Medium | Large | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| **Small (1–2)** | 14,537 | 6,994 | 3,634 | 0.618 | 0.578 | 0.597 |
| **Medium (3–6)** | 6,505 | 10,478 | 7,388 | 0.399 | 0.430 | 0.414 |
| **Large (7+)** | 2,470 | 8,769 | 40,059 | 0.784 | 0.781 | 0.783 |


## Takeaways

- **Switch F1 ranges from 0.625 (Cantonese) to 0.813 (Korean).** Korean particles attach directly to English words, so a switch is easy to anticipate. Cantonese switches mid-clause, rarely at punctuation, and is half the test data.
- **Duration: Large is easy (F1 0.78–0.89), Medium is hard (0.27–0.41).** Medium is mostly confused with its neighbours, and 91–95% of predictions are within one class. Medium is weakest on Korean and Japanese, where only 12.5% and 13.5% of switches are Medium.
- **Class-weighted duration training changed neither the switch scores nor, after val tuning, the duration scores** (see the full record). We use the plain model plus tuning.
