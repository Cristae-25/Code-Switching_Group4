# Team meeting 1: XLM-R on the 6 script-disjoint pairs

Results of XLM-R on the six language pairs whose scripts don't overlap with English: Cantonese, Chinese, Japanese, Korean, Russian and Arabic → English. Because the scripts differ, every word's language label is exact, so these errors are the model's, not the tagger's.

**Run:** `xlmr_6pairs`, Kaggle T4, 2026-10-12.
- **Model:** XLM-R base (277 M parameters) with a causal mask.
- **Training:** 12,000 steps, seed 0, on these 6 pairs only.
- **Data split:** train on 80% of conversations, tune on 10% (val), report on the other 10% (test), scored once.
- **Full record:** [`results/2026-10-12_xlmr_6pairs_duration.md`](results/2026-10-12_xlmr_6pairs_duration.md), with the bootstrap intervals and the class-weighted comparison run.

## How the switch and duration labels were made

SwitchLingua has no word-level language labels, only the text, so every label below is derived automatically, with no hand labelling.

1. **Tag each word with its language.** For these 6 pairs this is the easy case. None of the six first languages uses the Latin alphabet, so each word's Unicode script tells us its language: Han, Kana, Hangul, Cyrillic or Arabic script means the first language (L1), and Latin script means English. No model is needed, and the tags are exact by construction, which is why we picked these pairs for the first results. Punctuation and digits get no language. Names follow the same rule: a Latin-script name ("Netflix") counts as English, and a native-script name counts as L1.
   - **The harder pairs need a model.** In the other 9 pairs (Spanish, French, German, Italian, Malay, Vietnamese, Filipino, romanized Hindi → English), both languages use the Latin alphabet, so script tells us nothing. Those rows were tagged with AnE-LID, a pretrained code-switching language-ID model (dict-frequency tagging for German→French). Those tags are *provisional* rather than exact, and they only matter for the 9 held-out pairs scored below.
   - **Where the labels came from in this run:** nothing was recomputed. The Kaggle notebook downloaded the stored final labels (`tagged.parquet` from `CodeSwitchGroup4/switchlingua-tagged-final` on Hugging Face) and only re-split the text into XLM-R pieces (step 2). That file is byte-identical to our local `data/tagged.parquet` (same SHA-256). It holds the Unicode tags for these 6 pairs and the stored AnE-LID tags for the Latin-script pairs. We checked the 6 pairs' final labels against the bare script rule: all but about 0.04% of words match. The exceptions are a few native-script characters inside names that the names rule counts as English.
2. **Map the tags onto XLM-R's sub-word pieces.** The original text is split with the XLM-R tokenizer, and each piece takes the language of the word its characters come from. Punctuation, digits and XLM-R's bare `▁` space piece are marked *no language*: they neither break nor extend a run, and are never scored.
3. **Derive the two labels at every piece t**, looking only at the pieces after t:
   - **Switch** = 1 if the next language-bearing piece is in the other language, else 0. The last piece of each conversation has no next piece, so it gets no label (masked, not set to 0).
   - **Duration** (only where switch = 1) = how many pieces the new-language stretch lasts before the language changes back: **Small** 1–2, **Medium** 3–6, **Large** 7+. If the text ends before the stretch does, we only know a lower bound, so the point is scored only once that bound reaches 7 (it is certainly Large); otherwise its duration is left out.

**Example** (Cantonese → English), as XLM-R splits it:

| Piece | 我 | 今日 | 好 | 忙 | , | 要 | 準備 | 個 | ▁presentation | ▁for | ▁tomorrow | ▁ | 啊 | 。 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Language | L1 | L1 | L1 | L1 | — | L1 | L1 | L1 | EN | EN | EN | — | L1 | — |
| Switch label | 0 | 0 | 0 | 0 | — | 0 | 0 | **1** | 0 | 0 | **1** | — | — | — |
| Duration label | | | | | | | | **Medium** (3) | | | unscored | | | |

At 個 the model must predict that English comes next and lasts 3 pieces (Medium). At ▁tomorrow the switch back to 啊 is scored, but its duration isn't: the text ends after 1 piece, so the stretch could have been Small, Medium or Large. 啊 is the last language-bearing piece, so it has no label. The model is never given the language row; it sees only the pieces.

## What we trained on and what we evaluated on

- **Trained on the 6 pairs only.** Starting from pretrained XLM-R base, we fine-tuned it on the train split (80% of conversations, chosen by `row_id % 10`) of Cantonese, Chinese, Japanese, Korean, Russian and Arabic → English. The other 9 pairs (the Latin-script ones) were not used in training at all.
- **Evaluated on the test split of the same 6 pairs**, on conversations the model never saw. Every number in this document is from those test conversations. The switch threshold and the duration offsets were picked on the val split.
- **Also scored the 9 held-out Latin-script pairs, zero-shot,** as a transfer check: F1 0.514 (average over pairs 0.520, from 0.394 for German→French to 0.585 for Vietnamese→English). An earlier model trained on all 15 pairs scored 0.659 on them, so the 6-pair model keeps about 78% without ever seeing a Latin-script switch.
- **Training on all 15 pairs instead doesn't change these 6.** The earlier all-15-pairs model scored average F1 0.698 on these 6 pairs, against 0.703 here, within run-to-run noise.

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

**Switch direction.** A switch either goes *into English* (the current piece is in the first language) or goes *back into the first language* (the current piece is English). Going back is easier on 5 of the 6 pairs: once an English stretch has run a few pieces, its end is predictable. Going into English means anticipating from content alone, and that is the core research question.

| Pair | Into English: F1 (P / R) | Back into L1: F1 (P / R) | Gap |
|---|---|---|---:|
| Korean | **0.822** (0.841 / 0.804) | 0.803 (0.737 / 0.882) | +0.019 |
| Japanese | 0.712 (0.713 / 0.711) | 0.775 (0.713 / 0.849) | −0.063 |
| Chinese | 0.646 (0.625 / 0.670) | 0.767 (0.694 / 0.858) | −0.121 |
| Russian | 0.594 (0.606 / 0.582) | 0.754 (0.714 / 0.798) | −0.160 |
| Arabic | 0.586 (0.598 / 0.573) | 0.768 (0.698 / 0.854) | −0.182 |
| Cantonese | **0.444** (0.538 / 0.378) | 0.784 (0.719 / 0.862) | −0.340 |
| **All 6, pooled** | **0.499** (0.576 / 0.440) | **0.782** (0.717 / 0.860) | −0.283 |
| Average over pairs | 0.634 | 0.775 | −0.141 |

- **Read the pooled row with care.** Cantonese accounts for 55,791 of the 70,770 switches into English, so the pooled 0.499 mostly reflects Cantonese. The average over pairs (0.634) is fairer to the other five.
- **Recall is what drops.** Going back into L1, the model catches 80–88% of switches. Going into English it catches 38–80%: the model hesitates rather than raising false alarms.
- **Korean is the exception.** Korean particles attach straight onto English words, so the piece itself often announces the coming switch.
- **Cantonese is the main problem.** It catches only 38% of its switches into English, and it is half the test data.

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

## Compared with no training

Simple baselines that need no model training, scored on the same test positions. The count-based ones fit their tables on the train split.

- **Always switch:** predict a switch at every piece.
- **Naive (current language):** predict a switch whenever the current piece is English, i.e. "English stretches end". One bit of context.
- **Run-length hazard:** P(switch | current language, how long the current stretch has run, the pair), with a val-tuned threshold. Duration: the most common class for that pair and direction.
- **Piece lookup:** P(switch | the current piece), a table with one probability per XLM-R piece, plus the most common duration class after that piece. This is the best you can do from the current piece alone.

**Switch F1:**

| Pair | Always switch | Naive | Run-length hazard | Piece lookup | **XLM-R** | XLM-R − lookup |
|---|---:|---:|---:|---:|---:|---:|
| Korean | 0.173 | 0.177 | 0.265 | 0.610 | **0.813** | +0.203 |
| Japanese | 0.161 | 0.156 | 0.244 | 0.495 | **0.741** | +0.246 |
| Chinese | 0.175 | 0.161 | 0.242 | 0.399 | **0.702** | +0.303 |
| Arabic | 0.174 | 0.175 | 0.246 | 0.390 | **0.669** | +0.279 |
| Russian | 0.172 | 0.183 | 0.245 | 0.391 | **0.666** | +0.275 |
| Cantonese | 0.180 | 0.244 | 0.251 | 0.417 | **0.625** | +0.208 |
| **All 6, pooled** | 0.178 | 0.225 | 0.251 | 0.427 | **0.645** | +0.218 |
| **Average over pairs** | 0.173 | 0.183 | 0.249 | 0.450 | **0.703** | +0.253 |

**Duration macro-F1** (at true switch points; XLM-R val-tuned):

| Pair | Most common class | Piece lookup | **XLM-R** | XLM-R − lookup |
|---|---:|---:|---:|---:|
| Korean | 0.263 | 0.431 | **0.597** | +0.166 |
| Japanese | 0.271 | 0.464 | **0.623** | +0.159 |
| Chinese | 0.261 | 0.405 | **0.607** | +0.202 |
| Arabic | 0.254 | 0.400 | **0.596** | +0.196 |
| Russian | 0.254 | 0.396 | **0.593** | +0.197 |
| Cantonese | 0.388 | 0.479 | **0.598** | +0.119 |
| **All 6, pooled** | 0.390 | 0.473 | **0.602** | +0.129 |

**How much did fine-tuning help?**
- **Switch: from 0.25 to 0.70 average F1 against the run-length baseline, and +0.25 over the piece lookup.** The lookup shows how much a single piece gives away (Korean particles make it 0.610 on its own); everything XLM-R adds on top comes from reading the context before it. The gain is largest on Chinese, Arabic and Russian (+0.28 to +0.30), where the current piece says least.
- **Duration: +0.12 to +0.20 macro-F1 over the piece lookup.** The baselines barely predict Medium (lookup Medium F1 0.11 pooled; the most-common-class baseline never predicts it). XLM-R reaches 0.41.
- **Accuracy would hide all of this:** always predicting "no switch" scores about 0.90 accuracy and F1 0.
- **We have no "XLM-R without fine-tuning" number.** The pretrained XLM-R has no switch or duration head, so before fine-tuning its predictions would be random. The baselines above are the meaningful no-training comparison.

## Per-pair detail

Each pair has two confusion matrices: switch on all test positions, and duration (Small / Medium / Large) at true switch points. Precision and recall in the duration table are per class.

**Reading the figures:** rows are the actual class, columns the prediction. Each cell shows its count and its share of the row, so the diagonal is the recall of each class. Colour encodes that share, not the raw count, on one 0–100% scale shared by every panel. Raw counts would make a useless colour scale, because "no switch, predicted no switch" is about 50× every other cell. The figures come from `scripts/plot_confusion.py data/anticipation_xlmr_6pairs.json --out docs/figures/team_meeting1`.

### Korean → English

![Korean → English confusion matrices](figures/team_meeting1/korean.png)

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

![Japanese → English confusion matrices](figures/team_meeting1/japanese.png)

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

![Chinese → English confusion matrices](figures/team_meeting1/chinese.png)

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

![Arabic → English confusion matrices](figures/team_meeting1/arabic.png)

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

![Russian → English confusion matrices](figures/team_meeting1/russian.png)

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

![Cantonese → English confusion matrices](figures/team_meeting1/cantonese.png)

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

- **Fine-tuning earns its keep.** Average switch F1 goes from 0.450 for the best no-training baseline (piece lookup) to 0.703, and duration macro-F1 from 0.473 to 0.602. The gain is the context: the lookup already knows everything the current piece gives away.
- **Switching into English is the weak direction** on 5 of 6 pairs (average over pairs 0.634 vs 0.775 back into L1). The gap ranges from −0.06 (Japanese) to −0.34 (Cantonese), and Korean is the exception (+0.02). This is where improvements matter most.
- **Switch F1 ranges from 0.625 (Cantonese) to 0.813 (Korean).** Korean particles attach directly to English words, so a switch is easy to anticipate. Cantonese switches mid-clause, rarely at punctuation, and is half the test data.
- **Duration: Large is easy (F1 0.78–0.89), Medium is hard (0.27–0.41).** Medium is mostly confused with its neighbours, and 91–95% of predictions are within one class. Medium is weakest on Korean and Japanese, where only 12.5% and 13.5% of switches are Medium.
- **Class-weighted duration training changed neither the switch scores nor, after val tuning, the duration scores** (see the full record). We use the plain model plus tuning.
