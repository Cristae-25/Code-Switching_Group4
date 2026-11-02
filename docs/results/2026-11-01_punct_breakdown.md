# 2026-11-01: Do switches sit behind punctuation? (local, no GPU)

**Why:** to test the idea that punctuation signals an upcoming switch. A switch label sits on the last language-bearing piece *before* any punctuation (`anticipatory_labels`, where NONE slots are transparent). In `I went home, 然后…` the decision is made at `home`, and the causal model has not seen the comma yet. So the punctuation right at a switch is currently invisible to the model.

**How:** `./.venv/bin/python scripts/punct_breakdown.py xlmr_6pairs` (9 s). Every labelled position is bucketed by what lies between it and the next language-bearing piece:
- `turn`: a new utterance starts
- `sentence_end`: . ! ? 。 ！ ？ …
- `comma`: , ; : 、 ， ；
- `other_gap`: digits, quotes, brackets
- `adjacent`: nothing in between

- **Part A** covers the labels on all splits.
- **Part B** scores the `xlmr_6pairs` test predictions at the run's own τ = 0.377. Its pooled F1 (0.6452) matches the run's json.

Output: `data/punct_breakdown_xlmr_6pairs.json`.

## Part A: where switches happen (all 15 pairs, 18.2 M positions)

| Bucket | Positions | Share of all switches | Switch rate |
|---|---:|---:|---:|
| adjacent (mid-clause) | 15.86 M | 0.412 | 0.049 |
| comma | 1.08 M | 0.184 | 0.324 |
| sentence_end | 0.35 M | 0.142 | 0.776 |
| other_gap | 0.20 M | 0.027 | 0.260 |
| turn | 0.73 M | 0.236 | 0.611 |

Wilson intervals are all narrower than ±0.002. **59% of switches sit behind a boundary,** and a sentence end is followed by a switch 78% of the time, against 5% mid-clause. Hindi and Philippines are the exceptions: 65–66% of their switches are mid-clause, and they have the highest mid-clause switch rate (0.13–0.15).

## Part B: how well XLM-R catches each kind (`xlmr_6pairs`, test, all scored positions)

| Bucket | n | Base rate | Precision | Recall | F1 | Missed switches |
|---|---:|---:|---:|---:|---:|---:|
| adjacent | 1,160,369 | 0.045 | 0.465 | 0.475 | **0.470** | **27,430** |
| comma | 83,873 | 0.268 | 0.617 | 0.630 | 0.623 | 8,337 |
| sentence_end | 26,743 | 0.756 | 0.915 | 0.751 | 0.824 | 5,047 |
| other_gap | 15,104 | 0.246 | 0.661 | 0.582 | 0.619 | 1,552 |
| turn | 58,816 | 0.554 | 0.884 | 0.824 | 0.853 | 5,724 |

The word-end (shared) positions give the same ranking: adjacent 0.659, comma 0.737, sentence_end 0.799, turn 0.920.

## Reading

- **Boundary switches are already mostly caught.** At sentence ends and turns, F1 is 0.82–0.85 without seeing the punctuation. The model seems to recognise that a clause is complete.
- **Commas are the gap punctuation could close:** F1 0.623 and 8.3 k missed switches. Cantonese is the weakest pair there (0.57), and it also has the lowest comma switch rate (0.24).
- **Mid-clause switches are the main bottleneck:** F1 0.470, and 57% of all missed switches (27.4 k of 48.1 k). Punctuation cannot help here by definition.
- **Ceiling on deciding at the punctuation piece:** if the comma and sentence-end buckets became perfect, pooled F1 would rise from 0.645 to about 0.74. A realistic gain is a fraction of that. Part of any gain would also come from the task getting easier (after a `.` the switch rate is 0.78), not from a better model. So a "decide at punctuation" variant must report its numbers next to the current decision points, not in place of them.
