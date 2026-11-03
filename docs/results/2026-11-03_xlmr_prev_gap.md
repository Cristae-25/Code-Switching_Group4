# 2026-11-03: Deciding after the punctuation (`--decide-at gap`) on XLM-R (Kaggle T4)

**Run:** `kaggle/train_pretrained.ipynb`, TAG `xlmr_prev_gap`. Seed 0, 12,000 steps (1.02 epochs), batch 16, window 384 / stride 288, lr 2e-5 for the body and 1e-3 for the heads, all 15 pairs, `PREV_LANG=True`, `DECIDE_AT="gap"`. Training took 77 min. The causality check passed (drift 0.00e+00). The log confirms the flag: `decide_at=gap: 3 blank piece ids [6, 244871, 248618]`.

**Reference:** `xlmr_prev` (2026-11-02), which has the same settings with `--decide-at word`.

## What the flag changes

A label normally sits on the last piece of the word before the next word, so in `home , 然后` the model decides at `home` without having read the comma. With `--decide-at gap`, `anticipation.gap_landing` moves the label to the last piece of the gap that is not a bare space marker. The model then decides after reading the punctuation and still before the next word.

The labels themselves don't change. The switches (`y_sw`), the durations and the number of scored positions all stay the same. Only 12.9% of positions move (236,293 of 1,825,011), and the `n` and `base` columns of every report table match `xlmr_prev` exactly. The bare `▁` is never a landing piece: XLM-R emits it only before characters that don't merge with the marker (CJK), so landing on it would reveal the next word's script.

## A bug in the first attempt

The first upload of this TAG (07:54, kept as `data/BUGGED_*` and `logs/BUGGED_train_xlmr_prev_gap.log`) is **not** a gap run. The notebook added `_gap` to the TAG, but it never passed `--decide-at` to the training command, so the script fell back to `word`. That run is a second seed-0 `xlmr_prev`: switch F1 0.647 vs 0.646, paired difference +0.001 [−0.000, +0.001]. The builder now passes `--decide-at {DECIDE_AT}`.

## Results (test)

| Run | Switch F1 | Macro-F1 (pairs) | Into L2 | Into L1 | Duration macro-F1 (val-tuned) | End-to-end acc | τ |
|---|---:|---:|---:|---:|---:|---:|---:|
| `xlmr_prev` (decide at word) | 0.646 | 0.677 | 0.536 | 0.760 | 0.574 (0.606) | 0.477 | 0.332 |
| `xlmr_prev_gap` (decide at gap) | **0.744** | **0.794** | **0.634** | **0.858** | **0.596 (0.624)** | **0.522** | 0.386 |

Paired row bootstrap, `xlmr_prev_gap − xlmr_prev` (`scripts/bootstrap_preds.py xlmr_prev xlmr_prev_gap`):
- **switch F1:** +0.098 [+0.096, +0.100]
- **duration macro-F1:** +0.022 [+0.020, +0.024]
- **Medium F1:** +0.024 [+0.020, +0.028]
- **end-to-end accuracy:** +0.045 [+0.043, +0.046]

The gain isn't a threshold effect: with the best possible threshold, F1 is 0.647 vs 0.744.

Per pair (switch F1, gap run): Korean 0.896, Japanese 0.872, French 0.859, German→English 0.842, Chinese 0.837, Spanish 0.827, Arabic 0.825, Italian 0.825, Russian 0.778, German→French 0.772, Vietnamese 0.760, Hindi 0.759, Cantonese 0.709, Philippines 0.702, Malay 0.652. Every pair improves. Hindi (+0.043) and Philippines (+0.063), whose switches are mostly mid-clause, gain least. Cantonese gains +0.090, Malay +0.104, and the European and Arabic pairs +0.11 to +0.16 (French +0.160, Arabic +0.161).

## Where the gain comes from

`scripts/gap_split.py xlmr_prev xlmr_prev_gap`:

| Positions | n | Switch share | Word P / R / F1 | Gap P / R / F1 | Average precision (word → gap) |
|---|---:|---:|---|---|---|
| Unmoved (no punctuation before the next word) | 1,588,718 | 0.049 | 0.425 / 0.519 / 0.467 | 0.644 / 0.459 / 0.536 | 0.485 → 0.584 |
| Moved (word run: on the word; gap run: on the punctuation) | 236,293 | 0.474 | 0.799 / 0.773 / 0.786 | 0.849 / 0.883 / 0.866 | 0.865 → 0.941 |
| All | 1,825,011 | 0.104 | 0.625 / 0.669 / 0.646 | 0.783 / 0.709 / 0.744 | 0.709 → 0.827 |

1. **Moved positions: the model reads the punctuation.** 59.1% of switches sit behind punctuation (as `punct_breakdown.py` found), and reading the `,` `。` `?` before deciding is worth +0.08 F1 there. This is causal: the decision still comes before the next word. The cost is that the warning comes one piece later.
2. **Unmoved positions: a change in what is scored.** These are the same positions, with the same inputs and the same labels, but precision rises from 0.425 to 0.644. In word mode, the model had to hedge at every word end for punctuation it couldn't see yet, and that hedging produced false positives at clause-final words with no punctuation after them. In gap mode, a word followed by punctuation is never scored, so the model learns, and is tested on, only word ends where the next piece is a word. Which positions get scored depends on the next piece. Word mode also peeks one piece ahead to find where a word ends, but gap mode's peek also reveals whether punctuation is coming, which is a strong switch cue. This part of the gain belongs to the new task definition, not to a better model.
3. **The model beats a simple rule.** "Predict a switch exactly when the label sits on punctuation" scores P 0.474 / R 0.591 / **F1 0.526**. That is the simplest baseline at the gap decision points, and the model is +0.22 above it.

## Decision

**We are now deciding at the gap.** Each switch is predicted just before the next word, after any punctuation the writer has typed. Recorded in `docs/groupUpdate2.md`, §1. The notebook default is now `DECIDE_AT = "gap"`; `train_anticipation.py` still defaults to `word`, so earlier runs reproduce unchanged.

Consequences:
- **Don't compare with Update #1's numbers as if they were like for like.** 0.744 (gap) and 0.646 (word) answer different questions: the gap run decides later, and is scored on a different set of positions.
- **Every later run uses `--decide-at gap`**, so flag ablations (LoRA, large backbone, longer training) are compared against `xlmr_prev_gap`.
- **The training-free baselines need recomputing at the gap positions.** The Update #1 baselines were scored at word ends.

## Reproducing

```bash
./.venv/bin/python scripts/bootstrap_preds.py xlmr_prev xlmr_prev_gap
./.venv/bin/python scripts/gap_split.py xlmr_prev xlmr_prev_gap
```
