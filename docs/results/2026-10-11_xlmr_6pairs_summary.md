# XLM-R on the 6 script-disjoint pairs: results and how they are measured (2026-10-11)

A summary for the team. It pulls the 6 script-disjoint pairs (Cantonese, Chinese, Japanese, Korean, Russian, Arabic → English) out of the 2026-09-29 XLM-R run, then explains what the model sees, what it is scored on, and how the future is kept out. No new training was run; every number comes from [`2026-09-29_xlmr_causal.json`](2026-09-29_xlmr_causal.json) and [`data/results/baseline_anticipation_subword_6pairs.json`](../../data/results/baseline_anticipation_subword_6pairs.json).

**Why these 6:** they are `unicode`-routed, so each piece's language comes from its script, and the labels are exact by construction. Any error on them comes from the model, not the tagger.

**Caveat:** this XLM-R model was trained on all 15 pairs, not just these 6. The matched 6-pair run (`xlmr_6pairs`) and the current-token run (`xlmr_ctx1_6pairs`) are queued.

## 1. Switch prediction (`y_sw`), per pair, test split

The question at each piece t: will piece t+1 be in a different language from piece t? The threshold τ = 0.337 was tuned once on val (all pairs pooled) and applied to test unchanged.

| Pair | Test positions | True switches | Base rate | **F1** | Precision (95% CI) | Recall (95% CI) | Run 2 F1 |
|---|---:|---:|---:|---:|---|---|---:|
| Korean | 55,454 | 5,240 | 9.4% | **0.808** | 0.775 (0.764–0.786) | 0.844 (0.834–0.854) | 0.809 |
| Japanese | 64,510 | 5,658 | 8.8% | **0.736** | 0.694 (0.682–0.705) | 0.783 (0.772–0.794) | 0.734 |
| Chinese | 62,016 | 5,942 | 9.6% | **0.699** | 0.647 (0.635–0.658) | 0.761 (0.750–0.772) | 0.698 |
| Arabic | 51,395 | 4,908 | 9.5% | **0.664** | 0.617 (0.604–0.630) | 0.717 (0.705–0.730) | 0.667 |
| Russian | 49,679 | 4,677 | 9.4% | **0.659** | 0.631 (0.617–0.644) | 0.690 (0.676–0.703) | 0.661 |
| Cantonese | 1,061,851 | 104,881 | 9.9% | **0.619** | 0.621 (0.618–0.624) | 0.618 (0.615–0.621) | 0.621 |
| **Macro (6 pairs)** | | | | **0.698** | | | |

Against the training-free baselines on the same test positions (switch F1):

| Pair | Always switch | `naive_l2` | Piece lookup P(switch \| piece id) | **XLM-R, full prefix** | XLM-R gain over lookup |
|---|---:|---:|---:|---:|---:|
| Korean | 0.173 | 0.177 | 0.610 | **0.808** | +0.198 |
| Japanese | 0.161 | 0.156 | 0.495 | **0.736** | +0.241 |
| Chinese | 0.175 | 0.161 | 0.399 | **0.699** | +0.300 |
| Arabic | 0.174 | 0.175 | 0.390 | **0.664** | +0.274 |
| Russian | 0.172 | 0.183 | 0.391 | **0.659** | +0.268 |
| Cantonese | 0.180 | 0.244 | 0.417 | **0.619** | +0.202 |

**Reading.**
- XLM-R is about 3.4–4.7× the always-switch F1 on every pair, and +0.20 to +0.30 above the best current-piece-only baseline. That gain is what the prefix adds.
- **Korean is easiest (0.808).** Korean particles attach directly to English stems, so the piece id already tells a lot (lookup 0.610).
- **Cantonese is hardest (0.619), and it is half the test set.** It switches mid-clause, rarely at punctuation, and is the only pair where precision ≈ recall. It is also why the pooled F1 over all 15 pairs (0.646) sits below the macro-F1 (0.676).
- **The two independent trainings agree within ±0.003 on these pairs.** With Wilson intervals of about ±0.01 (±0.003 for Cantonese), the ranking Korean > Japanese > Chinese > {Arabic, Russian} > Cantonese is stable; Arabic and Russian are tied.

## 2. Duration (`y_dur`)

The question at each true switch point: how long will the new run be? The classes are Small (1–2 pieces), Medium (3–6) and Large (7+).

**Per-pair duration is not available for this run.** `src/evaluation.full_report` writes duration pooled over all 15 pairs, and the Kaggle run predates saving predictions, so it cannot be split by pair after the fact. The pooled numbers (n = 180,990 switch points, all 15 pairs):

| Metric | XLM-R | Mode per direction (training-free) |
|---|---:|---:|
| 3-class accuracy | **0.678** | 0.509 |
| Macro-F1 | **0.575** | 0.225 |
| Within one class | **0.910** | 0.732 |
| F1 Small / Medium / Large | 0.636 / 0.278 / 0.813 | — |

Medium is the weak class: most Medium runs are predicted Large (18,216) or Small (14,015). Class-weighted duration loss (`DUR_WEIGHTS="balanced"`) raised Medium F1 from 0.075 to 0.295 in the local ablations ([record](2026-09-30_ablations.md)).

**Per-pair duration results:** see [2026-10-12_xlmr_6pairs_duration.md](2026-10-12_xlmr_6pairs_duration.md). XLM-R trained on these 6 pairs reaches duration macro-F1 0.602 after val tuning (Medium 0.406), with 0.593–0.623 per pair.

## 3. What information is used to train and evaluate

**Data:** the SwitchLingua conversations for these 6 pairs, from `data/subword.parquet`. Each row is one conversation, tokenized by the `xlm-roberta-base` tokenizer into sub-word pieces.

**The model's input:** only the sequence of XLM-R piece ids for the row, read left to right. There is:
- no language tag,
- no pair or language-ID embedding (the model is never told which pair it is reading),
- no punctuation-derived feature,
- and no special tokens.

**The labels**, derived from the exact per-piece language (each piece takes the language of the word its characters fall in), binarised as L1/L2 relative to the row's pair:

| Label | Defined at piece t as | Scored where |
|---|---|---|
| `y_sw[t]` | 1 if the next language-bearing piece has a different language from piece t, else 0 | every language-bearing piece except the last in the row |
| `y_dur[t]` | the length class of the run that starts at t+1 | only where `y_sw[t] = 1` (a true switch) **and** the class is certain |

**Pieces that are never scored:**
- **NONE pieces:** punctuation, digits and bare `▁` pieces. They are transparent: they neither break nor extend a run, and are never scored.
- **The final piece of a row,** which has no t+1. It is masked out, not labelled 0; a 0 would hand out a free true negative per row.
- **A run cut off by the end of the text.** It is a duration target only if it already has 7+ pieces, because only then is its class certain.

**The split** is by row, never by token: `row_id % 10` gives 0–7 train, 8 val, 9 test. A conversation is entirely in one split.

**The protocol:**
1. Train on the train split.
2. Pick the switch threshold τ that maximises F1 on val.
3. Score test once with that τ.

The metrics are positive-class precision, recall and F1 with Wilson intervals, plus macro-F1 across pairs. Accuracy is not reported, because "never switch" already scores 90%.

**Duration is scored given the true switch:** the duration head is evaluated at every real switch point, whether or not the switch head fired there. So the duration numbers answer "if you knew a switch was coming, could you size it?" They are not end-to-end "predict a switch and its length" numbers.

## 4. Masking: what the model can and cannot see

**It is not one word at a time.** The model does not get the first word, answer, then get the first two words, and so on. That is what it is *equivalent* to, but it is done in one pass:

1. The whole conversation (up to 384 pieces) goes in at once.
2. A causal attention mask (`is_decoder=True` in Hugging Face, a lower-triangular mask in every self-attention layer) lets the hidden state at piece t attend only to pieces 0…t.
3. The switch and duration heads read that hidden state, so the prediction at t is a function of the prefix 0…t alone. It is exactly what you would get by feeding the prefix up to t on its own.
4. One forward pass yields a prediction at every position: n prefix-only predictions for the price of one pass.

Worked example, for a Korean→English row `나는 meeting 에 갔어` (pieces simplified):

| t | Visible to the model | Question asked | `y_sw` | `y_dur` |
|---|---|---|---:|---|
| 0 | `나는` | will the next piece switch? | 1 | Small (`meeting` is a 1-piece English run) |
| 1 | `나는 meeting` | will the next piece switch? | 1 | not scored: the Korean run `에 갔어` reaches the end of the text at 2 pieces, so its class is uncertain |
| 2 | `나는 meeting 에` | will the next piece switch? | 0 | not scored (no switch) |
| 3 | `나는 meeting 에 갔어` | — | masked | masked (last piece) |

At t = 0 the model has never seen `meeting`. The label at t uses piece t+1; the input at t never contains it.

**How the answers are kept from leaking:**

| Possible leak | How it is closed | How it is checked |
|---|---|---|
| Attention to future pieces | Causal mask in every XLM-R layer | `check_causality` rewrites every piece after t with random ids and asserts the logits at ≤ t do not move. It ran on the assembled model before training: **max drift 0.00** over 31 positions. |
| A label used as an input | No language input on sub-word pieces at all, because a mid-word piece's language depends on pieces not yet read | By construction (`use_lang=False`) |
| Normalisation across time | Only LayerNorm (per position), no BatchNorm | Covered by `check_causality` |
| Tokenization seeing the future | The pieces come from tokenizing the whole text. Re-tokenizing `text[:end of piece t]` alone reproduces every stored piece, because unigram SentencePiece is prefix-consistent. | `scripts/labelling/audit_prefix_tokenization.py`: **0 of 2.68 M** scored pieces differ on these 6 pairs ([record](2026-10-06_current_token_xlmr.md)) |
| Same conversation in train and test | Row-level split (`row_id % 10`) | By construction |
| Tuning on test | τ tuned on val only; test scored once | `scripts/training/train_anticipation.py` |

**Long conversations (windowing).** A row longer than 384 pieces is cut into overlapping windows with stride 288. In every window after the first, the first 96 pieces are context only: they are fed in but masked out of the loss and the scores. So every position is scored exactly once, always with at least 96 pieces of real prefix behind it. The visible prefix is capped at 384 pieces, not the whole conversation. That is a limit on how much past the model sees, not a leak from the future.

**What is still in the input, legitimately:** the row's pieces up to t. That includes the earlier turns of the same conversation and any punctuation already typed, both of which a real-time listener would also have.
