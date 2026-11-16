# Current-token-only XLM-R on the 6 script-disjoint pairs (2026-10-06)

**Question:** how much of a switch can be anticipated from piece t *alone*, with no prefix? The 6 pairs (Cantonese, Chinese, Japanese, Korean, Russian, Arabic → English) are the `unicode`-routed ones, so their labels are exact and any gap between models comes from the models. The ladder per pair is `naive_l2` → piece-id lookup → **XLM-R, current piece only** (`--context 1`) → XLM-R, full prefix.

## How prefix-only context is enforced

- **The model.** `CausalPretrained(context=1)` encodes every piece as its own length-1 sequence, so the prediction at t reads piece t and nothing else (not even the past), and every piece has the same position. `check_context_free` (`src/causal_model.py`) rewrites every *other* position, past and future, and asserts that no logit at t moves. It runs before training, alongside `check_causality`, which a current-token model passes trivially. The check fails on the full-prefix model, which is the test that it can detect a leak (`tests/test_xlmr_model.py`).
- **Side inputs.** `--context 1` refuses `--prev-lang` and `--turn-sep` (extra context) and `--predict-at word_end` (scoring only word ends tells the model the next piece starts a new word). There is no pair embedding.
- **Bare `▁` pieces** carry the *next* character's offsets. They are NONE, so they are never scored (pinned in `tests/test_subword.py`).
- **Tokenization.** The pieces come from tokenizing whole utterances; a streaming model only has the prefix. `scripts/labelling/audit_prefix_tokenization.py` re-tokenizes `text[:end of piece]` for every scored piece:

  | Pair | Scored pieces (val + test) | Last piece differs | Some prefix piece differs |
  |---|---:|---:|---:|
  | Cantonese | 2,121,423 | 0 | 0 |
  | Chinese | 121,967 | 0 | 0 |
  | Japanese | 129,677 | 0 | 0 |
  | Korean | 110,068 | 0 | 0 |
  | Russian | 98,055 | 0 | 0 |
  | Arabic | 103,223 | 0 | 0 |

  **There is no tokenization leak for XLM-R, and the reason is structural.** XLM-R uses unigram SentencePiece, which decodes by Viterbi. The best segmentation of the whole text, cut at a piece boundary, is also the best segmentation of that prefix, so the prefix always reproduces the stored pieces (up to ties). The same holds for the full-prefix XLM-R. BPE (RoBERTa) has no such guarantee; run the audit with `--tokenizer roberta-base --path data/subword_roberta.parquet` before relying on it there.

## Training-free results (test split, the 6 pairs only, local)

`./.venv/bin/python scripts/training/baseline_anticipation.py --unit subword --langs Cantonese,Chinese,Japanese,Korean,Russian,Arabic --json data/results/baseline_anticipation_subword_6pairs.json` (28 s; 10.7 M train positions, 1.34 M test, base rate 0.098). Thresholds tuned on val.

| Model | P | R | F1 | Macro-F1 | F1 into L2 | F1 into L1 |
|---|---:|---:|---:|---:|---:|---:|
| `naive_l2` | 0.149 | 0.461 | 0.225 | 0.183 | 0.000 | 0.259 |
| hazard, per pair | 0.175 | 0.443 | 0.251 | 0.249 | 0.144 | 0.291 |
| **piece lookup** P(switch \| piece id) | 0.403 | 0.455 | **0.427** | **0.450** | 0.253 | 0.547 |

| Pair | hazard, per pair | piece lookup | XLM-R full prefix (trained on all 15) |
|---|---:|---:|---:|
| Korean | 0.265 | 0.610 | 0.808 |
| Japanese | 0.244 | 0.495 | 0.736 |
| Cantonese | 0.251 | 0.417 | 0.619 |
| Chinese | 0.242 | 0.399 | 0.699 |
| Russian | 0.245 | 0.391 | 0.659 |
| Arabic | 0.246 | 0.390 | 0.664 |

The piece id alone carries far more than the one bit `naive_l2` uses: 0.427 against 0.225. Most of it is switching *back* into L1 (0.547). Some English pieces (sub-word stems, words that close an insertion) are reliably followed by L1, and Korean particles attach to English stems. The lookup table is the Bayes-optimal current-token classifier on the training data. The current-token XLM-R can only beat it where its pretrained embedding generalises to rare or unseen pieces.

## Still to run

The XLM-R current-token model itself: Kaggle, `CONTEXT = 1`, `PAIRS = "Cantonese,Chinese,Japanese,Korean,Russian,Arabic"` (TAG `xlmr_ctx1_6pairs`). Sanity bounds: F1 ≥ the piece lookup's 0.427 on most pairs, and well below the full prefix. A result close to the full prefix would mean a leak.

**Confound:** the full-prefix XLM-R numbers above come from training on all 15 pairs. A matched full-prefix run on the same 6 pairs (`PAIRS` set, `CONTEXT = None`, about 1 h) makes the context gap exact.
