# Sub-word token labels — design

Date: 2026-09-27 · Milestone: Week 4 ("sub-word tokenization") · Status: approved in chat, awaiting spec review

## Intent

- **Stated by the user.** Break the labelled corpus down to sub-word tokens and label them; this is labelling only, with no model or training. It uses a tokenizer that works on all 15 pairs, and each token takes the language of the word it sits in. Switch and duration labels are then built **per token**. The course requirement defines duration as a 3-class problem: *Small 1–2, Medium 3–6, Large 7+ tokens*. The stored label is the **integer** run length, from which the classes are derived.
- **Assumed.** The Week 5 pretrained backbone is XLM-R, so the tokenizer must be XLM-R's.
- **Success.**
  - Every piece has a language, and every scored piece has labels that follow deterministically from the final word labels.
  - The labels are stored in `data/subword.parquet` next to readable piece strings.
  - Existing models and scripts keep working.
  - All tests pass.

## Evidence behind the decisions

Measured on 300 rows per pair (scratch probe, not kept):

| Tokenizer | Problem on our corpus |
|---|---|
| **XLM-R** (250 K SentencePiece) | none: 0% UNK, 0% byte fallback on all 15 pairs, fertility 0.93–1.66 |
| mBERT | 2.8% UNK on Cantonese |
| XGLM | 4.8% UNK on Cantonese |
| BLOOM | 45% byte pieces on Korean; fertility 2.46 on Russian |
| mGPT | 19% byte pieces on Cantonese |
| Qwen2.5 | 25% byte pieces on Cantonese, 23% on Hindi |

- **No tokenizer produced a piece spanning two languages.** The worst case was 0.01%.
- **XLM-R merges characters in CJK.** 24–51% of its CJK pieces span more than one character: Cantonese 24.4%, Korean 24.8%, Japanese 45.5%, Chinese 50.7%. So pieces can't simply be laid over the per-character units. Pieces become the units instead.
- **The original text aligns with the stored tokens.** On 3,000 random rows, re-tokenizing the CSV text with `tokenize_mixed` reproduced the stored tokens exactly, and every token was found in order in its utterance.

## Design

### 1. Sub-word labels

1. **Text.** Take the original utterances (`data_generation_result` in the cached SwitchLingua CSVs, joined on `row_id`). Tokenize each utterance with the `xlm-roberta-base` fast tokenizer, with offsets and no special tokens. A conversation's utterances are concatenated, as in the word pipeline (`cross_utterance=True`).
2. **Word spans.** Align the stored `tokens` to character spans by an in-order search of each utterance. Each word's language is its final `token_lang`, after `resolve_names(..., DEFAULT_NAMES)` and `absorb_other`.
3. **Piece language.** A piece takes the language of the word characters it overlaps.
   - **Punctuation and digits.** A piece overlapping only non-word tokens (`is_word` false) gets `NONE`.
   - **Word plus punctuation.** A piece overlapping a word and punctuation takes the word's language.
   - **Mixed pieces.** A piece overlapping words of two languages takes the majority language by characters (ties go to L1). It is counted in `mixed_pieces`.
4. **Letterless pieces (punctuation, digits).** `NONE` pieces stay in the input but are removed from the label timeline. They neither break a run nor count toward it, the same rule as `anticipation.is_word` in the word pipeline. This is unrelated to the `"transparent"` *names policy*, which is not used.
   - **Names are never `NONE` for being names.** Name words get their language at the word level from `resolve_names(..., "script_words")` (script in non-Latin rows; the words themselves when the dictionary margin is clear; otherwise the neighbouring language). Every piece of a name inherits that language. Examples: `Leo Messi` in a Spanish clause → all pieces Spanish; `Severance Season 2` in a Spanish row → `Severance`/`Season` pieces English, and `2` is `NONE` as a digit.
5. **Labels on the language-bearing pieces**, indexed back onto the full piece sequence:
   - `y_sw[t]` = 1 if the next language-bearing piece is in the other language. Undefined (-1) on the last one and on `NONE` pieces.
   - `y_len[t]`: at a switch, the integer length in language-bearing pieces of the run that starts at the next piece.
   - `censored[t]`: that run is still open at the end of the row.
   - `dur_class(len)`: 0 Small (1–2), 1 Medium (3–6), 2 Large (7+).
   - `dur_known` = switch ∧ (¬censored ∨ `y_len` ≥ 7). A cut-off run is used only when its class is already certain.
6. **Mid-word pieces** get `y_sw = 0`. The model has to learn where words end. This is not a leak, because its inputs are pieces ≤ t only (see §3).

### 2. The same labels for the word pipeline

`anticipation.anticipatory_labels` returns integer `y_len`, `dur_class` and `dur_known` in place of the 7-bucket `y_dur`. The word and sub-word streams share this code path. `DUR_BUCKETS`/`DUR_LABELS` become the 3 classes. The report's 7-bucket duration numbers become historical until they're rerun.

### 3. Components

| Unit | Responsibility |
|---|---|
| `src/subword.py` | `align_words(utterances, tokens)` → char spans. `piece_languages(...)` → per-piece language and mixed count. `subword_row(...)` → `piece_ids`, `pieces`, `piece_lang`. `build_frame` adds the labels from `anticipation.anticipatory_labels`. |
| `src/anticipation.py` | Label building shared by both units: integer lengths, classes, `dur_known`. |
| `scripts/build_subword.py` | Reads the CSVs and `data/tagged.parquet`; writes `data/subword.parquet` in shuffled row order (seed 0). Columns: `row_id`, `first_language`, `second_language`, `quality`, `pieces` (str), `piece_ids` (int32), `piece_lang` (int8: 0 L1, 1 L2, 2 NONE), `y_sw` (int8), `y_len` (int32), `y_dur` (int8), `censored`, `dur_known` (bool), `mixed_pieces`. Prints alignment-failure, mixed-piece, switch and duration-class counts. |
| `src/causal_model.py`, `scripts/eval_anticipation_pair.py` | Only so existing models keep loading after the switch to 3 classes: an `n_dur` parameter, read from the checkpoint. |

### 4. Errors and data checks

- **Alignment failure** (a token not found in order): the row is skipped and counted. The build fails if more than 0.1% of rows fail, since the probe found 0 of 3,000.
- **Mixed pieces** are counted and reported. They are not an error.
- **Rows with fewer than 2 language-bearing pieces** are kept, with every label -1.

### 5. Tests (written first)

- **`tests/test_subword.py`:**
  - word alignment on repeated substrings
  - CJK merged pieces take the character language
  - punctuation pieces are `NONE`
  - a word-plus-punctuation piece takes the word's language
  - majority rule on a mixed-piece fixture
  - names follow `script_words`: fixtures from `tests/fixtures/name_cases.json` give every piece of a name its word's resolved language, and no name piece is `NONE` unless it has no letters
- **Labels:**
  - `NONE` is transparent
  - `y_len` integers
  - class cut-offs 2|3 and 6|7
  - a censored run of 7+ is known, a censored run below 7 is not
  - the final piece is unlabelled
- **Regression:** word-level `y_sw`, run lengths and censoring match the previous implementation, and a 7-class checkpoint still loads.
- **Build:** a round trip through the build script on a 20-row fixture; the stored labels equal `anticipatory_labels(piece_lang)`.

### 6. Deliverables

1. `data/subword.parquet` (gitignored).
2. The build log's counts in `groupUpdate1.md` §5.1; `TIMELINE.md` Week 4 updated; `CLAUDE.md` commands and architecture notes.

## Out of scope

- **Deferred to the Week 5 model work:**
  - sub-word streaming and a compact vocabulary
  - model inputs (no language input on sub-words, since a word's label depends on pieces not yet read; `--no-lang` for a fair word-model comparison)
  - `--unit` script flags and 3-class duration metrics
  - all training runs and results
- CJK word segmentation (jieba/fugashi).
- The Week 5 XLM-R backbone itself.
- Random-crop augmentation.
- A per-token subword CSV on Hugging Face.
