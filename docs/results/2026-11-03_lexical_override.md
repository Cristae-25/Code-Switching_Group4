# 2026-11-03: Lexical override on AnE-LID (labels version 2)

**What changed:** `src/lexical_override.py`, applied by `scripts/apply_names_policy.py`, which is on by default; `--no-lexical-override` reproduces version 1. It also runs inside `dispatch.tag_tokens` for any future re-tag. Version-1 files are kept in `data/v1final/`.

## Why

AnE-LID was chosen on LinCE spa-eng / hin-eng and Denglisch. No human-annotated set exists for Malay, Vietnamese or Tagalog, so it was never measured on them. Two signals, with no hand labels, showed it failing there:

- **The tagger consensus** (`consensus_corpus.py`): agreement on the words where both name a language was 95.8% for Malay and 99.0% for Vietnamese.
- **The trained model** (`consensus_f1.py xlmr_prev_gap`): where the taggers disagreed, 45% (Malay) and 40% (Vietnamese) of positions were labelled as switches, against about 11% elsewhere. The model predicted almost none of them (recall 0.18 / 0.16). The disagreements were AnE calling L1 function words English: Malay `akan` ×1,496, `memang`, `dan`, `yang`, `dengan`, `saya`; Vietnamese `và`, `sẽ`, `của`, `nhưng`. It also tagged whole spans English, such as "Kementerian/E … akan/E bekerjasama/E dengan/E agensi/E lain/E untuk/E".

## The rule

For word tokens in `ane` rows with English L2, excluding romanized Hindi:

1. **Vietnamese letters:** in a Vietnamese row, a word with `đ ă ơ ư` or a stacked tone mark (U+1EA0–U+1EF9) is Vietnamese.
2. **Frequency margin:** the label flips to the pair's other language when the word's Zipf score there is ≥ 4.0 and ≥ 2.0 above its score in the labelled language. A 2.0 margin means 100× more frequent.

Never touched: names (`is_entity`), all-caps acronyms, punctuation, romanized Hindi (wordfreq's Hindi list is in Devanagari script), German→French (already tagged by dict-freq) and the script-tagged pairs. Each flip is recorded in the per-token `lexical_override` column. Tests: `tests/test_lexical_override.py` (11).

## What it changed

20,075 tokens in 6,127 rows:

| Pair | Tokens | Share of words | Switches (v1 → v2) | L2 tokens |
|---|---:|---:|---|---:|
| Malay | 14,809 | 2.85% | 40,444 → 38,505 (−4.8%) | −8.0% |
| Vietnamese | 4,209 | 0.66% | 34,168 → 33,764 (−1.2%) | −2.4% |
| Tagalog | 792 | 0.14% | 56,507 → 56,805 (+0.5%) | −0.3% |
| French | 197 | 0.04% | 26,272 → 26,206 | −0.1% |
| Italian / German / Spanish | 36 / 17 / 15 | ≤ 0.01% | unchanged | 0.0% |

Malay switches fall by less than the flip count suggests. Many flipped words sat next to genuine English words ("akan contact supplier"), so fixing them shortens a run rather than removing a switch.

Sub-word training labels: scored positions unchanged (18,211,198); switches 1,896,074 → 1,892,220.

## Validation

1. **The rebuild with the override off reproduces version 1 exactly:** every column, all 234,172 rows. So the override is the only change.
2. **Human-annotated sets** (`eval_public.py`, column `AnE+override`):

   | Set | Tokens | AnE | AnE + override |
   |---|---:|---:|---:|
   | LinCE spa-eng dev | 31,667 | 99.4% | 99.4% |
   | LinCE hin-eng dev | 12,303 | 98.8% | 98.8% (out of scope) |
   | Denglisch | 59,485 | 99.9% | 99.9% |

   The rule fires on 17 tokens of LinCE spa-eng and 20 of Denglisch (`logs/eval_public_override.log`), and accuracy is unchanged to 0.1%.
3. **Consensus after the fix:** Malay 95.8% → 99.1%, Vietnamese 99.0% → 99.8%, Tagalog 98.6% → 98.8%. This is **partly circular**: on the flipped tokens, the labels now follow dict-freq.
4. **Spot-check of random flips in context:** correct (`yang`, `dengan`, `untuk`, `và`, `của`, `sa`, `parang`, `matchs`, `stade`). The one error class found, acronyms (`AI` → French ×55), is now excluded.
5. **The same model, re-scored on the new labels without retraining** (`scripts/rescore_preds.py xlmr_prev_gap data/subword.parquet`):

   | Pair | F1, v1 labels | F1, v2 labels |
   |---|---:|---:|
   | Malay | 0.652 | **0.703** |
   | Vietnamese | 0.760 | **0.777** |
   | Tagalog | 0.702 | 0.705 |
   | every other pair | | within 0.001 |
   | All | 0.744 | 0.747 |

   2,958 of 1,825,011 test labels change. The model was trained on the v1 labels, so retraining on v2 should help these pairs further.

## Consequences

- **Results on v1 and v2 labels are not comparable.** Every run up to and including `xlmr_prev_gap`, plus the two Kaggle runs started on 2026-11-03 (`xlmr_prev_gap_36k_rs`, `xlmr_prev_lora16_gap`), uses v1.
- **Uploaded** to `CodeSwitchGroup4/switchlingua-tagged-final` as commit `6af7d7ad31` (v1 stays at `5a12e6a842`). A download matches the local file byte for byte.
- **Retrain the reference** `xlmr_prev_gap` on v2. The Kaggle notebook reads `CodeSwitchGroup4/switchlingua-tagged-final`, so training only sees v2 after the upload.
