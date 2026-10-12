# Which models labelled the data, and how accurate they are

Every token of SwitchLingua has a language label, and **no label was written by hand**. Each language pair is routed to the tagger that is most accurate for it (`src/routing.py`), and every tagger is scored automatically against human-annotated public data. For the full history of what was tried and rejected, see [`labelling_pipeline.md`](labelling_pipeline.md).

## Who labels what

| Rows | Tagger | How it decides | Accuracy |
|---|---|---|---|
| Cantonese, Chinese, Japanese, Korean, Russian, Arabic, and Devanagari-script Hindi (6 pairs + part of Hindi) | **Unicode script** (`src/unicode_tagger.py`) | The script *is* the language: Han/Kana/Hangul/Cyrillic/Arabic/Devanagari → first language, Latin → English | exact by construction (`quality = "exact"`) |
| Latin-script first languages with English as L2, including romanized Hindi (Spanish, French, Italian, German, Vietnamese, Malay, Tagalog, Hindi) | **AnE-LID** ([`igorsterner/AnE-LID`](https://huggingface.co/igorsterner/AnE-LID), XLM-R-large) | A token classifier trained on code-switched text: English vs. not-English, using the whole sentence as context | 99.4% on LinCE Spanish–English, 98.8% on LinCE Hindi–English |
| German → French | **dict-freq** (`src/dict_tagger.py`, [`wordfreq`](https://github.com/rspeer/wordfreq)) | For each word, compares its frequency in German and in French; the more frequent wins | no public German–French set exists; 96.2% on Denglisch (German–English) |
| Names, in every row | **AnE-NER** ([`igorsterner/AnE-NER`](https://huggingface.co/igorsterner/AnE-NER)) | Marks name tokens (`is_entity`); the name still gets a language, see below | spot-checked on 20 edge cases (`tests/test_name_strategies.py`) |

Rows labelled by AnE-LID or dict-freq carry `quality = "provisional"`, since their labels are model output rather than exact.

## Scores on human-annotated data

Measured by `scripts/eval_public.py` on sets that people labelled word by word:

| Set | Words | dict-freq | mBERT (retired) | **AnE-LID (used)** |
|---|---:|---:|---:|---:|
| LinCE Spanish–English (dev) | 31,667 | 94.1% | 98.8% | **99.4%** |
| LinCE Hindi–English (dev) | 12,303 | 80.5% | 97.4% | **98.8%** |
| Denglisch (German–English) | 59,485 | 96.2% | — | 99.9%* |

\* AnE-LID saw Denglisch in training, so treat 99.9% as an upper bound.

**Coverage on our own corpus.** `scripts/consensus_corpus.py` runs dict-freq next to AnE-LID on every AnE-labelled row. The two share no evidence (a neural model vs. a word list). Where they agree, the label was right **99.3–100%** of the time on the public sets above, and they disagree on only 2–4% of tokens.

**What the numbers mean for the labels.** On the Latin-script pairs, single-word "islands" in the first language are the signature of tagging errors. After the switch to AnE-LID, Spanish, French, Italian and Vietnamese sit in the same 1–4% band as the script-exact pairs. Hindi, Tagalog and Malay stay higher (14–28%), which is most likely genuine speech style (a single Hindi or Tagalog function word between English words), not error. See `groupUpdate1.md` §5.2.

## How names are labelled

A name is marked by AnE-NER, but it still gets a language, and it only counts as a switch when there is visible evidence. This is the default policy, `script_words`:

- **Non-Latin rows.** A Latin-script name is English ("Severance Season 2" inside Arabic → English). A name in the row's own script is the first language (ترامب → Arabic).
- **Latin-script rows.** The name's own words decide when they are clearly one language's ("Game of Thrones" → English). Otherwise the name takes its neighbours' language ("Leo Messi" inside a Spanish clause → Spanish) and can't start a run of its own.

Two alternatives are stored and selectable in `kaggle/load_labels.ipynb`:
- `transparent`: names are skipped like punctuation (LinCE's convention).
- `script`: every Latin-script name is English.

## From word labels to prediction labels

- **Word level** (`data/tagged.parquet`): one language per word; CJK scripts per character.
- **Sub-word level** (`scripts/build_subword.py`): the original text is tokenized with the `xlm-roberta-base` tokenizer, and each piece takes the language of the word it sits in. Only 155 of 21.7 M pieces straddle two languages.
- **Targets, at either level:**
  - `y_sw`: whether the next token switches.
  - `y_len`: at a switch, the length of the new run.
  - `y_dur`: its class, Small 1–2, Medium 3–6 or Large 7+.

## Licences

- **AnE-LID and AnE-NER** are released for academic use only.
- **SwitchLingua is gated.** Derived labels stay in the private Hugging Face dataset `CodeSwitchGroup4/switchlingua-tagged-final` and are never committed to git.
