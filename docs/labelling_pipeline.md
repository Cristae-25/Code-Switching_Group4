# Appendix A — Code-switch token tagging: techniques and results

> This is the **label-provenance appendix** to `groupUpdate1.md`. The
> anticipatory task's `y_sw` / `y_dur` labels are a deterministic function of
> the per-token language labels produced here, so label quality *is* tagger
> quality. Read `groupUpdate1.md` first; come here for how the labels were
> made and how accurate they are.

**Project:** per-token language labelling of the SwitchLingua corpus, so that
code-switch *run lengths* (how many second-language tokens appear consecutively
before switching back) can be measured.

**Status:** all 234,172 rows tagged (22,349,157 tokens). Eight techniques
evaluated. dict-freq has replaced the fastText and mBERT-spaeng tiers in
production; romanized Hindi is the only pair still model-routed.

---

## 1. The corpus

`Shelton1013/SwitchLingua_text` — 234,172 rows, 15 CSVs, 14 first languages.
`data_generation_result` is a stringified Python list of utterances.

The corpus splits into two regimes that need completely different treatment:

| Regime | Languages | Rows | Share |
|---|---|---|---|
| **Non-Latin script** | Cantonese, Chinese, Japanese, Korean, Russian, Arabic | 160,922 | 68.7% |
| **Latin script** | Spanish, Italian, Malay, Philippines, German, French, Vietnamese | 64,864 | 27.7% |
| **Mixed** | Hindi — 59.7% romanized (Latin), 40.3% Devanagari | 8,386 | 3.6% |

The first regime is trivial: a token's *script* is its language. All difficulty
lives in the second, where both languages share an alphabet and only vocabulary
distinguishes them.

---

## 2. How we measure accuracy

Three methods, in decreasing order of trustworthiness.

### 2a. Gold evaluation set — the primary measure

`data/gold_eval.json` — **324 hand-labelled tokens** across 8 Latin-script
languages, drawn from real corpus rows.

* Each token is labelled `L1` (the row's first language), `EN` (English), or
  `NE`.
* `NE` covers named entities, punctuation and digits. **`NE` tokens are excluded
  from scoring** — no technique is being asked to get "Carney" or "," right, and
  including them would inflate every score.
* Accuracy = (correctly labelled non-`NE` tokens) / (total non-`NE` tokens).

| Language | Sentences | Scored tokens |
|---|---|---|
| Hindi | 5 | 59 |
| Philippines | 3 | 50 |
| Malay | 3 | 48 |
| Vietnamese | 2 | 48 |
| Spanish | 2 | 32 |
| French | 2 | 31 |
| German | 2 | 30 |
| Italian | 2 | 26 |
| **Total** | **21** | **324** |

**Known weakness: these labels were produced by the model assisting on this
project, not an independent annotator, and 26–50 tokens per language is small.**
A 100% score on 26 tokens is not the same claim as 100% on 3,000. Section 7
describes the human review built to replace this.

### 2b. Dataset-internal ratio check — independent but noisy

Every row carries `cs_ratio_result`, an aggregate the *generator* computed:
`{'computed_ratio': '60% Spanish : 40% English'}`. It is not token-level truth,
so it cannot score a tagger directly — but it exists on all 234,172 rows and is
independent of our labels, so it catches a systematically broken language.

We compare each tagger's second-language token share against that figure and
report mean absolute error.

**Calibrating it:** the six non-Latin languages are tagged *exactly*, so their
disagreement with `computed_ratio` measures noise **in the reference signal**,
not tagger error. That band — **|error| ≤ 0.134** — is the yardstick. A
Latin-script language inside it behaves like an exact tagger; far outside it is
genuinely broken.

One caveat: the metric is biased across scripts. CJK is tokenised per character
while English is per word, which mechanically understates the English share —
which is why every CJK language skews negative and Cyrillic/Arabic/Latin skew
positive. Compare *within* a script, not across.

### 2c. LinCE benchmark dev set — for the trained model only

4,823 train / 744 dev sentences of human-annotated Hindi-English, used to train
and validate XLM-R. Twitter text, so there is a domain shift from SwitchLingua's
clean generated prose.

---

## 3. Techniques

### 3.1 Unicode script detection — `src/unicode_tagger.py`

**How it works.** Classify each token by its Unicode script (`\p{Script=Han}`,
`\p{Script=Cyrillic}`, …). If the first language is written in Han and the
second in Latin, a token's script *is* its language. No model, no threshold.

**Result: exact by construction** for the six script-disjoint languages, plus
Devanagari Hindi rows. Covers **68.7% of the corpus**. Ratio-check errors range
−0.118 to +0.114, which we interpret as noise in the reference rather than
tagger error.

**Limits.** Useless when both languages share an alphabet — i.e. for the 31.3%
that actually matters.

---

### 3.2 Constrained fastText (`lid.176`) — `src/model_tagger.py`

**How it works.** Take fastText's 176-way language-ID distribution for a token
and read off only the two labels the row can be, picking the higher. The model
never has to *find* the language, only rank two known candidates. No training.

**Result: 83% overall** on the gold set. Strong on German (100%) and French
(90%); weak on Malay (75%).

**Why it fails where it does.** fastText scores each token **in isolation**.
`mas`, `bila`, `real` carry no signal alone — only their neighbours disambiguate
them. It also has far less Malay/Tagalog training data than European languages.

**Practical note.** The `fasttext` package ships source-only and fails to build
on Python 3.13 (`RuntimeError: libc++ is needed!`). `fasttext-predict` is a
prebuilt wheel exposing the same API.

---

### 3.3 GlotLID — tested, rejected

**How it works.** Same fastText architecture, trained on 2,000+ languages, so
better low-resource coverage in principle.

**Result: 53% vs lid.176's 47%** on a curated set of 19 hard Malay/Tagalog
tokens. A 1.6 GB model for ~6 points on a deliberately adversarial sample.

**Conclusion: rejected.** It shares lid.176's fundamental limitation — scoring
tokens in isolation — so more training data cannot fix it. Deleted.

---

### 3.4 mBERT LinCE token classifiers — `src/bert_tagger.py`

**How it works.** `BertForTokenClassification` fine-tuned on the LinCE
code-switch benchmark, from `sagorsarker/`. Unlike fastText it sees the **whole
utterance**, so context disambiguates shared words. Two models:

* `codeswitch-hineng-lid-lince` — Hindi-English, 96.58% published LinCE accuracy
* `codeswitch-spaeng-lid-lince` — Spanish-English, 98.32% published

Both are `bert-base-multilingual-cased` (vocab 119,547), pretrained on
Wikipedia. Their label set includes `ne`, which solves the proper-noun problem
for free.

**Result: 90% overall**, the best of the model-based approaches.

| Language | mBERT | vs fastText |
|---|---|---|
| Hindi | **100%** | 56% |
| Spanish | **100%** | 94% |
| Italian | **100%** | 96% |
| Philippines | **94%** | 86% |
| Malay | **88%** | 75% |
| Vietnamese | 88% | **90%** |
| German | 80% | **100%** |
| French | 68% | **90%** |

**The headline finding — cross-lingual transfer works, selectively.** The
*Spanish*-English model applied to Malay and Tagalog beats fastText decisively
(88% and 94%), acting as a generic "Latin-L1 vs English" discriminator. But it
**regresses** German (100→80) and French (90→68). Routing therefore has to be
per-language and empirical, not by language family.

`hineng` does **not** transfer (17% Malay, 64% Tagalog) — it is Hindi-only.

**Romanized Hindi is where it earns its keep.** fastText tags
`Woh kal Prime Minister Carney se milne wali hain` as almost entirely English;
mBERT gets `Woh kal/hin`, `Prime Minister/en`, `Carney/ne` essentially perfect.

---

### 3.5 XLM-RoBERTa, fine-tuned — `archive/train_xlmr_lid.py`

**Hypothesis.** XLM-R is pretrained on 2.5 TB of CommonCrawl across 100
languages, versus mBERT's Wikipedia. It should have far better Malay and Tagalog
representations, and so transfer better.

**How it works.** Fine-tuned `xlm-roberta-base` for token classification on
LinCE Hindi-English, 4 epochs, manual training loop, word-level alignment with
the first sub-token carrying each word's label.

**Training result: 97.13% LinCE dev accuracy — beating mBERT's published
96.58%.** The fine-tune genuinely worked.

**Transfer result: total failure (42% overall).**

| Language | XLM-R | mBERT |
|---|---|---|
| Hindi | **100%** | 100% |
| Philippines | 58% | **94%** |
| German | 37% | 80% |
| Spanish | 34% | **100%** |
| Vietnamese | 27% | 88% |
| French | 19% | 68% |
| Malay | 15% | **88%** |
| Italian | 4% | **100%** |

**Why.** Inspecting predictions shows it labels *every* non-Hindi word as
English:

```
Italian:  Non/lang1 vedo/lang1 l'ora/lang1 di/lang1 vedere/lang1 …
Hindi:    Woh/lang2 kal/lang2 Prime/lang1 Minister/lang1 Carney/ne   ✓
```

Fine-tuning taught it `lang2` = *romanized Hindi*, not `lang2` = *not English*.
Pretraining breadth gave it representations for Malay; nothing in the training
signal told it to use them. Epochs 2–4 did not improve transfer at all,
confirming this is not undertraining.

**Conclusion: the training pair dominates the base model.** XLM-R is
simultaneously the best model on Hindi and the worst on everything else. The
fair test — XLM-R vs mBERT both trained on *spaeng* — is blocked because only
the Hindi-English LinCE split has a script-free mirror; the official dataset
uses a loading script that `datasets` 4.x dropped.

---

### 3.6 Dictionary membership — `src/dict_tagger.py:tag_naive`

**How it works.** The intuitive rule: look the word up in the second language's
dictionary; if present, label it the second language. Uses `wordfreq`.

**Result: 52% overall — barely above chance.**

**Why it fails, measured.** The dictionaries overlap enormously:

| Language | Words present in **both** dictionaries |
|---|---|
| French | **97%** |
| German | 88% |
| Italian | 82% |
| Spanish | 81% |
| Philippines | 79% |
| Malay | 69% |
| Hindi | 53% |
| Vietnamese | 43% |

For French, 97% of tokens appear in both lists — loanwords (`management`,
`online`, `hospital`), shared vocabulary, proper nouns. "Is it in the L2
dictionary?" answers *yes* almost always. The rule tags every word of
`Como se dice Neuroplasticity en Espanol` as English.

---

### 3.7 Dictionary frequency comparison — `src/dict_tagger.py:tag_frequency`

**How it works.** One change from 3.6: instead of asking *"is this word in the
dictionary"*, ask *"which language is it more characteristic of"*. Compare Zipf
frequency in each language and take the higher. `no` is in both lists, but Zipf
7.16 in Spanish vs 6.35 in English. Words unknown to both are labelled `other`,
which correctly catches proper nouns.

**Result: 94% overall — the best technique tested.**

| Language | dict-freq | best model |
|---|---|---|
| Malay | **100%** | 88% (mBERT) |
| Philippines | **100%** | 94% (mBERT) |
| Spanish | **100%** | 100% (mBERT) |
| German | **100%** | 100% (fastText) |
| French | **100%** | 90% (fastText) |
| Italian | **100%** | 100% (mBERT) |
| Vietnamese | **100%** | 90% (fastText) |
| Hindi | 66% | **100%** (mBERT) |

**Independent confirmation.** On **767 rows** scored against the dataset's own
`cs_ratio_result` (method 2b — not our labels), mean absolute error is
**0.118 for dict-freq vs 0.147 for the current production pipeline**, winning on
5 of 7 languages and by the largest margins exactly where production is weakest
(Malay 0.103 vs 0.183; Philippines 0.147 vs 0.222).

**The one genuine failure: romanized Hindi (66%).** `wordfreq`'s Hindi list is
Devanagari. Romanized Hinglish has no standard orthography — `woh`/`wo`/`vo` are
all valid spellings — so **no dictionary can exist**. This is the only case that
genuinely requires a model.

---

### 3.8 CANINE, character-level — `archive/train_canine_lid.py`

**Hypothesis being tested.** Every other model here tokenises into subwords
learned from a fixed corpus. Romanized Hindi has *no standard orthography* —
the same word appears as `kya`/`kiya`, `nahi`/`nahin`/`nehi` — so a subword
vocabulary trained on Devanagari Hindi and English shatters it into fragments.
CANINE removes the vocabulary entirely: it consumes raw Unicode codepoints and
downsamples with convolutions, so a respelling is a small perturbation of the
input rather than a different token sequence. If spelling variation is what
hurts on romanized Hindi, CANINE should be the model that shrugs it off.

**Setup.** `google/canine-s`, fine-tuned as a character-level token classifier
on the LinCE `hineng` LID split. Word labels are broadcast to the characters of
each word; predictions are pooled back to words by majority vote over each
word's character span. Unlike the subword models there is no `word_ids()` to
align with, so the character spans are computed manually from the whitespace
offsets — a real source of off-by-one risk, and the reason the alignment is
unit-tested.

**Results.** Dev character accuracy 95.3% after one epoch, 96.2% after two.
Training was stopped during epoch 3 — it was competing with the full-corpus
re-tagging run for CPU and the gain per epoch had already dropped to under a
point. The epoch-2 checkpoint is kept at `models/canine-hineng-lid/`.

**The hypothesis did not survive.** A spelling stress test — programmatically
respelling romanized Hindi tokens along their common variation axes and
re-scoring both models on **n = 6,890 tokens** — found no CANINE advantage:

| Model | Original spelling | Respelled | Drop |
|---|---|---|---|
| mBERT `hineng` | 97.3% | 97.3% | ~0 |
| CANINE | 97.1% | 97.1% | ~0 |

mBERT does not degrade under respelling in the first place, so there is no
degradation for a character model to fix. The likely reason: mBERT's WordPiece
already fragments romanized Hindi into short, spelling-robust pieces, and the
*context* — which is what the token classifier actually uses — is unchanged by a
respelling.

**Verdict: negative result, retained as one.** CANINE costs more per token and
buys nothing here. mBERT `hineng` stays in production for romanized Hindi. This
is worth recording precisely because the hypothesis was reasonable: it is the
kind of idea that gets re-proposed unless the refutation is written down.

---

## 4. Summary — all techniques, gold set (324 tokens)

| Technique | Overall | Best at | Worst at | Cost |
|---|---|---|---|---|
| Unicode script | exact* | non-Latin scripts | unusable on Latin | none |
| **dict-freq** | **94%** | 7 of 8 languages | romanized Hindi (66%) | ~0.01 ms/token |
| mBERT LinCE | 90% | Hindi, Spanish, Italian | French (68%) | 679 MB, 16 ms/row |
| fastText lid.176 | 83% | German (100%) | Malay (75%) | 125 MB, 0.5 ms/row |
| GlotLID | — | — | rejected | 1.6 GB |
| XLM-R (hineng) | 42% | Hindi (100%) | Italian (4%) | 1.0 GB, 110 min train |
| dict-naive | 52% | Vietnamese (75%) | French/Italian (23%) | ~0.01 ms/token |
| CANINE (hineng) | — | romanized Hindi (98%) | Hindi-only, like mBERT | 530 MB, 2 epochs |

\* *Unicode is not scored on the gold set — that set is Latin-script only, where
Unicode does not apply. Its exactness follows from the method.*

---

## 5. Current production routing — `src/routing.py`

Updated after the dict-freq result: **fastText and mBERT-spaeng are both
retired.** dict-freq matched or beat them on every Latin-script pair and, unlike
either LinCE model, handles a non-English second language (the 8,511
German↔French rows), so nothing was left for them to do. That removes ~800 MB
of model weights from the pipeline.

| Tier | Languages | Rows |
|---|---|---|
| `unicode` | Cantonese, Chinese, Japanese, Korean, Russian, Arabic, + Devanagari Hindi rows | 160,922 |
| `dict` (wordfreq Zipf) | Spanish, Italian, Malay, Philippines, German, French, Vietnamese | 64,869 |
| `bert` (hineng) | romanized Hindi rows | 8,381 |

Hindi routes **per row** by script, not per language — Devanagari rows go to the
exact tagger, romanized rows to the model.

Re-tagging the corpus under this routing changed the labels on **57,050 of
234,172 rows (24.4%)**. Aggregate run-length statistics barely moved (mean 4.42,
median 2), which is the expected signature of a change that fixes individual
tokens rather than restructuring runs.

Trust tiers on the output: `exact` 160,922 · `provisional` 73,250.
`low` is now empty — Malay's 88% under mBERT was the only entry, and dict-freq
supersedes it. `ok` was renamed `provisional` to stop the tier from reading as a
finished result: every Latin pair scores 100%, but on 26–50 self-labelled tokens.

## 5b. Lexical override on AnE-LID (2026-11-03) — `src/lexical_override.py`

AnE-LID was chosen on the only human-annotated sets available: LinCE spa-eng and hin-eng, and Denglisch. It was **never measured on Malay, Vietnamese or Tagalog**, because no gold set exists for them.

**How the error was found.** Two independent signals, with no hand labelling:
- **The tagger consensus.** Malay was the worst pair, at 95.8% agreement on the words where both taggers name a language.
- **The trained anticipation model.** Where the taggers disagreed, 40–45% of positions were labelled as switches, against about 11% elsewhere. The model refused almost all of them (recall 0.16–0.18), because the "switches" were AnE calling Malay/Vietnamese function words English:
  - Malay: `akan` ×1,496, `dan`, `yang`, `dengan`, `saya`
  - Vietnamese: `và`, `sẽ`, `của`, `nhưng`
  - whole spans, such as "bekerjasama dengan agensi lain untuk"

**The rule.** dict-freq overrules AnE only on overwhelming evidence:
- **Frequency margin:** the word must be common in the other language (Zipf ≥ 4) and at least 2.0 Zipf (100×) more common there.
- **Vietnamese letters:** in a Vietnamese row, a word with Vietnamese-only letters (`đ ă ơ ư` and stacked tone marks) is Vietnamese.
- **Never touched:** names, acronyms (`AI` is not French `ai`), romanized Hindi (wordfreq's Hindi list is in Devanagari script) and German→French.

| Pair | Tokens relabelled | Share of words | Switches | L2 tokens |
|---|---:|---:|---:|---:|
| Malay | 14,809 | 2.85% | −4.8% | −8.0% |
| Vietnamese | 4,209 | 0.66% | −1.2% | −2.4% |
| Tagalog | 792 | 0.14% | +0.5% | −0.3% |
| French | 197 | 0.04% | −0.3% | −0.1% |
| Italian / Spanish / German | 36 / 15 / 17 | ≤ 0.01% | 0.0% | 0.0% |

**Validation.**
- **Human-annotated sets:** accuracy is unchanged (`eval_public.py`, column `AnE+override`). See `docs/results/2026-11-03_lexical_override.md`.
- **Consensus:** agreement rises to 99.1% for Malay and 99.8% for Vietnamese. That rise is partly circular: on the flipped tokens, dict-freq is no longer an independent check.
- **Spot-check:** random flips in context were correct, apart from the acronym case, which is now excluded.

## 6. Bugs found and fixed

Each was found by inspecting output, and each changed the results.

1. **Tokenizer split accented Latin.** The planned `[A-Za-z]+` pattern cut
   `difícil` into `dif|í|cil`, shattering most Spanish/French/German/Vietnamese
   words. Fixed with `\p{Script=Latin}`.
   *Impact:* a de-accented evaluation scored Vietnamese at 42%; with diacritics
   restored, 90%.

2. **Punctuation inflated every run length.** Boundary punctuation was absorbed
   into the *following* language, so an L1 sentence's final `.` counted as an L2
   token. **1,094,594 tokens — 18% of all counted L2 tokens.**
   *Impact:* mean run 5.37 → **4.40**; median 3 → **2**; single-token switches
   29.9% → **37.5%**.

3. **Model cache wiped by `%autoreload`.** A module-level dict meant a 679 MB
   model reloaded per row. Cache now lives on `sys`.
   *Impact:* an 82-minute cell that should take ~20 seconds.

4. **Recompute path disagreed with the live tagger.** `recompute_runs` read
   pre-absorption labels while `tag_row` counted post-absorption. Now tested for
   exact agreement.

5. **`transformers` 4.47.1 segfaults under `torch` 2.10** (exit 139, no
   traceback). Resolved with a dedicated `.venv` on transformers 5.17 / torch
   2.14.

**81 tests pass** (66 for the labelling pipeline, 15 for the anticipatory task).

---

## 7. Open questions

**1. ~~Should dict-freq replace the model tiers?~~ Resolved — yes.** Done, and
the corpus re-tagged. The caveat stands: the evidence is 26–50 self-labelled
tokens per language. dict-freq is more accurate on the gold set, closer on the
independent ratio check, and drops ~800 MB of models plus most of the runtime,
so the decision was cheap to make and cheap to reverse. `data/tagged_prev.parquet` (since deleted)
holds the previous labels for comparison.

**2. ~~Is Malay really below the bar?~~ Superseded.** Malay's `low` flag came
from mBERT's 88%. Under dict-freq it scores 100% on 48 tokens and is no longer
the outlier. Still only 48 tokens.

**3. Human review — built, not yet run. This is the blocking item.**
**Superseded (2026-09-27):** hand review was dropped in favour of automated evaluation on human-annotated public sets (`scripts/labelling/eval_public.py`) and corpus-wide tagger agreement (`scripts/labelling/consensus_corpus.py`), and the review tool was deleted. The original plan follows.

`scripts/labelling/review_server.py` served a stratified sample with per-token correction,
storing verdicts in SQLite and marking each row `✓ HAND-VERIFIED`:

```
./.venv/bin/python scripts/labelling/review_server.py --hard --control 5   # review 430 rows
./.venv/bin/python scripts/labelling/review_server.py --report --hard      # accuracy + Wilson CIs
./.venv/bin/python scripts/labelling/review_server.py --apply              # write back to parquet
```

430 rows (~27,800 tokens) — ~70× the current gold set. `--apply` adds
`hand_verified` / `verified_at` columns and recomputes the run lengths from the
corrected labels. **Zero rows reviewed so far**, so every accuracy figure in this
appendix still rests on 324 self-labelled tokens.

**4. Distilling spaeng into XLM-R** remains untested — the only route to the
XLM-R comparison the missing LinCE spaeng split blocks. Lower priority now that
spaeng is retired from production.

---

## 8. Reproducing these numbers

```bash
cd ~/Northeastern_GitHub/code-switch          # use ./.venv/bin/python throughout

./.venv/bin/python -m pytest tests/ -q        # 81 tests
./.venv/bin/python archive/eval_dict.py       # §3.6, §3.7 + overlap table
./.venv/bin/python archive/eval_xlmr.py       # §3.5
./.venv/bin/python archive/train_xlmr_lid.py  # retrain XLM-R (~110 min, MPS)
./.venv/bin/python scripts/labelling/run_tagging.py     # full corpus -> data/tagged.parquet
./.venv/bin/python scripts/labelling/export_sample.py   # 2,100 rows, per-token CSV
./.venv/bin/python scripts/labelling/eval_public.py       # human-annotated LinCE + Denglisch
./.venv/bin/python scripts/labelling/consensus_corpus.py  # corpus-wide agreement
```

The `archive/` scripts need model weights that were deleted to save disk space: retrain XLM-R and CANINE, and re-download fastText's `lid.176.bin`.

| Artifact | Contents |
|---|---|
| `data/tagged.parquet` | 234,172 rows · 22,349,157 labelled tokens · 67 MB |
| `data/gold_eval.json` | 324 hand-labelled tokens, 8 languages |
| `data/sample_tokens.csv` | 160,942 tokens, one row each, from `scripts/labelling/export_sample.py` (not in the repo: it contains corpus text) |
| `data/tagged_pre_ane.parquet` | labels from before AnE-LID and NE, for diffing |
| `models/xlmr-hineng-lid/` | fine-tuned XLM-R, 97.13% LinCE dev |

---

## 9. Result

Across **1,085,096 switches** in all 234,172 rows:

* **Median second-language run: 2 tokens** (mean 4.42)
* **36.9% of all switches are a single token**
* 53.2% are one or two tokens

| Percentile | Run length |
|---|---|
| 50th | 2 |
| 75th | 7 |
| 90th | 11 |
| 99th | 17 |
| 99.9th | 26 |
| max | 1,040 |

The distribution is heavily right-skewed: a large spike at 1–3 tokens
(insertional switching) and a very thin long tail. Only **63 of 1,085,096 runs
exceed 100 tokens (0.006%)**. The 1,040-token maximum is a genuine outlier — a
Cantonese row of 1,392 tokens, 1,082 of them English — not a tagging artifact;
it is Unicode-tagged and therefore exact.

**One caveat that matters for cross-language comparison:** these run lengths are
not measured in the same unit for every pair. `tokenize_mixed` splits Han, Kana
and Hangul at the *character* level, so a "token" in a Cantonese row is roughly a
morpheme while a token in a Spanish row is a word. That is why first-language
runs average 16.7 tokens for Cantonese and 3.8 for Hindi. See §5 of
`groupUpdate1.md` — it is the largest open threat to the universality claim.
