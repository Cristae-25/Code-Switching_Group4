# Group Update #1 — Anticipatory Code-Switch Prediction

> **Numbers regenerated 2026-09-27 on the final labels.** Latin-script rows
> with English as L2 are now tagged by AnE-LID rather than dict-freq, and
> names follow the `script_words` policy (`docs/labelling_pipeline.md`,
> `CLAUDE.md`). Every table below was rerun on those labels; the conclusions
> are unchanged, and where a number moved materially the text says so.

## Summary

**What we are building.** Existing code-switch systems *detect* a switch after
reading the sentence. We are testing whether a switch can be **anticipated**
from a prefix alone. At token `x_t`, with `x_{t+1}..x_n` hidden, predict whether
the next token crosses into the other language (`y_sw`) and how long that new
run will last (`y_dur`).

**Corpus.** `Shelton1013/SwitchLingua_text` — 234,172 conversations, 15 language
pairs, 22.3 M tokens, all per-token language-labelled by our own pipeline
(Appendix A). After dropping punctuation and the final token of each row, the
task has **19,365,955 scored positions** and **1,896,078 switches** — a
**9.79%** base rate.

**1 · Objectives & language selection (§1).** We use all 15 SwitchLingua pairs
rather than a convenient subset, because the confounds only separate across the
full range: 6 non-Latin scripts vs. 8 Latin, eleven language families, a 4×
spread in switch rate (5.9% Japanese → 25.7% Hindi), and one pair —
German→French — whose second language is not English. Universality is defined
so it can fail: we report macro-F1 across pairs, and we hold **four pairs out of
training entirely** and score them zero-shot.

**2 · Data engineering (§2).** A streaming `IterableDataset` reads the tagged
parquet incrementally — **175 MB peak RSS, 0.3 s per full pass** over 22.3 M
tokens. Splits are row-level (`row_id % 10`), never token-level, so no
conversation straddles a split. `y_sw` and `y_dur` are *derived*, not annotated:
a deterministic function of the per-token language sequence, with punctuation
transparent, the final token unlabelled, and end-of-text runs flagged as
right-censored. We also caught an **unintended language curriculum** — the
corpus is contiguous per language and the shuffle buffer was far smaller than
one language block — and fixed it with a one-off row shuffle.

**3 · Causal constraints (§3).** Three leak paths are closed: an additive
`-inf` attention mask (not just `is_causal`, which is only a kernel hint);
`LayerNorm` over features rather than any time-axis pooling; and a strict
separation between the language *input* at `t` and the language *label* at
`t+1`. Rather than assert this, `check_causality` rewrites every token after `t`
with random values and measures whether any logit at or before `t` moves — it
reports **0.00e+00 drift**, and a companion test proves the check fails when the
encoder is made bidirectional.

**4 · Baseline metrics (§4).** Switches are 9.79% of positions, so accuracy is
a trap: predicting "never switch" scores 90.2% accuracy at **0.000 F1**. All
numbers are Anticipatory F1 on the positive class, with thresholds tuned on
validation.

| Model | Anticipatory F1 | macro-F1 |
|---|---|---|
| chance (Bernoulli prior) | 0.098 | 0.106 |
| **naive — "switch iff `x_t` is L2"** | **0.267** | 0.231 |
| best training-free (run-length hazard) | 0.295 | 0.259 |
| **causal transformer (15 M params, ~1 epoch)** | **0.503** | **0.503** |

The naive one-bit model reaches 0.267 because second-language runs are short
(median 2 tokens). Its ceiling is structural, and the hazard table shows why:
**first-language runs are effectively memoryless** (hazard flat at 0.055–0.078
regardless of length) while second-language runs decay sharply (0.333 → 0.081).
So the best training-free model scores **0.000 on switching *into* the second
language** — no threshold extracts a decision from a flat line. Predicting the
switch *out* of an insertion is counting; predicting the switch *into* one needs
to know what is about to be said. That is the gap the transformer closes, and
it is this update's main finding. Duration: 53.3% bucket accuracy vs. a 42.8%
mode baseline.

**Universality (§4.5).** Trained on 10 language pairs and scored on 4 it had
never seen, the model reaches **F1 0.383 — above the 0.295 of the best
training-free model, which was fitted with access to every pair.** The prefix
signal transfers. But it transfers across *first* languages only: on
German→French, the corpus's one non-English second language, the model scores
0.243 against 0.212 for "always predict a switch" — little real signal. With 14 of
15 pairs sharing English as L2, "the other language" and "English" are nearly
the same variable in training, so full universality is not yet demonstrable
from this corpus.

**5 · Next steps (§5).** Two concrete blockers. **(a) The token is not the same
unit across languages** — CJK is tokenized per character, Latin per word, so
run lengths are not comparable and vocabulary pressure is wildly uneven (OOV at
50 K types: 0.28% Cantonese, 12.54% Russian). Moving to shared subwords reopens
alignment, where **last-subword-wins** — not the first-subword-wins used for
tagging — is the causally correct pooling. **(b) Label noise on the Latin
pairs, now largely fixed.** One mis-tagged token manufactures two spurious
switches; measured via single-token language islands, the script-exact pairs
sit at 1.0–4.0%. Under the old dict-freq labels the Latin pairs reached 28.5%;
with AnE-LID (99.4% on human-annotated LinCE Spanish–English, vs 94.1% for
dict-freq) Spanish, French, Italian and Vietnamese fall inside the exact band.
Hindi, Tagalog and Malay stay high (14–28%), which AnE's 98.8% on LinCE
Hindi–English suggests is mostly genuine insertion style, not error.

---

| Checklist item | Where | Headline |
|---|---|---|
| 1. Objectives & language selection | §1 | All 15 pairs; universality tested by **holding 4 pairs out of training entirely** — zero-shot **F1 0.383**, above every training-free baseline (§1.4, §4.5) |
| 2. Data engineering | §2 | Streaming `IterableDataset`, **175 MB peak RSS** on a 22.3 M-token corpus; `y_sw`/`y_dur` derived, not annotated; caught and fixed an **unintended language curriculum** in the file order |
| 3. Causal constraints | §3 | Additive `-inf` mask + `is_causal`; **verified empirically at 0.00e+00 logit drift**, and the test is shown to fail on a bidirectional model |
| 4. Baseline metrics | §4 | Naive "current token is L2" → **F1 0.267**. Best training-free → **0.295**. Causal transformer → **0.503**; zero-shot **0.383** (§4.5) |
| 5. Next steps | §5 | Two real blockers: **CJK tokens are characters, not words**, and **only 1 of 15 pairs has a non-English L2**. Latin-pair label noise, the third, is largely fixed by AnE-LID (§5.2) |

Everything below is reproducible from this repo; commands are in §6.
**210 tests pass** (`./.venv/bin/python -m pytest tests/ -q`).

---

## 1. Objectives & Language Selection

### 1.1 What "universality" has to mean to be testable

A claim that switch anticipation is universal is only interesting if it can
fail. Three progressively stronger readings:

1. *The model works on many pairs.* Weak — training on all 15 pairs and
   reporting a mean is a capacity result, not a universality result.
2. *The model works equally well across pairs.* Better — this is the **macro-F1
   across pairs**, unweighted. It matters here because Cantonese→English is
   50.9% of the corpus; a micro average is a Cantonese score wearing a corpus's
   name.
3. *The model works on pairs it has never seen.* This is the real test, and the
   one we adopt. **Four pairs are held out of training entirely** and scored
   zero-shot.

We therefore report all three: micro F1, macro F1 across pairs, and zero-shot F1
on held-out pairs.

### 1.2 The pairs

We use **all 15 pairs** in SwitchLingua rather than a convenience subset,
because the confounds we care about are only visible across the full range.

| L1 | L2 | Rows | Words | Switch rate | L2 share | L2 run mean / med | L1 run mean / med |
|---|---|---:|---:|---:|---:|---:|---:|
| Cantonese | English | 119,248 | 12,317,971 | 8.6% | 0.184 | 3.99 / 2 | 16.66 / 13 |
| Arabic | English | 8,548 | 385,384 | 12.9% | 0.439 | 6.04 / 5 | 7.44 / 7 |
| Philippines | English | 8,537 | 484,035 | **22.2%** | 0.487 | 4.17 / 2 | 4.33 / 3 |
| German | **French** | 8,511 | 405,653 | 12.1% | 0.406 | 6.00 / 5 | 8.32 / 8 |
| Chinese | English | 8,458 | 819,886 | 7.3% | 0.268 | 6.63 / 5 | 17.57 / 17 |
| Vietnamese | English | 8,457 | 574,495 | 10.8% | 0.301 | 5.07 / 3 | 11.38 / 11 |
| Malay | English | 8,386 | 446,080 | 16.8% | 0.434 | 4.78 / 3 | 6.09 / 5 |
| Japanese | English | 8,385 | 955,997 | **5.9%** | 0.225 | 6.78 / 6 | 22.77 / 22 |
| Hindi | English | 8,381 | 470,438 | **25.7%** | 0.460 | 3.42 / 2 | 3.98 / 3 |
| Korean | English | 8,266 | 732,203 | 7.2% | 0.237 | 5.79 / 4 | 18.30 / 18 |
| Spanish | English | 8,110 | 445,461 | 10.7% | 0.341 | 5.67 / 4 | 10.40 / 10 |
| French | English | 8,070 | 423,592 | 10.9% | 0.355 | 5.72 / 5 | 10.02 / 10 |
| Russian | English | 8,017 | 376,909 | 12.6% | 0.397 | 5.64 / 4 | 8.17 / 8 |
| Italian | English | 7,979 | 428,908 | 10.8% | 0.371 | 6.06 / 5 | 9.93 / 9 |
| German | English | 6,819 | 333,115 | 11.9% | 0.364 | 5.40 / 4 | 9.08 / 9 |

**The switch rate varies 4× across pairs — 5.9% (Japanese) to 25.7% (Hindi).**
That single column is the reason a global decision threshold cannot be
universal, and it is why §4 tunes a threshold on validation data and then
reports what a *fixed* threshold costs on each pair.

### 1.3 The four axes we selected the pairs to span

| Axis | Contrast in the data | Why it threatens universality |
|---|---|---|
| **Script disjointness** | 6 non-Latin L1s vs. 8 Latin-script L1s | Where the scripts differ, the language of a token is visible from its *characters*. A model could learn "predict a switch when the next character set changes" — except it cannot see the next token, so what it actually learns is whether script gives the *prefix* a cheaper cue. It also means labels for those pairs are exact and labels for the Latin pairs are not (§5.2). |
| **Typological family** | Sinitic, Japonic, Koreanic, Semitic, Indo-Aryan, Slavic, Romance ×3, Germanic ×2, Austronesian ×2, Austroasiatic | Morphological richness changes how much a *word*-level model can generalise; Russian and Arabic have 11–13% OOV at a 50 K vocabulary where Cantonese has 0.3% (§5.1). |
| **Switch density** | 5.9% → 25.7% | Directly sets the positive-class prior, and therefore F1. |
| **Second language** | 14 pairs are `L2 = English`; **German→French is not** | Without this pair, "anticipate a switch" and "anticipate English" are indistinguishable hypotheses. It is 8,511 rows — small, but it is the only thing in the corpus that separates them. |

### 1.4 The held-out design

The zero-shot set is chosen to hit one axis each, not to be easy:

| Held out | Rows | The axis it probes |
|---|---|---|
| **Korean**→English | 8,266 | Non-Latin script + agglutinative; no Koreanic language remains in training |
| **Italian**→English | 7,979 | *Near* transfer — Spanish and French stay in training, so this is the optimistic end |
| **Hindi**→English | 8,381 | Highest switch rate (25.7%), and 59.7% of rows are romanized, so script is not a cue |
| **German**→English and →**French** | 15,330 | Germanic; and the only non-English L2 in the corpus |

Training keeps the other 10 first-languages: Cantonese, Arabic, Philippines,
Chinese, Vietnamese, Malay, Japanese, Russian, Spanish, French.

Because the held-out pairs are never seen, the zero-shot model **drops the
language-pair embedding entirely** (`--no-pair` is forced whenever a holdout is
specified) — otherwise it would index an embedding row that was never trained,
and the result would measure random initialisation rather than transfer.

Recording the predictions this design makes, so §4.5 can be checked against
them rather than narrated after the fact: Italian should be the easiest of the
four (two Romance languages remain in training), Korean the hardest (nothing
related remains), and German→French the most diagnostic. **One of those three
turned out to be wrong, narrowly** — Hindi edges out Italian — see §4.5.

---

## 2. Data Engineering

### 2.1 The streaming DataLoader — `src/streaming.py`

Materialising the corpus as Python lists of strings costs roughly 3 GB, which is
what made an earlier notebook-based attempt unusable. Nothing in this layer ever
holds more than one Arrow batch.

```python
pf = pq.ParquetFile(path)
for bi, batch in enumerate(pf.iter_batches(batch_size=512, columns=_COLUMNS)):
    if bi % nworkers != wid:          # worker sharding
        continue
    for row in batch.to_pylist():
        if not in_split(row["row_id"], split):
            continue
        yield row
```

Measured on the real file: **peak 175 MB RSS, 0.3 s for a full pass** over the
six columns the task needs. The parquet is a single 67 MB row group and
`iter_batches` still decodes column pages incrementally — we verified this rather
than assuming it, because a row-group-at-a-time reader would have silently
loaded everything.

Four properties this layer has to guarantee:

**(a) Splits are row-level, never token-level.** Two windows from the same
conversation share topic, speaker and vocabulary; splitting *tokens* would leak
across the split boundary and inflate every number in §4. `row_id % 10` gives
0–7 train / 8 val / 9 test. The row ids are sequential and the source CSVs are
contiguous per language, so a modulo is an even interleave: every language lands
in every split in proportion, with no shuffle, no stored manifest, and no state
to keep in sync. Result: **15,482,668 / 1,942,548 / 1,940,739** scored positions.

**(b) Long conversations are windowed without amputating context.** A row longer
than the window becomes overlapping windows of `window=256`, `stride=192`. The
first `window - stride = 64` positions of every *continuation* window are fed to
the model but **masked out of the loss** — they exist to rebuild the prefix, and
scoring them would score a prediction made from a truncated history. A test
asserts that across all windows of a row, every position is scored exactly once
and none is scored twice.

**(c) Workers do not duplicate data.** Each DataLoader worker opens its own
parquet handle and keeps only batches where `batch_index % num_workers ==
worker_id`.

**(d) Shuffling is bounded.** A reservoir buffer (default 4,096 windows)
shuffles locally; the interleaved split means consecutive windows are already
from different conversations. Val and test are not shuffled.

**(e) The file order is a curriculum, and had to be broken.** This one was
found by watching the training loss, not by reading the code.
`iter_batches` can only read forward, and the source CSVs are contiguous per
language — Cantonese occupies the first ~119 K rows. A 4,096-window shuffle
buffer is three orders of magnitude smaller than that block, so streaming the
file in order trains the model as an *unintended curriculum by language*: tens
of thousands of consecutive steps on one pair, then a hard switch. The loss says
so plainly:

```
step 1400-3400   loss 0.61 -> 0.59      flat: still inside the Cantonese block
step 3600        loss 0.75              <- block boundary
step 3800-4200   loss 0.79 -> 0.84      higher-switch-rate languages
step 4600        loss 0.51              <- another boundary
```

That is not convergence; it is the model being handed one language at a time,
and it biases the final weights toward whichever pair came last — fatal for a
macro-F1 that is supposed to measure universality (§1.1).

`scripts/shuffle_parquet.py` fixes it once, offline: permute the rows, write
`data/tagged_shuffled.parquet` in 29 row groups. `row_id` travels with the row,
so the `row_id % 10` split assignment is **bit-identical** — train/val/test
membership does not move and the vocabulary stays valid. All 14 first languages
now appear within the first 2,000 rows instead of one. §4.5 reports both runs,
because the unshuffled one is the evidence that this mattered.

Padding is handled so it can never be mistaken for signal: padded positions get
word id `0` (`<pad>`, a zeroed embedding row) and **language id `2`**, a third
slot reserved for padding — not `0`, which is a real language and would make
every pad look like first-language text.

### 2.2 Generating `y_sw` and `y_dur` — `src/anticipation.py`

Both labels are **derived, never annotated**: they are a deterministic function
of the per-token language sequence. Given the language slot sequence
`L_1..L_n ∈ {L1, L2}`:

```
y_sw[t]  = 1  if  L_{t+1} ≠ L_t                    defined for t < n
y_len[t] = length of the run that L_{t+1} begins    defined only where y_sw[t] = 1
y_dur[t] = class of y_len[t]: Small 1-2 | Medium 3-6 | Large 7+
```

```python
y_sw[:-1] = (langs[1:] != langs[:-1]).astype(np.int8)

starts, lengths = run_starts_and_lengths(langs)
for s, ln in zip(starts[1:], lengths[1:]):
    y_len[s - 1] = ln                  # the run starting at s is anticipated at s-1
    y_dur[s - 1] = dur_class(ln)
    censored[s - 1] = (s + ln == n)    # ... and is cut off by the end of the text
    dur_known[s - 1] = not censored[s - 1] or ln >= 7   # class already certain
```

Worked example — a real Spanish→English row (`row_id 217646`), exactly as the
loader produces it. `·` means "no label"; `*` means right-censored:

```
  t  token           lang   y_sw   y_dur
  0  El              L1        0      ·
  1  negocio         L1        0      ·
  2  está            L1        0      ·
  3  creciendo       L1        0      ·
  4  rápidamente     L1        1   large*    <- switch into a 10-token L2 run that
  5  Success         L2        0      ·         is still open when the text ends
  6  feels           L2        0      ·
  7  like            L2        0      ·
  8  a               L2        0      ·
  9  joke            L2        0      ·
 10  when            L2        0      ·
 ...
 14  it              L2        –      ·      <- final token: no t+1, no label
```

Under the earlier dict-freq labels this row read `like/L2 a/L1 joke/L2`: the
English article "a" was tagged as the Spanish preposition, which split one run
into three and invented two switches (§5.2). AnE-LID tags it correctly.

**Duration classes.** Run lengths are heavy-tailed (max 1,040), so regressing on
raw length is dominated by a handful of rows. The project requirement defines
three classes: **Small 1–2, Medium 3–6, Large 7+ tokens**. The integer length is
stored as `y_len` and the class derived from it (`dur_class`), so the cut-offs
can change without rebuilding labels. 34.7% of all second-language runs are a
single token. (The results in §4.4–§4.5 predate this and use the earlier seven
buckets `1, 2, 3, 4, 5, 6-10, 11+`.)

**Three decisions that materially change the numbers, made explicit:**

1. **Punctuation is dropped, not labelled.** It is *transparent*: it neither
   breaks a run nor counts toward one. This matches the descriptive pipeline
   (`l2_word_runs`), so a run length in §4 means the same thing as a run length
   in Appendix A §9. Treating punctuation as predictable tokens would pad the
   positive class with tokens nobody wants predicted. Cost: 2,749,030 of
   22,349,157 tokens (12.3%), leaving 19,600,127 words.
2. **The final token of every row carries no label.** There is no `t+1`. It is
   masked out rather than assigned a convenient `0` — assigning `0` would hand
   the model 234,172 free true negatives and quietly lift precision.
3. **Durations that hit the end of the text are right-censored and flagged.** A
   run still in progress when the text ends is *at least* that long. It keeps
   the observed length with `censored=True`. It is a duration target only when
   that lower bound already fixes the class — a cut-off run of 7+ is certainly
   Large — so `dur_known = switch & (~censored | y_len >= 7)`. The duration head
   never learns that runs end early.

### 2.3 What the model is given at each position

| Input at `t` | Legal? | Why |
|---|---|---|
| word identity of `x_t` | ✅ | in the prefix |
| language of `x_t` | ✅ | in the prefix — it is the *previous* token's label, not the current target |
| position index | ✅ | |
| language-pair id | ✅ in-distribution, ❌ zero-shot | a held-out pair has no trained embedding; dropped there |
| language of `x_{t+1}` | ❌ | this **is** `y_sw` |

The language input and the switch label are built from the same array, one
shifted against the other. That shift is the only thing separating a legitimate
model from a 100% score, which is why §3 verifies it empirically instead of by
reading the code.

---

## 3. Causal Constraints

The task is defined by what the model cannot see. Three paths by which the
future can reach position `t`, and how each is closed — `src/causal_model.py`.

**1. Attention.** `nn.TransformerEncoderLayer` is bidirectional by default. We
build an additive float mask with `-inf` above the diagonal and pass it together
with `is_causal=True`:

```python
mask = torch.triu(
    torch.full((t, t), float("-inf"), device=word_ids.device), diagonal=1
)
h = self.encoder(x, mask=mask, is_causal=True)
```

`is_causal` alone is only a *hint* to PyTorch's fused kernel — it is not a
guarantee, and passing it without a mask is a common way to ship a bidirectional
model believing it is causal. The explicit mask is what enforces the constraint;
the flag lets the fast path be taken when it is available.

**2. Normalisation and pooling.** Any operation that mixes across the time axis
leaks. `LayerNorm` normalises over the *feature* axis per position and is safe.
`BatchNorm1d` over time would not be, and is not used. There is no mean-pooling
and no `[CLS]`-style aggregate — the two heads read position `t`'s hidden state
directly.

**3. Labels entering as inputs.** Covered in §2.3: the language embedding at `t`
carries `lang(x_t)`, never `lang(x_{t+1})`.

**And one that is specific to windowing:** a continuation window starts
mid-conversation, so its first positions have an artificially short prefix. They
are fed as context and excluded from the loss (§2.1b). This is not a leak — it
is the opposite, an *absence* of context — but it corrupts the metric in the
same way if ignored.

### 3.1 Verifying it instead of asserting it

`check_causality` runs the assembled model — heads included — and, for every
prefix length `t`, rewrites **every token and language after `t`** with random
values, then checks whether any logit at or before `t` moved:

```
causality check: max logit drift 0.00e+00 over 31 prefixes -> PREFIX-ONLY OK
```

Drift is exactly zero, not merely small. The check runs at the start of every
training run and aborts before the first gradient step if it fails.

A check that always passes is worthless, so there is a second test asserting the
check **fails** when the encoder is monkey-patched to run bidirectionally:

```python
def test_bidirectional_model_would_fail_the_same_check():
    m.encoder.forward = bidirectional          # drop the mask
    assert not check_causality(m, 500, seq_len=24)["causal"]
```

---

## 4. Baseline Metrics

### 4.1 Why F1 and not accuracy

Switches are 9.79% of positions. A model that predicts "no switch" forever
scores **90.2% accuracy and 0.000 F1**. Every number below is precision, recall
and F1 on the **positive (switch) class** — the *Anticipatory F1*. Thresholds
are tuned on the **validation** split and applied unchanged to test.

### 4.2 Training-free baselines — full test split, 1,940,739 positions

| Model | P | R | **F1** | macro-F1 | Accuracy |
|---|---|---|---|---|---|
| `never` — always "no switch" | 0.000 | 0.000 | **0.000** | 0.000 | **0.902** |
| `always` — always "switch" | 0.098 | 1.000 | **0.178** | 0.219 | 0.098 |
| `prior` — Bernoulli(base rate) | 0.098 | 0.098 | **0.098** | 0.106 | 0.823 |
| **`naive_l2` — switch iff `x_t` is second-language** | 0.189 | 0.457 | **0.267** | 0.231 | 0.755 |
| `hazard` @ threshold 0.5 | 0.000 | 0.000 | **0.000** | 0.000 | 0.902 |
| `hazard` @ τ = 0.129, tuned on val | 0.254 | 0.352 | **0.295** | 0.259 | 0.836 |

### 4.3 How the naive model performs, and what it reveals

**The checklist's naive model — one bit of context, "is the current token in the
second language?" — reaches Anticipatory F1 0.267**, against a 0.098 chance
baseline. It is 2.7× chance from a single bit.

Why it works at all: second-language runs are short (median 2 tokens, 34.7% are
a single token) while first-language runs are long (median 13 for Cantonese).
So "I am currently in L2" really does mean "a switch back is imminent". The rule
gets **recall 0.457 at precision 0.189** — it fires on every L2 token, and about
one in five is right.

Its failure is structural, and the hazard table shows exactly where:

```
P(switch at t+1 | language of x_t, how long the current run has already run)

run length so far    after L1    after L2
        1              0.058       0.333
        2              0.058       0.231
        3              0.057       0.110
        4              0.055       0.081
        5              0.062       0.087
        6              0.064       0.105
        7              0.069       0.129
        8              0.073       0.154
        9              0.078       0.172
       10              0.077       0.211
```

Two findings:

- **First-language runs are close to memoryless.** The hazard after L1 sits at
  0.055–0.078 regardless of how long the run has gone on — an almost flat line,
  i.e. approximately geometric. *How long you have been speaking Spanish tells
  you essentially nothing about when you will switch to English.*
- **Second-language runs are strongly duration-dependent.** The hazard starts at
  0.333 and falls by 4× to 0.081 by the fourth token, then slowly climbs. Short
  insertions snap back immediately; anything that survives four tokens has
  become a genuine clause and behaves differently.

> **Correction (2026-09-28, `notebooks/switch_eda.ipynb` §2a).** The table above is pooled over all pairs, and half of its tokens are Cantonese. Per pair, the first-language hazard is flat only for Chinese, Japanese, Korean and Cantonese. In the Latin-, Cyrillic- and Arabic-script pairs it **rises steadily** with run length, from about 2% to about 20% by the 15th word, so how long someone has been speaking Spanish *does* predict a switch. The second-language hazard is U-shaped in every pair. A per-pair hazard baseline should therefore do better than the pooled one on switching into L2 outside CJK.

This asymmetry is why `hazard@tuned` beats `naive_l2` — and why it beats it *in
only one direction*:

| Direction | n | Base rate | P | R | F1 |
|---|---|---|---|---|---|
| L1→L2 | 1,481,918 | 0.069 | 0.000 | 0.000 | **0.000** |
| L2→L1 | 458,821 | 0.189 | 0.254 | 0.770 | **0.382** |

**The best training-free model scores exactly zero on switching *into* the
second language.** Its whole signal is run-length, the after-L1 hazard is flat
at ~0.06, and no threshold can extract a decision from a flat line. Predicting
the switch *out* of an insertion is nearly a counting problem; predicting the
switch *into* one requires knowing what is about to be said. That is the gap a
language model has to fill, and it is the headline result of this update.

**The 0.5-threshold trap.** `hazard@0.5` scores 0.000 F1 at 90.2% accuracy — not
because the model is bad, but because it is *well calibrated*: the true switch
probability never exceeds 0.333, so a 0.5 threshold is never crossed. Any
probabilistic model of this task, argmax'd, predicts "no switch" forever. We
report both thresholds because this is the most likely way to report a broken
result as a good one.

### 4.4 Duration baseline

*These 7-bucket numbers are historical. Duration now uses the 3 required
classes (Small 1–2, Medium 3–6, Large 7+ tokens), stored as integer run
lengths; results in those classes come with the next training round (§5.1).*

Evaluated at **oracle switch positions** (n = 166,512) — the duration head is
only asked about positions where a switch genuinely occurs, so duration error is
not contaminated by switch-detection error.

| Target | n | Mode | Acc | 1 | 2 | 3 | 4 | 5 | 6–10 | 11+ |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| into L2 | 85,727 | `1` | 0.388 | **.388** | .174 | .062 | .041 | .040 | .201 | .093 |
| into L1 | 80,785 | `11+` | 0.471 | .075 | .064 | .058 | .051 | .051 | .230 | **.471** |
| overall | 166,512 | — | **0.428** | | | | | | | |

Predicting the per-direction mode gives **42.8% bucket accuracy, 63.0% within
one bucket, MAE 1.83 buckets**. The two distributions are nearly mirror images
— switching *into* L2 most often produces a one-token insertion; switching back
*into* L1 most often produces a long run — which is the same asymmetry the
hazard table shows, seen from the other side.

### 4.5 Causal transformer

**Setup.** A from-scratch causal transformer — **14,980,872 parameters**,
4 layers, `d_model` 256, 4 heads, window 256 / stride 192, batch 32,
AdamW + OneCycle at `lr` 3e-4, 6,000 steps on MPS. Inputs are the word id and
the current token's language slot; a language-pair embedding is added
in-distribution and dropped for the zero-shot run (§2.3).

6,000 steps × 32 = 192,000 windows against **187,581 windows per epoch**, so the
model sees the training split **1.02 times**. Worth noting alongside that
number: 187,338 training rows produce 187,581 windows, i.e. all but 243 rows fit
inside a single 256-token window. The overlapping-window and warm-up-masking
machinery in §2.1(b) is therefore correct but barely exercised on this corpus —
it matters for the long Cantonese conversations and almost nowhere else.

This model is deliberately small and undertrained. It exists to answer *"does a
prefix carry signal the count-based baselines cannot reach?"*, not to be a
result worth defending. Its causality check reports **0.00e+00 drift** before
the first gradient step.

**Headline — all 15 pairs in training, full test split.** Threshold τ = 0.261
tuned on validation (val F1 0.505), applied unchanged to test.

| Model | P | R | **F1** | macro-F1 | Accuracy |
|---|---|---|---|---|---|
| `naive_l2` (checklist naive) | 0.189 | 0.457 | 0.267 | 0.231 | 0.755 |
| `hazard@tuned` (best training-free) | 0.254 | 0.352 | 0.295 | 0.259 | 0.836 |
| **causal transformer** | 0.473 | 0.537 | **0.503** | **0.503** | 0.896 |

**+0.208 F1 over the best training-free model**, and 5.1× the 0.098 chance
baseline. The gain is concentrated exactly where §4.3 predicted it would have to
be: the hazard model scored **0.000** on L1→L2 because run length alone is a
flat signal there, and lexical context is the only thing that can fill it.

**Macro-F1 (0.503) equals micro-F1 (0.503).** That is the universality number
from §1.1 reading #2, and the equality matters: Cantonese is 50.9% of the
corpus, so a model that was really only good at Cantonese would show micro
well above macro. It does not.

| Pair | n | Base | P | R | F1 |
|---|---:|---:|---:|---:|---:|
| Korean | 72,637 | 0.072 | 0.565 | 0.715 | **0.631** |
| Hindi | 45,754 | 0.259 | 0.490 | 0.808 | 0.610 |
| Philippines | 46,130 | 0.224 | 0.447 | 0.714 | 0.550 |
| Japanese | 94,997 | 0.060 | 0.496 | 0.566 | 0.529 |
| French | 41,448 | 0.109 | 0.468 | 0.557 | 0.509 |
| Cantonese | 1,224,548 | 0.086 | 0.485 | 0.497 | 0.491 |
| Spanish | 45,175 | 0.108 | 0.450 | 0.528 | 0.486 |
| Arabic | 37,052 | 0.132 | 0.426 | 0.539 | 0.476 |
| German | 72,202 | 0.119 | 0.441 | 0.507 | 0.471 |
| Italian | 41,459 | 0.111 | 0.454 | 0.486 | 0.469 |
| Russian | 37,302 | 0.125 | 0.419 | 0.521 | 0.464 |
| Malay | 44,558 | 0.169 | 0.396 | 0.537 | 0.456 |
| Vietnamese | 55,973 | 0.110 | 0.463 | 0.433 | 0.448 |
| Chinese | 81,504 | 0.073 | 0.479 | 0.418 | **0.446** |

The spread is **0.446–0.631**, a 1.41× ratio between the worst and best pair.
Note that the pairs are *not* ordered by script, family or switch rate — Korean
(non-Latin, 7.2% switch rate) and Hindi (romanized, 25.9% on test) sit at the top
together, and Chinese (non-Latin) and Vietnamese (Latin) sit at the bottom.
Whatever is driving the residual differences, it is not the axes §1.3 was built
to separate.

**Duration head**, scored at oracle switch positions so duration error is not
contaminated by switch-detection error (n = 166,512):

| Model | Bucket accuracy | Within one bucket | MAE (buckets) |
|---|---|---|---|
| mode per direction (§4.4) | 0.428 | 0.630 | 1.827 |
| **causal transformer** | **0.533** | **0.748** | **1.187** |

**Ablation — the language curriculum was worth 0.044 F1.** Identical
configuration and seed; the only difference is whether the rows were shuffled
before streaming (§2.1e):

| Training order | P | R | **F1** | macro-F1 | Per-pair spread | Duration acc |
|---|---|---|---|---|---|---|
| file order (curriculum) | 0.432 | 0.490 | 0.459 | 0.460 | 0.387 – 0.594 | 0.507 |
| **row-shuffled** | 0.473 | 0.537 | **0.503** | **0.503** | 0.446 – 0.631 | **0.533** |

Shuffling gains **+0.044 F1** and, more tellingly, lifts the *floor* by 0.059
while lifting the ceiling by 0.037 — it compresses the per-pair spread rather
than just raising the average. That is the expected signature of removing a
curriculum: the languages that were starved by being early in the file recover
the most.

**Zero-shot — the universality test from §1.4.** A second model was trained on
**10 first languages only** (Cantonese, Arabic, Philippines, Chinese,
Vietnamese, Malay, Japanese, Russian, Spanish, French) with **German, Hindi,
Korean and Italian held out entirely**. The pair embedding is dropped
automatically, so nothing indexes a row that was never trained. τ = 0.263 tuned
on the validation split of the *trained* pairs only.

| | n | Base rate | P | R | **F1** | macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| the 10 trained pairs | 1,708,687 | 0.093 | 0.472 | 0.519 | 0.494 | 0.481 |
| **the 4 held-out pairs** | 232,052 | 0.131 | 0.381 | 0.385 | **0.383** | **0.383** |

**The headline: 0.383 zero-shot beats 0.295 — the best training-free model,
which was fitted with full access to every pair including these four.** A
prefix-based switch signal learned from ten language pairs transfers to pairs
the model has never seen, and transfers well enough to outscore a count-based
model that was allowed to look. That is the strongest of the three readings of
universality in §1.1, and it survives.

It survives with a real cost, though: **0.494 → 0.383 is a 22% relative drop.**
Universality is demonstrated, not achieved.

| Held-out pair | n | Base | P | R | F1 |
|---|---:|---:|---:|---:|---:|
| Hindi→English | 45,754 | 0.259 | 0.546 | 0.379 | **0.447** |
| Italian→English | 41,459 | 0.111 | 0.396 | 0.470 | 0.430 |
| Korean→English | 72,637 | 0.072 | 0.323 | 0.352 | 0.337 |
| German (both L2s) | 72,202 | 0.119 | 0.281 | 0.368 | **0.319** |

**Relatedness helps, but it is not the whole story.** Italian was chosen as the
optimistic case — Spanish and French both stay in training, so a Romance
language should be nearly free — and scores **0.430**. Korean was chosen as the
hard case — no Koreanic language in training, a script no trained pair uses —
and scores **0.337**, the lowest of the English-L2 pairs, as §1.4 predicted.
(Under the earlier dict-freq labels the two were tied at 0.383 and 0.389; the
cleaner Italian labels are what separated them.) The prediction that failed is
the narrow one: Hindi, held out with the corpus's highest switch rate and no
reliable script cue, tops the group at 0.447 rather than Italian.

*The model learned "anticipate English", not "anticipate a switch".* German is
the weak pair, and it is the only held-out language with two second languages.
Splitting it (`scripts/eval_anticipation_pair.py`) shows the two are not the
same problem at all:

| Held-out German pair | n | Base | P | R | F1 | `always` | `hazard@tuned` |
|---|---:|---:|---:|---:|---:|---:|---:|
| German→**English** | 32,290 | 0.120 | 0.452 | 0.431 | **0.441** | 0.214 | 0.248 |
| German→**French** | 39,912 | 0.119 | 0.198 | 0.316 | **0.243** | 0.212 | 0.198 |

Both are unseen first languages, so the difference is *entirely* the second
language. On German→English the model beats the best training-free baseline by
+0.193. On German→French it scores **0.243 against 0.212 for "always predict a
switch"** — precision collapses to 0.198. That is the trivial baseline with a
little signal on top.

**So the universality claim has to be stated carefully:** switch anticipation
transfers across *first* languages, including to unrelated scripts and
families. It does **not** transfer to an unseen *second* language. That is not
surprising in hindsight — 14 of 15 pairs share English as L2, so "the other
language" and "English" are nearly the same variable in training — but it means
the corpus as it stands cannot support a claim of full universality, and
German→French is the single pair that exposes it. §5.3 treats this as a data
problem, not a model problem.



---

### 4.6 Sub-word units and 3-class duration: training-free baselines

The same training-free baselines, rerun (2026-09-28) with the 3 required duration classes (Small 1–2, Medium 3–6, Large 7+), on words and on XLM-R pieces. Test split; the pieces' labels are built as in §5.1.

| Switch (`y_sw`) | Words: P / R / **F1** | Pieces: P / R / **F1** |
|---|---|---|
| `always` | 0.098 / 1.000 / 0.178 | 0.104 / 1.000 / 0.188 |
| `naive_l2` | 0.189 / 0.457 / 0.267 | 0.147 / 0.457 / 0.222 |
| `hazard@tuned` | 0.254 / 0.352 / **0.295** | 0.180 / 0.338 / **0.235** |
| hazard on L1→L2 | F1 0.000 | F1 0.000 |

| Duration (oracle switch positions) | Words | Pieces |
|---|---|---|
| Scored switches (class certain) | 179,741 | 180,990 |
| Mode per direction: 3-class accuracy | 0.585 | 0.509 |
| … macro-F1 over the classes | 0.428 | 0.225 |
| … within one class | 0.766 | 0.732 |

Three readings:

- **Pieces are harder for the rules.** A word split into several pieces adds "no switch" positions in the middle of the word, and `naive_l2` then fires on every English piece instead of every English word, so its precision drops from 0.189 to 0.147. The finding from §4.3 holds at both units: the run-length hazard still scores **0.000** on switching *into* the second language.
- **Large dominates on pieces.** Counting in pieces lengthens runs, so the most common class in both directions becomes Large, and the mode baseline predicts Large everywhere (macro-F1 0.225). On words, switches into L2 are most often Small (a one- or two-word insertion) and switches back into L1 most often Large, the same asymmetry as §4.4.
- **The pipeline works end to end.** A 400-step sub-word smoke run, with pieces only and no language input, passes the causality check at 0.00e+00 drift, and reaches switch F1 0.259. That figure is on only the first 50 test batches (124,800 positions, about 7% of the split, 57% of them Cantonese), with the threshold tuned on 50 validation batches, so it is **not** comparable with the full-split baseline above. It shows the pipeline trains; the full runs, and the XLM-R backbone, come next.

### 4.7 All models on one scorecard

Since 2026-09-28, every model is scored with the same report (`src/evaluation.py`), and `scripts/compare_models.py` puts them side by side, overall and pair by pair. The current table is [`model_comparison.md`](model_comparison.md). Headlines:
- **The per-pair hazard** (switch F1 0.329 on words) is now the training-free bar, not the pooled 0.295.
- **The from-scratch transformer** reaches 0.513 on words and 0.453 on pieces, and is best on all 15 pairs.
- **German→French** remains the hardest pair for every model.

XLM-R with a causal mask is trained on Kaggle (`kaggle/train_xlmr.ipynb`), and its results file drops into the same table.

## 5. Next Steps — challenges with sub-word tokenization and Language-ID→segment mapping

### 5.1 Tokenization: the unit is not the same across languages

This is the largest unsolved problem, and it is upstream of the model.
`tokenize_mixed` splits Han, Kana and Hangul at the **character** level (they
are not whitespace-delimited) and Latin/Cyrillic/Arabic/Devanagari at the
**word** level. So a "token" in a Cantonese row is roughly a morpheme, and a
token in a Spanish row is a word. Consequences:

- **Run lengths are not comparable across pairs.** First-language runs average
  16.7 tokens for Cantonese and 4.0 for Hindi. Some of that is real; an unknown
  part of it is the unit. `y_dur` inherits the problem directly.
- **Vocabulary pressure is wildly uneven.** At a 50,000-type word vocabulary,
  OOV on the test split:

| Russian | Arabic | Italian | German | Spanish | Hindi | French | … | Vietnamese | Korean | Chinese | Japanese | Cantonese |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 12.57% | 11.26% | 6.59% | 6.12% | 5.42% | 4.99% | 4.89% | | 1.28% | 0.59% | 0.57% | 0.40% | **0.28%** |

  Overall 1.63% — but that average is Cantonese. Morphologically rich languages
  are starved exactly where the word-level model needs them most, which is a
  direct threat to the universality claim: the current model may look universal
  only because 51% of the corpus is the easy case.

**Planned fix, and the new problem it creates.** Move to a shared subword
vocabulary (XLM-R sentencepiece) so every language is segmented by the same
mechanism. That reopens the alignment question, with a twist specific to this
task:

- In the *tagging* pipeline we used **first-subword-wins** — standard for token
  classification. For *anticipation* that is the wrong choice and arguably
  anti-causal: if a word is `neuro|plastic|ity` and we predict at `neuro`, we
  are predicting from a prefix that excludes the rest of a word the reader has
  already seen. **Last-subword-wins** is the causally correct pooling for this
  task, and it is a different alignment from the one the labels were built with.
- A switch boundary can fall *inside* a subword. XLM-R's sentencepiece will
  happily merge across a language boundary when the surface string permits it,
  which makes `y_sw` ambiguous at exactly the positions we care about most.
- CANINE (character-level, no vocabulary) would sidestep both. We tested it on
  romanized Hindi and it gave **no advantage** — 97.1% vs mBERT's 97.3% under a
  spelling stress test at n = 6,890 (Appendix A §3.8). That was a tagging
  result, not an anticipation result, so it does not settle the question here,
  but it lowers our prior.

**Built (2026-09-27): sub-word token labels.** `scripts/build_subword.py`
writes `data/subword.parquet`. Every row keeps its readable XLM-R pieces, a
language per piece, and the labels `y_sw`, `y_len` (integer), `y_dur` (class)
and the censoring flags.

*Choosing the tokenizer.* We measured six candidates on 300 rows per pair:

| Tokenizer | Problem on our corpus |
|---|---|
| **XLM-R** (250 K SentencePiece) | none: 0% UNK, 0% byte fallback on all 15 pairs, 0.93–1.66 pieces per unit |
| mBERT | 2.8% UNK on Cantonese |
| XGLM | 4.8% UNK on Cantonese |
| BLOOM | 45% byte pieces on Korean; 2.46 pieces per word on Russian |
| mGPT | 19% byte pieces on Cantonese |
| Qwen2.5 | 25% byte pieces on Cantonese, 23% on Hindi |

*How labels reach the pieces.* We tokenize the original SwitchLingua text, not
a re-joined copy. Each piece takes the language of the word its characters
fall in, and names are resolved first, so a name's pieces keep their
`script_words` language. Pieces with no letters (punctuation, digits, and a
bare `▁` that XLM-R emits with the next character's offsets) are transparent.

*The second bullet above turned out not to be a problem.* Only **155 of
21,740,404 pieces (0.0007%)** straddle a language boundary, because XLM-R's
pre-tokenizer splits at whitespace and script changes. These pieces take the
majority language.

*CJK.* XLM-R merges characters in 24–51% of CJK pieces (Cantonese 24.4%,
Korean 24.8%, Japanese 45.5%, Chinese 50.7%). A merged piece such as 東京
becomes one unit, so pieces, not characters, are the unit there. This moves CJK
units toward word size, but run lengths are still not comparable across
scripts; that needs word segmentation (§5.3 item 5).

*Result.*

| Quantity | Value |
|---|---|
| Rows | 234,172, with 0 alignment failures |
| Pieces | 21,740,404: 56.9% L1, 27.9% L2, 15.2% no language |
| Scored pieces | 18,211,198 |
| Switches | 1,896,074 (10.41%); the word-level count is 1,896,078, so the boundaries carried over intact |
| Durations of known class | Small 26.8%, Medium 22.3%, Large 50.9%. This includes 145,049 cut-off runs of 7+; counting closed runs only, the split is 29.1% / 24.3% / 46.6% |

The last-subword-wins question in the first bullet is a modelling choice, and
moves to the Week 5 backbone work. There, the sub-word model will get no
language input, because a word's label depends on pieces not yet read. Some
rows exceed XLM-R's 512-position limit (the longest has 1,689 pieces), so the
existing overlapping-window streaming is required, not optional.

### 5.2 Language-ID → segment mapping: label noise sets the ceiling

`y_sw` is only as good as the per-token language labels it is derived from, and
**a single mis-tagged token creates two spurious switches**. The worked example
in §2.2 is a real instance: under the earlier dict-freq labels, `"like a joke"`
was split in three because `"a"`, a very frequent Spanish preposition, was
tagged Spanish, inventing two positives inside one English run.

We can measure how often this happens without any new annotation. Count
**interior single-token L1 runs bracketed by L2 on both sides** — the exact
signature of the error above. For the six script-disjoint pairs the labels are
*exact by construction*, so their rate is the genuine linguistic rate. The last
two columns give the share of switches spent on islands (two per island) under
the earlier dict-freq labels and under the final AnE-LID labels:

| | Language | Switches | L1 islands | dict-freq | **final** |
|---|---|---:|---:|---:|---:|
| **exact** | Chinese | 58,833 | 301 | 1.0% | **1.0%** |
| | Japanese | 55,896 | 410 | 1.4% | **1.5%** |
| | Korean | 52,236 | 587 | 2.2% | **2.2%** |
| | Russian | 46,348 | 768 | 3.3% | **3.3%** |
| | Cantonese | 1,050,873 | 17,920 | 3.4% | **3.4%** |
| | Arabic | 48,527 | 966 | 4.0% | **4.0%** |
| **AnE-LID** | Spanish | 46,921 | 372 | 15.3% | **1.6%** |
| | French | 45,494 | 582 | 14.5% | **2.6%** |
| | Italian | 45,454 | 623 | 11.4% | **2.7%** |
| | Vietnamese | 61,001 | 1,206 | 7.1% | **4.0%** |
| | German (→English AnE, →French dict-freq) | 86,831 | 2,664 | 11.3% | **6.1%** |
| | Malay | 73,546 | 5,128 | 14.1% | **13.9%** |
| | Philippines | 105,329 | 12,106 | 26.8% | **23.0%** |
| **mixed** | Hindi | 118,789 | 16,486 | 28.5% | **27.8%** |

The exact pairs sit in a **1.0–4.0%** band. Under dict-freq every Latin-script
pair sat above it, up to 7× higher — shared-function-word tagging error,
exactly the failure the original pipeline plan predicted for Romance pairs.
**AnE-LID removes most of it:** Spanish, French, Italian and Vietnamese now fall
inside the exact band, and Spanish alone lost 17,432 switches, most of them
spurious. This
matches the automated accuracy checks — on human-annotated LinCE
Spanish–English, AnE-LID scores 99.4% against dict-freq's 94.1%.

**Three pairs stay high: Malay, Philippines and Hindi (14–28%).** For Hindi,
AnE-LID scores 98.8% on human-annotated LinCE Hindi–English, which is too
accurate to produce a 28% island rate by itself. The likelier explanation is
genuine style: Hinglish, Taglish and Manglish routinely drop a single L1
function word between English content words (`"movie ki story"`). That is a
real switch pattern, not noise, but there is no human-annotated Tagalog or
Malay set to confirm it for those two pairs, so their per-pair F1 is still
`provisional`.

**What this means for §4.** The Latin-pair ceiling that the dict-freq labels
imposed is largely gone. The overall numbers barely moved (0.509 → 0.503),
since half the test positions are Cantonese, whose labels hardly changed.
Per pair, Spanish and French rose (0.465 → 0.486, 0.498 → 0.509) and Italian
fell (0.490 → 0.469). All three are now measured against labels about as clean
as the script-exact ones.

### 5.3 Ranked next actions

1. ~~Run the human review~~ **Done differently.** Hand review was replaced by
   automated evaluation: human-annotated public sets (`scripts/eval_public.py`:
   LinCE Spanish–English and Hindi–English, Denglisch) plus corpus-wide
   agreement between two independent taggers (`scripts/consensus_corpus.py`).
   That moved production to AnE-LID and removed most of the Latin-pair noise
   (§5.2). The remaining gap is Tagalog and Malay, which have no public
   human-annotated set.
2. **Get a second non-English second language into the data.** §4.5 shows the
   model scores 0.243 on German→French — barely above the trivial baseline — while
   reaching 0.441 on German→English from the same held-out first language. With 14 of 15
   pairs sharing English as L2, the corpus cannot distinguish "anticipate a
   switch" from "anticipate English", and no modelling change fixes that. Two
   options: mine SwitchLingua's other releases for non-English pairs, or
   re-frame the label as *language-identity-agnostic* — predict "the other
   language" without letting the model see which language that is. The second
   is free and testable this week; it is the cheaper experiment and should come
   first.
3. **Re-run §4 restricted to `exact`-quality pairs only.** This gives a
   noise-free universality number, at the cost of dropping the Latin pairs.
   It matters less now that most Latin pairs are inside the exact noise band. If macro-F1 holds up there, the label noise is the explanation for the
   gap; if it does not, the model is the explanation.
4. **Subword rebuild with last-subword-wins pooling** (§5.1), measured against
   the current word-level model on the same splits. *The labels are built
   (§5.1); training on them is the remaining step.*
5. **A CJK word-segmentation pass** (`jieba` / `fugashi`, both already in
   `requirements.txt`) so run lengths are in comparable units. This changes
   `y_dur` for 51% of the corpus and is the prerequisite for any cross-language
   duration claim.
6. **Scale the causal model.** The §4.5 baseline is a from-scratch 15 M-parameter
   transformer trained for **1.02 epochs** — it reaches F1 0.503 knowing nothing
   about any language beyond what 192,000 windows taught it. A pretrained causal
   LM (or a causally-masked XLM-R) starts with exactly the multilingual lexical
   knowledge §4.3 identifies as the missing ingredient for the L1→L2 direction,
   and the word-level 50 K vocabulary that currently starves Russian and Arabic
   (§5.1) disappears with it. This is the cheapest remaining source of headroom,
   but it is ranked below the label work deliberately: a better model measured
   against noisier labels tells us less, not more.

---

## 6. Reproducing this update

```bash
cd ~/Northeastern_GitHub/code-switch          # use ./.venv/bin/python throughout

./.venv/bin/python -m pytest tests/ -q                       # 210 tests
./.venv/bin/python scripts/baseline_anticipation.py          # §4.2-§4.4, ~4 min
./.venv/bin/python scripts/shuffle_parquet.py                # breaks the language curriculum
./.venv/bin/python scripts/train_anticipation.py --build-vocab \
    --path data/tagged_shuffled.parquet --steps 6000 \
    --out data/anticipation_model_shuffled.json    # §4.5 headline, ~23 min on an M3 (MPS)
./.venv/bin/python scripts/train_anticipation.py --steps 6000 \
    --path data/tagged_shuffled.parquet \
    --train-langs Cantonese,Arabic,Philippines,Chinese,Vietnamese,Malay,Japanese,Russian,Spanish,French \
    --out data/anticipation_zeroshot.json --ckpt models/anticipation_zeroshot.pt

# split the one held-out language that has two second languages; a
# first_language filter merges them and hides the result that matters
./.venv/bin/python scripts/eval_anticipation_pair.py \
    --ckpt models/anticipation_zeroshot.pt --l1 German --l2 French
./.venv/bin/python scripts/eval_anticipation_pair.py \
    --ckpt models/anticipation_zeroshot.pt --l1 German --l2 English

# optional: the unshuffled ablation in §4.5
./.venv/bin/python scripts/train_anticipation.py --steps 6000 \
    --out data/anticipation_model.json
```

| Module | Role |
|---|---|
| `src/anticipation.py` | `y_sw` / `y_len` / `y_dur` generation, 3 duration classes, censoring |
| `src/streaming.py` | streaming `IterableDataset`, splits, windowing, vocab |
| `src/causal_model.py` | causal transformer + `check_causality` |
| `src/metrics.py` | Anticipatory F1, macro-F1, Wilson intervals, duration scores |
| `scripts/baseline_anticipation.py` | training-free baselines |
| `scripts/train_anticipation.py` | causal transformer training + zero-shot eval |
| `scripts/eval_anticipation_pair.py` | score a checkpoint on one exact L1→L2 pair |
| `scripts/shuffle_parquet.py` | one-off row shuffle that breaks the language curriculum |
| `tests/test_anticipation.py` | 15 tests: labels, windowing, splits, causality |

| Artifact | Contents |
|---|---|
| `data/tagged.parquet` | 234,172 rows · 22,349,157 labelled tokens · 67 MB |
| `data/tagged_shuffled.parquet` | the same rows, permuted (seed 0) — what the model trains on |
| `data/baseline_anticipation.json` | every number in §4.2–§4.4 |
| `data/anticipation_model_shuffled.json` | §4.5 headline — the row-shuffled run |
| `data/anticipation_model.json` | §4.5 ablation — the unshuffled (curriculum) run |
| `data/anticipation_zeroshot.json` | §4.5 zero-shot — 10 pairs trained, 4 held out |
| `data/vocab_anticipation.json` | 50,000 word types, train split only |
| `models/anticipation_causal.pt` | trained causal transformer + tuned threshold |
| `models/anticipation_zeroshot.pt` | the 10-pair model, no pair embedding |

**Appendix A — `docs/labelling_pipeline.md`** documents where the per-token
language labels come from: eight techniques evaluated, the production routing,
and the accuracy evidence. Read it for anything about label quality; §5.2 above
is the part that constrains this update.
