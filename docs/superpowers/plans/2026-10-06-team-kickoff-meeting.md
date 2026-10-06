# Team Kickoff: Start Here (2026-10-06)

Written for teammates who are new to the project. Read sections 1–4 before the meeting (about 10 minutes). Section 5 is your task for this week.

---

## 1. What this project is, in one minute

People who speak two languages often mix them in one sentence. This is called **code-switching**:

> El negocio está creciendo rápidamente. **Success feels like a joke when** …

Existing tools spot a switch *after* reading the whole sentence. We ask a harder question: **reading word by word, with nothing after the current word visible, can a model predict that the next word will switch language?** If it can, it also predicts **how long** the new language lasts: Small (1–2 words), Medium (3–6) or Large (7+).

- **Data:** SwitchLingua, 234,172 conversations in 15 language pairs (Cantonese, Chinese, Japanese, Korean, Arabic, Russian, Hindi, Spanish, French, Italian, German, Vietnamese, Malay, Tagalog, each mixed with English, plus German mixed with French).
- **Why it's hard:** only about 10% of words are followed by a switch, so a model that always says "no switch" is 90% accurate and useless. We score with **F1 on the switch class**: 0 is useless and 1 is perfect.
- **The research question:** is switch prediction *universal*? Does a model trained on some language pairs also work on pairs it has never seen? We test this by hiding four languages (Korean, Italian, Hindi, German) from training and scoring them afterwards.

## 2. How we labelled the languages

SwitchLingua gives us text but no per-word language labels. Every model needs a label for every word ("this word is Spanish, this one is English"), and the switch and duration targets are computed from those labels. So labelling came first.

**No label was written by hand.** Each language pair goes to whichever method is most accurate for it, and every method is scored against public datasets that humans labelled word by word.

| Language pairs | Method | How it works | Accuracy |
|---|---|---|---|
| Cantonese, Chinese, Japanese, Korean, Russian, Arabic (and Hindi in Devanagari script) | **Script lookup** | The alphabet tells you the language: Chinese characters → Chinese, Latin letters → English | Exact, no model needed |
| Spanish, French, Italian, German, Vietnamese, Malay, Tagalog, romanized Hindi (all mixed with English) | **AnE-LID**, a neural model (XLM-R-large fine-tuned on code-switched text) | Reads the whole sentence and decides, for each word, English or not | **99.4%** on human-labelled Spanish–English, **98.8%** on Hindi–English |
| German–French | **Word frequency** (`wordfreq`) | Is this word more common in German or in French? | 96% on German–English (no public German–French test set exists) |
| Names, everywhere | **AnE-NER** | Marks names like "Messi" or "Game of Thrones", so a name alone doesn't count as a switch | Checked on 20 tricky cases |

**Why the script method and the Latin languages need different tools.** With different alphabets the answer is free. When both languages use Latin letters, a word like *real* or *mas* is ambiguous on its own, and only the words around it settle it. That is why the Latin-script pairs need a model that reads context.

**What we tried and dropped:**

| Method | Result | Why we dropped it |
|---|---|---|
| fastText language ID | 83% on our check set | Judges each word alone, with no context, so it fails on short ambiguous words |
| GlotLID | rejected | Same limitation as fastText, and a 1.6 GB model |
| Dictionary lookup ("is it in the English dictionary?") | 52% | Most common words exist in both languages |
| mBERT code-switch taggers | 98.8% / 97.4% on human data | Good, but AnE-LID beat it on both benchmarks |
| Our own fine-tuned XLM-R and CANINE taggers | no gain | Each was trained on one language pair and didn't transfer to the others |

So "the good model" (AnE-LID) won on **accuracy**. It isn't the cheapest option; the cheap ones were the ones that made mistakes. The full history is in `docs/labelling_pipeline.md`, and a one-page summary is in `docs/labelling_models.md`.

## 3. What is in the Hugging Face dataset

`cristae/switchlingua-tagged-final` (private) holds the finished labels, ready for training:

| File | What's in it |
|---|---|
| `token_labels_word.parquet` | one row per conversation, one label per word: language (L1/L2), `y_sw` (does the next word switch?), `y_len` / `y_dur` (how long the new run lasts) |
| `token_labels_subword.parquet` | the same labels per XLM-R piece, the units a transformer actually reads |
| `tagged.parquet`, `switchlingua_tagged_rows.csv` | the raw per-word languages |

**Train/val/test split:** by conversation, `row_id % 10`: 0–7 train, 8 val, 9 test. Everyone uses this split so that results stay comparable.

## 4. What we have found so far

| Model | Switch F1 | Notes |
|---|---|---|
| Simple rules (how long the current language has run) | 0.27 | no learning; the bar to beat |
| Small transformer trained from scratch | 0.45 | |
| **XLM-R** (pretrained multilingual) with a "no peeking ahead" mask | **0.65** | best; 4 pairs above 0.7 |
| mBERT, same setup | 0.63 | XLM-R wins on 14 of 15 pairs |
| XLM-R on the 4 languages it never saw | **0.63** | only about 13% worse than when trained on them: switch prediction mostly transfers |

**What's still open, and why we need four people:**
- tuning the best model;
- repeating runs with different random seeds to get error bars;
- working out *why* some pairs (Malay, Tagalog, Vietnamese) are hard;
- the paper;
- a live demo.

---

## 5. Your task this week

### Do you run your own backbone? Yes, but not the same one

Each of you trains a model on a free Kaggle GPU. Don't all rerun the same XLM-R setup: two identical runs already agree to within ±0.004, so a fourth copy tells us nothing. Each person runs **one different configuration**, so the team gets four new results for the price of one run each.

### Step 1: Accounts (today)

1. **Hugging Face:** create an account at huggingface.co and send your username to Cristae, who adds you to the dataset (see section 6).
2. **SwitchLingua access:** open [`Shelton1013/SwitchLingua_text`](https://huggingface.co/datasets/Shelton1013/SwitchLingua_text) and click "request access". **Do this today**, because approval can take time and the training notebook needs it.
3. **HF token:** HF → Settings → Access Tokens → new **Read** token. Keep it private.
4. **Kaggle:** create an account and **verify your phone number**; Kaggle only allows GPUs after phone verification. You get 30 free GPU hours a week.

### Step 2: Look at the labels (30 min, no GPU)

1. On Kaggle: Create → New Notebook → File → Import Notebook → upload `kaggle/load_labels.ipynb` from the repo.
2. Turn Internet **on**. Add your token under Add-ons → Secrets, named exactly `HF_TOKEN`.
3. In the config cell, set `PAIRS` to your group's pairs and `LIMIT = 200`, then Run All.
4. Read a few conversations with their labels. Write down 3 things you notice about your pairs, such as where switches tend to happen or anything that looks mislabelled.

### Step 3: Smoke test the training notebook (15 min of GPU)

1. Import `kaggle/train_pretrained.ipynb` the same way. Settings: Accelerator **GPU T4 ×2**, Internet **on**, secret `HF_TOKEN`.
2. In the config cell:
   - Check that `MODEL = "xlmr"` (the default).
   - Set `SMOKE = True`.
3. Run All. Success means the output shows `PREFIX-ONLY OK` (the no-peeking check) and a small results table at the end. If it fails, post the error in the group chat.

### Step 4: Your real run (~1.2–2.5 h of GPU)

1. Add a row to `docs/RUN_LOG.md` (or post it in the chat) **before** you start, so nobody duplicates it.
2. Set `SMOKE = False` and your configuration from the table below. Then **Save Version → Save & Run All**, so the run continues after you close the browser.
3. When it's done, open the version's **Output** tab and download `data/anticipation_<TAG>.json` and `data/preds_<TAG>.npz`. Bring the headline numbers (switch F1, macro-F1, duration macro-F1) and your pairs' rows to the next meeting.

| Person | Configuration (in the notebook's config cell) | The question it answers |
|---|---|---|
| 1 | `MODEL="xlmr"`, all tuning flags off, `SEED=0` | A clean baseline with saved predictions, for a fair comparison with mBERT |
| 2 | `MODEL="xlmr"`, `DUR_WEIGHTS="balanced"`, `PREV_LANG=True`, `PREDICT_AT="word_end"`, `SEED=0` | Do the three tweaks that helped the small model also help XLM-R? |
| 3 | Same as person 2, but `SEED=1` | How much do results move from randomness alone? (error bars) |
| 4 | Same as person 2, plus `ZERO_SHOT=True` (`RUN_FULL=False`) | Do the tweaks also help on the 4 unseen languages? |

All four runs fit easily in one week's quota (30 h each).

### Step 5 (optional): Pick a pair group

The longer-term plan (`docs/TEAM_PLAN.md`) gives each person a group of language pairs to study in depth, plus one team-wide job. We'll choose at the meeting; look at the table and think about which you'd like.

| Group | Pairs | Team-wide job |
|---|---|---|
| A | Cantonese, Chinese, Japanese, Korean | Training lead |
| B | Spanish, French, Italian, German→English, German→French | Evaluation lead |
| C | Hindi, Tagalog, Malay, Vietnamese | Paper lead |
| D | Russian, Arabic | Demo lead |

---

## 6. For Cristae: before the meeting

- [ ] **Hugging Face sharing.** HF doesn't let a personal account share a private dataset with individual users; sharing goes through an organization.
  1. Create a free org (huggingface.co → New → Organization) and invite the three members with the **read** role.
  2. Move the dataset into it: dataset Settings → "Rename or transfer this dataset".
  3. Update the dataset ID in code: `scripts/download_labels.py`, `scripts/build_labels_notebook.py` and `scripts/build_pretrained_notebook.py`. Then rebuild the two notebooks and update the ID in `README.md`, `CONTRIBUTING.md`, `CLAUDE.md`, `docs/labelling_models.md` and `docs/TIMELINE.md`. HF redirects the old ID after a transfer, but don't rely on that.
- [x] Notebook default changed to `MODEL = "xlmr"` (`scripts/build_pretrained_notebook.py`, notebook rebuilt).
- [x] Stale lines fixed: `TEAM_PLAN.md` now says Week 5; `TIMELINE.md` strikes the two finished gaps and adds a 2026-10-06 pace check.
- [ ] Share this file with the team before the meeting.

## 7. Meeting agenda (30 min)

1. **The project in 5 minutes** (Cristae): sections 1, 2 and 4. Show one labelled conversation from `load_labels.ipynb`.
2. **Questions on the labels** (5 min).
3. **Assign the four runs** (5 min): who is person 1–4 in the Step 4 table.
4. **Pick pair groups and team-wide jobs** (10 min). Fine to postpone to next week.
5. **Close** (5 min). Access problems go in the chat *today*. Next check-in in one week, where each person brings their run's numbers, the 3 things they noticed and any blocker.

## 8. Rules worth knowing from day one

- **Keep the data private.** SwitchLingua is gated, so never post labels, conversation text or your token publicly, and never commit them to git.
- **No hand labelling.** If you find a labelling mistake, report the pattern; we fix it in code and rerun.
- **Never report accuracy.** Report F1 on the switch class.
- **Thresholds are tuned on val; test is scored once.** The notebook already does this.
