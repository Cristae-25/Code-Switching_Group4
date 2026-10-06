# Team plan: who owns what, and when

The pipeline is built: the labels, the evaluation, the baselines, the from-scratch model, XLM-R and mBERT, and the tuning flags. From here the work splits **by language pair**. Each member owns a group of pairs end to end (analysis, error study, one experiment, a paper section), plus **one cross-cutting duty** for the work that isn't per-pair. Everyone uses the same shared tools and conventions ([`../CONTRIBUTING.md`](../CONTRIBUTING.md)), so nobody rebuilds the same thing.

**Where we stand:** the scorecard is [`model_comparison.md`](model_comparison.md), the dated results are in [`results/`](results/), and the schedule is [`TIMELINE.md`](TIMELINE.md).

- **Best model:** XLM-R with a causal mask. Switch F1 0.646, zero-shot 0.625, ahead of mBERT on 14 of 15 pairs.
- **Tuning flags that help,** tested locally: class-weighted duration loss, previous-word language, word-end prediction.

## Pair groups

| Group | Owner | Pairs | The group's research question | Cross-cutting duty |
|---|---|---|---|---|
| **A: CJK** | _name_ | Cantonese (half the corpus), Chinese, Japanese, Korean | CJK units are characters or pieces, not words, so run lengths aren't comparable with other scripts. Does word segmentation (`jieba` / `fugashi` are in `requirements.txt`) change duration results? Why does Korean transfer best zero-shot (0.775)? | **Training lead:** the final XLM-R runs on Kaggle (adopted flags, 3 epochs, zero-shot); merges PRs |
| **B: Romance + Germanic** | _name_ | Spanish, French, Italian, German→English, **German→French** | German→French is the only non-English L2 (zero-shot 0.450). How much "anticipate English" bias remains? Is near transfer free (Italian −0.010)? | **Evaluation lead:** seeded variance runs (Week 12) and the final scorecard (`compare_models.py`) |
| **C: South / Southeast Asian** | _name_ | Hindi (romanized + Devanagari), Tagalog (Philippines), Malay, Vietnamese | The highest switch rates, with many mid-clause insertions. Why are Malay (0.548), Vietnamese (0.622) and Tagalog (0.638) the hardest pairs? Tagalog and Malay labels have no human benchmark | **Paper lead:** structure, figures, assembling everyone's sections |
| **D: Cyrillic + Arabic** | _name_ | Russian, Arabic | Labels here are exact (the script gives the language away). Where does the model still fail when the label is certain? | **Demo lead:** the streaming demo (Week 14): type a sentence, watch the switch probability and the predicted duration update live |

Group D has fewer pairs because the demo is a bigger cross-cutting job.

## The per-pair checklist (the same for every owner)

Each item uses tools that already exist. Keep your findings in `docs/pairs/<pair>.md`, which `scripts/pair_report.py --md` starts for you.

1. **Know the data.** Read your pairs in [`../notebooks/switch_eda.ipynb`](../notebooks/switch_eda.ipynb): switch rate, hazard curves, duration classes, persona effects. Write 3–5 bullet findings per pair.
2. **Spot-check the labels.** Run `./.venv/bin/python scripts/review_server.py --lang <First_language>`, which is report-only. Look for *systematic* errors and fix them in code, never by hand. Labels come from three taggers: by script (exact), AnE-LID (98.8–99.4% on human data) and word frequency (German→French only). See [`labelling_models.md`](labelling_models.md).
3. **Study the errors.** Run `./.venv/bin/python scripts/pair_report.py --pair "<L1>-><L2>" --preds <preds.npz> --md docs/pairs/<L1>-<L2>.md`.
   - It shows every model's score for the pair, hits and misses by direction, and the most confident missed switches and false alarms as text. `┃` marks where the model was asked.
   - Sort the errors into types: label error, mid-clause insertion, names, tag phrases, and so on.
4. **Run one targeted experiment** for your group's question. Pick one, or propose your own, before running it:
   - **A specialist model:** fine-tune XLM-R only on your group's pairs (`--train-langs`) and compare it with the universal model on those pairs, which is Week 13's "best universal predictor" question.
   - **A:** CJK word segmentation, so durations count words.
   - **B:** German→French homographs and English tags (see "Starting leads").
   - **C:** whether the Tagalog/Malay errors are label errors (spot-check) or genuine mid-clause switches.
   - **D:** what the model attends to near a switch.
5. **Write your paper section:** a results table for your pairs (from `pair_report.py`) and 2–3 findings with examples.

## Starting leads

These came from a first look at German→French errors (a short run with `pair_report.py`):
- **Likely label errors on German/French look-alike words.** Positions like "Der *Film* ┃ sollte…" and "Im ┃ *Moment* sprechen…" are counted as switches, so *Film* and *Moment* were tagged French. They are spelled the same in both languages, and the frequency tagger used for this pair can't tell them apart. **Group B.**
- **English tag phrases inside German→French** ("…starten, *you know* ┃ ?") cause false alarms. **Group B.**
- **The per-pair hazard baseline on RoBERTa pieces scores 0 on German→French:** its global threshold never fires for that pair. It's a baseline quirk worth one sentence in the paper. **Evaluation lead.**

## Milestones

The calendar is in Week 5 now (Oct 5–11). Weeks follow [`TIMELINE.md`](TIMELINE.md).

| Week | Dates | Everyone (own pairs) | Cross-cutting |
|---|---|---|---|
| 5 | Oct 5–11 | checklist 1–2: EDA findings and the label spot-check | Training: XLM-R baseline and adopted-flags runs (the notebook saves predictions automatically). Demo: pick the framework |
| 6–7 | Oct 12–25 | checklist 3: error analysis on the new XLM-R predictions; propose the experiment | Training: 3-epoch run. Evaluation: add seeds to the plan. Paper: outline |
| 8–9 | Oct 26 – Nov 8 | checklist 4: run the experiment | Training: zero-shot run of the final setup. Demo: first working version |
| 10 | Nov 9–15 | finish the experiment; update `docs/pairs/` | Training: final checkpoints. Evaluation: freeze the protocol |
| 11–12 | Nov 16–29 | — | Evaluation: 3 seeds of the final model, variance and confidence intervals, macro averages |
| 13 | Nov 30 – Dec 6 | checklist 5: write the section | Evaluation: per-pair breakdown, the best universal model, specialist vs universal |
| 14 | Dec 7–13 | review the sections | Paper: assemble. Demo: finish |
| 15 | Dec 14–20 | presentation rehearsal | Final repo (README, results) and presentation |

## Kaggle budget

Each member has **30 GPU hours a week**, so the team has up to 120. Typical costs on a T4:
- XLM-R, 1 epoch (12,000 steps): about 1.2 h.
- The same run plus zero-shot: about 2.5 h.
- 3 epochs: about 3.6 h.
- A specialist run on one group's pairs is shorter, depending on `STEPS`.

Use `kaggle/train_pretrained.ipynb`, try `SMOKE = True` first, and **log every run in [`RUN_LOG.md`](RUN_LOG.md)** before starting it, so nobody spends hours repeating someone else's run.

## How we meet

- **Weekly check-in, 30 minutes:** each owner gives one thing learned about their pairs, one result, one blocker. Update `TIMELINE.md` afterwards.
- **One PR per task, reviewed by one other member** (see `CONTRIBUTING.md`).
- **Decisions** that change the protocol (the split, the metrics, the labels) go into the record in [`results/`](results/) before anyone acts on them.
