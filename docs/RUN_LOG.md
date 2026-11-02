# Run log

Every GPU (Kaggle) run, so nobody repeats someone else's run, and the team's 4 × 30 h/week quota is spent on purpose. **Add the row before you start**, then fill in the result. Local runs can be logged too.

`TAG` is what the notebook names the outputs: the model, plus the flags that are on, e.g. `xlmr_durw_prev_wend`. The results file is `data/anticipation_<TAG>.json`; the predictions are `data/preds_<TAG>.npz`.

| Date | Who | Where | TAG | Settings (seed, steps, flags, zero-shot) | Hours | Result (headline) | Record |
|---|---|---|---|---|---:|---|---|
| 2026-09-29 | Cristae | Kaggle T4 | `xlmr` | seed —, 12k steps, no flags, all pairs (run twice) | ~1.2 each | switch F1 0.646, macro 0.676; duration macro 0.575 | [results](results/2026-09-29_xlmr_causal.md) |
| 2026-09-29 | Cristae | Kaggle T4 | `xlmr` (zero-shot) | seed —, 12k steps, trained on 10 pairs | ~1.3 | held-out F1 0.625 | [results](results/2026-09-29_xlmr_causal_zeroshot.md) |
| 2026-09-30 | Cristae | Kaggle T4 | `mbert` + zero-shot | seed 0, 12k steps, no flags | ~2.5 | F1 0.634; held-out 0.565 | [results](results/2026-09-30_mbert_causal.md) |
| 2026-09-30 | Cristae | M3 (local) | ablations | seed 0, 6k steps, one flag at a time + combined | ~3.5 | combined flags: shared F1 +0.023, duration macro +0.073 | [results](results/2026-09-30_ablations.md) |
| _next_ | _training lead_ | Kaggle T4 | `xlmr` | seed 0, 12k steps, no flags (gives shared F1 + predictions) | ~1.2 | | |
| _next_ | _training lead_ | Kaggle T4 | `xlmr_durw_prev_wend` | seed 0, 12k steps, `DUR_WEIGHTS="balanced"`, `PREV_LANG=True`, `PREDICT_AT="word_end"` | ~1.2 | | |
| _next_ | _training lead_ | Kaggle T4 | `xlmr_ctx1_6pairs` | seed 0, 12k steps, `CONTEXT = 1` (current piece only), `PAIRS` = Cantonese, Chinese, Japanese, Korean, Russian, Arabic | ~1.2 | | [design](results/2026-10-06_current_token_xlmr.md) |
| 2026-10-12 | Cristae | Kaggle T4 | `xlmr_6pairs` | seed 0, 12k steps, `CONTEXT = None`, `PAIRS` = the 6 script pairs (the other 9 scored held out) | 1.2 | switch F1 0.645, macro 0.703; duration macro 0.581 (val-tuned 0.602); held-out 9 pairs 0.514 | [results](results/2026-10-12_xlmr_6pairs_duration.md) |
| 2026-10-12 | Cristae | Kaggle T4 | `xlmr_durw_6pairs` | as `xlmr_6pairs`, plus `DUR_WEIGHTS="balanced"` | 1.3 | switch F1 0.645 (+0.000 vs plain); duration macro 0.600 (+0.019, Medium +0.101), val-tuned 0.602 (same as plain) | [results](results/2026-10-12_xlmr_6pairs_duration.md) |
| 2026-10-12 | Cristae | Kaggle T4 | `xlmr_untrained_6pairs` | `kaggle/untrained_xlmr.ipynb`: pretrained XLM-R frozen, **random heads, 0 steps**; only the threshold and duration offsets are fit on val. `PAIRS` = the 6 script pairs | | switch F1 0.148, macro 0.139 (chance; always-switch 0.178); duration val-tuned macro 0.315; held-out 9 pairs 0.193 | [results](results/2026-10-12_xlmr_untrained_6pairs.json), [notes](team_meeting1.md) |
| _next_ | _training lead_ | Kaggle T4 | `xlmr_probe_6pairs` | same notebook: pretrained XLM-R frozen, **heads trained** (linear probe), 12k steps, `HEAD_LR = 1e-3` | ~0.8 | | |
| _on hold_ | _training lead_ | Kaggle T4 | `xlmr_ord_6pairs` | as `xlmr_6pairs`, plus `DUR_HEAD="ordinal"` (cumulative cuts; see the 2026-10-11 duration-decision record). Every run now also reports duration with offsets tuned on val (`duration_tuned`) | ~1.5 | | |
| 2026-10-11 | Cristae | M3 (local) | `ablation_ordinal`, `ablation_ordinal_durw` | as the 2026-09-30 ablations (scratch, pieces, seed 0, 6k steps), `--dur-head ordinal`, without and with `--dur-class-weights balanced` | ~3 | val-tuned duration macro-F1 0.536 vs softmax 0.548; Small↔Large errors 9.1% → 6.5%. Val tuning alone: 0.481 → 0.548 | [results](results/2026-10-11_duration_decision.md) |
| 2026-11-01 | Cristae | M3 (local) | `punct_breakdown` | `scripts/punct_breakdown.py xlmr_6pairs`: switches bucketed by what sits before the next word | 0.01 | 59% of switches sit behind punctuation or a turn; XLM-R F1 mid-clause 0.47 vs comma 0.62, sentence end 0.82, turn 0.85 | [results](results/2026-11-01_punct_breakdown.md) |
| 2026-11-01 | Cristae | M3 (local) | `grammar_breakdown` | `scripts/grammar_breakdown.py xlmr_6pairs`: mid-clause positions by word frequency, run length, clause position, length, direction | 0.01 | model probabilities track base rates in every group (no unused cue); 87% of mid-clause misses are switches out of L1 (F1 0.27) | [results](results/2026-11-01_grammar_breakdown.md) |
| _next_ | _training lead_ | Kaggle T4 | `xlmr_prev` | seed 0, 12k steps, `PREV_LANG=True`, all 15 pairs (`PAIRS=""`); compare with the seed-0 `xlmr` row above | ~1.2 | | |
| _next_ | _training lead_ | Kaggle T4 | `xlmr_pair` | seed 0, 12k steps, `PAIR_EMB=True` (15-pair embedding, zero-initialised), all 15 pairs | ~1.2 | | |
| _next_ | _training lead_ | Kaggle T4 | `xlmr_lora16` | seed 0, 12k steps, `LORA_R=16`, `LORA_LR=2e-4` (2.7 M of 280 M params train), all 15 pairs; also `ZERO_SHOT=True` to test less forgetting on held-out pairs | ~2.5 | | |
| _next_ | _training lead_ | Kaggle T4 | `xlmrL_lora16` | as `xlmr_lora16` with `BACKBONE="xlm-roberta-large"`; `SMOKE=True` first: if batch 16 runs out of memory, use `BATCH=8` and say so (half the examples per step) | ~3 | | |
| _next_ | _training lead_ | Kaggle T4 | `xlmr_36k_rs` | seed 0, `STEPS=36_000` (3.07 epochs on 15 pairs), `RESHUFFLE=True`, `EVAL_EVERY=3000`, no other flags; is one epoch enough? Compare with the seed-0 `xlmr` row (`--reshuffle-epochs` changes only the last 2.4% of a 12k run) | ~3.8 | | |
