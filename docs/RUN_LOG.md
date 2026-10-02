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
