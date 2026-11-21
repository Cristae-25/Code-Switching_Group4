# 2026-11-20: LinCE transfer test of the final models (local, MPS)

**Why:** every number so far comes from SwitchLingua, which is generated dialogue. LinCE has **real tweets with human word-level language tags** in two of our pairs: `spaeng` (Spanish→English) and `hineng` (romanized Hindi→English). The labels come straight from the gold tags (`src/lince.py`), so this measures the models alone, with no tagger error. No model was trained on LinCE, so both the `validation` and the `train` splits are test data.

**Models:** `xlmr_prev_gap_36k_rsv2` and `mbert_prev_gap_36k_rsv2`, the same checkpoints as the [error breakdown](2026-11-20_error_breakdown.md) (seed 0, 36k steps, `decide_at=gap`, `--prev-lang`, 3-class duration head). Units are each model's own pieces, so XLM-R and mBERT are scored on slightly different position counts. The comparison between them is paired by tweet.

**How:**
```
./.venv/bin/python scripts/training/eval_lince.py --ckpt models/anticipation_<TAG>.pt --save-preds data/preds/lince_<TAG>.npz
./.venv/bin/python scripts/analysis/error_breakdown.py <TAG> --lince        # gap buckets on LinCE
```
These write `data/results/lince_<TAG>.json` and `data/results/lince_breakdown_<TAG>.json`. The headline number uses each checkpoint's **SwitchLingua-val τ** (0.413 / 0.430). The "oracle" row re-tunes τ on LinCE validation and scores LinCE train, which gives an upper bound on what recalibration alone can recover. Duration is scored at true switches whose class is known. Short/Large collapses the 3-class head with the val-tuned P(large) cut (0.49 / 0.51).

Data: 29,929 tweets (spaeng 3,332 val + 21,030 train; hineng 744 val + 4,823 train). 28.9 k of them have a scored position, giving 0.38 M scored XLM-R pieces and 23,330 switches.

## 1. Switch prediction: a large drop

| | XLM-R | mBERT |
|---|---:|---:|
| **SwitchLingua test, Spanish→English** | 0.839 | 0.842 |
| **SwitchLingua test, Hindi→English** | 0.775 | 0.774 |
| LinCE val, Spanish→English | 0.263 | 0.262 |
| LinCE val, Hindi→English | 0.325 | 0.319 |
| LinCE train, Spanish→English | 0.266 | 0.229 |
| LinCE train, Hindi→English | 0.302 | 0.286 |
| **LinCE all, pooled F1** (P / R) | **0.279** (0.264 / 0.295) | 0.253 (0.232 / 0.278) |
| LinCE all, into L2 / into L1 | 0.218 / 0.333 | 0.199 / 0.299 |
| Always-switch F1 (LinCE base rate 0.06) | 0.116 | 0.107 |
| Oracle: τ re-tuned on LinCE val → train | 0.292 (τ 0.232, from 0.277) | 0.259 (τ 0.296, from 0.248) |

- **The models transfer, but weakly.** On tweets, F1 falls from 0.78–0.84 to 0.26–0.33. That is still 2.4× the always-switch floor, so the models carry real signal.
- **Recalibration is not the problem.** Re-tuning τ on LinCE gains only +0.015 / +0.011, and precision and recall are already balanced at the SwitchLingua τ. The loss is in what the models predict, not where the threshold sits.
- **XLM-R transfers better than mBERT,** although in-domain the two are tied. The paired tweet-level bootstrap (2,000 resamples, 28,863 tweets) gives XLM-R 0.279 [0.272, 0.284] against mBERT 0.253 [0.247, 0.259], a difference of **+0.026 [+0.021, +0.030]**. That is the first clear separation between the two backbones.
- Hindi transfers slightly better than Spanish (0.31 vs 0.27). Its tweets have more mid-clause switches, where the Hindi model is strongest in-domain.

## 2. Why: three differences between tweets and SwitchLingua

Switch mix and per-bucket F1 for XLM-R (SwitchLingua test → LinCE all):

| Spanish→English | Share of switches (SL → LinCE) | Switch rate (SL → LinCE) | F1 (SL → LinCE) |
|---|---:|---:|---:|
| adjacent (mid-clause) | 0.22 → **0.63** | 0.022 → 0.040 | 0.41 → 0.19 |
| comma | 0.28 → 0.07 | 0.54 → 0.20 | 0.86 → 0.27 |
| sentence end | 0.19 → 0.16 | **0.88 → 0.29** | 0.96 → 0.39 |
| turn | 0.30 → 0 | 0.83 → — | 0.98 → — |

| Hindi→English | Share of switches (SL → LinCE) | Switch rate (SL → LinCE) | F1 (SL → LinCE) |
|---|---:|---:|---:|
| adjacent | 0.67 → **0.75** | 0.154 → 0.059 | 0.71 → 0.34 |
| comma | 0.13 → 0.03 | 0.66 → 0.12 | 0.89 → 0.22 |
| sentence end | 0.04 → 0.11 | **0.89 → 0.18** | 0.95 → 0.27 |

1. **Punctuation stops being a cue.** In SwitchLingua, a sentence end is followed by a switch 88–89% of the time; in tweets, 18–29%. The model learned "sentence end → switch" (section 2 of the error breakdown), and on tweets that rule turns into false alarms. F1 at sentence ends falls from 0.96 to 0.39.
2. **Most tweet switches are mid-clause** (63–75%), and tweets have no speaker turns. That is the hardest bucket in-domain too. Even there, F1 roughly halves, because single-word insertions such as "El room huele…" and "Baby solo dime…" are rare in SwitchLingua's longer, planned switches.
3. **60% of the tweets never switch.** SwitchLingua rows are generated to code-switch, so the model has never seen a long monolingual stretch. Monolingual tweets hold 51% of the positions and **53% of the false alarms**. A typical high-confidence false alarm is an all-English tweet ("Best wishes ‖ for the movie…", "I am feeling pity ‖ for…"): the model expects the English run to end and return to L1, which is SwitchLingua's insertion pattern. On the switching tweets only, XLM-R's F1 is 0.351 (mBERT 0.327).

Mix-standardising doesn't rescue LinCE, because F1 drops *within* every bucket, not only in the mix. The domain shift is in the base rates themselves.

## 3. Duration

| | XLM-R | mBERT |
|---|---:|---:|
| SwitchLingua test, 3-class macro-F1 (Spanish / Hindi) | 0.658 / 0.597 | 0.667 / 0.611 |
| LinCE all, 3-class macro-F1 (accuracy) | **0.442** (0.488) | 0.427 (0.453) |
| — Small / Medium / Large F1 | 0.598 / 0.226 / 0.503 | 0.525 / 0.258 / 0.498 |
| — Spanish / Hindi | 0.452 / 0.384 | 0.409 / 0.427 |
| LinCE all, Short/Large macro-F1 (accuracy) | **0.586** (0.652) | 0.572 (0.628) |
| SwitchLingua test, Short/Large macro-F1 | 0.818 | 0.830 |
| Share of Large among LinCE switches | 0.37 | 0.39 |
| n (known-class switches) | 15,731 | 16,031 |

- Duration transfers better than switch timing. The 3-class macro-F1 falls by about 0.2 (0.62 → 0.44) and Short/Large by about 0.23, but both stay well above chance (always predicting the majority class gives macro-F1 0.195 for 3 classes and 0.386 for 2 on LinCE).
- As in-domain, Medium (3–6 words) is the weak class (0.23). Short insertions are the easiest in tweets.

## Reading

- The models learned a real anticipatory signal, but much of it is **SwitchLingua's style**: switch at sentence boundaries and turns, and return quickly from an English insertion. Real tweets break both habits.
- The fix is in the training data, not the threshold. Candidates are monolingual stretches (or rows), so that "no switch" is learnable, and natural code-switching data in training (for example LinCE train, keeping its validation split as the test).
- For the paper, report LinCE as the out-of-domain number, next to the in-domain one. XLM-R is the better universal backbone on this evidence.

## Caveats

- Single seed. The XLM-R vs mBERT difference is only test-set noise (bootstrap), not training noise.
- LinCE's `ne`, `mixed`, `ambiguous` and similar tags, and @handles and #hashtags, are transparent `NE`: they neither break nor extend a run. Tweets are joined with spaces around punctuation (LinCE is pre-tokenized), which differs slightly from SwitchLingua's text.
- The romanized Hindi in tweets has much freer spelling than SwitchLingua's generated Hindi.
- Positions differ by tokenizer (0.38 M XLM-R vs 0.41 M mBERT pieces), so the paired comparison counts errors per tweet.
