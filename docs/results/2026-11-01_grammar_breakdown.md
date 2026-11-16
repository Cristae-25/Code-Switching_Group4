# 2026-11-01: Do mid-clause misses follow grammatical patterns? (local, no GPU)

**Why:** mid-clause switches (nothing between word t and t+1) cause 57% of XLM-R's missed switches ([punctuation breakdown](2026-11-01_punct_breakdown.md)). Before building a grammar input, this checks whether those misses cluster on prefix-only word properties the model is not using.

**How:** `./.venv/bin/python scripts/analysis/grammar_breakdown.py xlmr_6pairs` (8 s). It covers the last piece of each word, mid-clause positions only (1.04 M; 52 k switches; 27,430 misses, the same misses as the punctuation breakdown), on the `xlmr_6pairs` test split at τ = 0.377. The switch loss is unweighted BCE, so the model's **mean predicted probability** can be compared with each group's **base rate**. A big gap would mean an unused cue. Word frequency comes from wordfreq tables, read directly with Chinese simplified first, because Cantonese is written in traditional characters. Output: `data/results/grammar_breakdown_xlmr_6pairs.json`, with per-pair tables.

## Results (base rate vs. the model's mean probability, then F1 and share of misses)

| Feature | Group | Base rate | Mean prob | F1 | Share of misses |
|---|---|---:|---:|---:|---:|
| direction | from L1 (into L2) | 0.039 | 0.057 | **0.274** | **0.873** |
| | from L2 (into L1) | 0.083 | 0.106 | 0.692 | 0.127 |
| run so far | 1 word | 0.130 | 0.131 | 0.752 | 0.143 |
| | 7+ words | 0.034 | 0.062 | 0.229 | 0.474 |
| frequency | top (Zipf ≥ 6) | 0.042 | 0.053 | 0.322 | 0.341 |
| | rare (< 3) | 0.126 | 0.136 | 0.682 | 0.044 |
| clause position | 1st word | 0.033 | 0.034 | 0.519 | 0.121 |
| | 8+ words | 0.049 | 0.092 | 0.377 | 0.163 |

## Reading

- **No unused cue in these features.** In every group the mean probability is within about 0.04 of the base rate. Where it differs, it is *above*: in long runs (0.062 vs 0.034) and late in a clause (0.092 vs 0.049), the model is unsure and spreads false alarms. It is never systematically below, which is what a missed cue would look like. XLM-R already encodes run length, word frequency and clause position.
- **The bottleneck is the start of an English insertion inside an L1 clause.** Switches out of L1 mid-clause have F1 0.27 and cause 87% of the mid-clause misses. Switching *back* is easy (0.69): after a single inserted word, the model knows the insertion ends (F1 0.75 at run length 1). When an insertion *starts* is largely a lexical choice, with a base rate of 3.9%. Coarse grammar features cannot predict it.
- **Implication:** a hand-built grammar input (function-word or position flags) is unlikely to help. The features worth trying are ones the text does not contain: speaker history (how often this speaker or conversation has switched so far) or a bigger model. That points to `xlmrL_lora16` before grammar features.
- **Caveats:** these are the 6 script-disjoint pairs, and the micro numbers are about 80% Cantonese. Function words matter most in the Latin pairs. Calibration within one group also does not rule out interactions between features. Re-run on the overnight 15-pair runs (`xlmr_prev`, `xlmr_pair`, `xlmr_lora16`) before ruling grammar out.
