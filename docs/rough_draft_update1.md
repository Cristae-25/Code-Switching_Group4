# Rough draft, update 1: labelling, models, and XLM-R results

Draft text for the group paper, to paste into the Overleaf LaTeX source. Prose is Markdown; tables and figures are ready-to-paste LaTeX blocks. Citation keys are placeholders (`\cite{...}`), listed with full details at the end. Every number comes from the repo's result files (`docs/labelling_pipeline.md`, `docs/labelling_models.md`, `data/anticipation_xlmr_*_12lang.json`, `data/baseline_anticipation_subword_15pairs.json`).

> **Note on file names:** the runs saved as `*_12lang.json` were trained and scored on **all 15 language pairs**. The paper should say 15.

Figures referenced below, from `docs/figures/chris/all_pairs/` (upload them to Overleaf under `figures/`):
- `switch_f1.png`: switch F1 per pair, untrained vs probe vs fine-tuned
- `duration_f1.png`: duration macro-F1 per pair, same three models
- `confusion/all_pairs.png`: pooled confusion matrices of the fine-tuned model, with the blind-guess reference
- `direction_f1.png`: switch F1 per pair by direction (L1→L2 vs L2→L1), from the fine-tuned run's saved test predictions

---

## 1. Data engineering: labelling every token's language

### 1.1 Why labelling is the foundation

SwitchLingua \cite{switchlingua} provides code-switched text but no token-level language labels. Our prediction targets are derived from such labels: whether the next token switches language, and how long the new run lasts. So we first had to tag every token of the corpus (234,172 conversations, 22.3 M word tokens, 15 language pairs) with its language. Label quality is therefore a ceiling on everything downstream: a tagging error becomes a false switch or a broken run.

We required the labelling to be **fully automatic, with no hand labelling**, and its accuracy to be **measured on human-annotated public data**.

### 1.2 Two regimes: script-disjoint and same-script pairs

The 15 pairs fall into two regimes that need different treatment.

**Script-disjoint pairs (68.7% of rows).** For Cantonese, Chinese, Japanese, Korean, Russian and Arabic paired with English, as well as Hindi rows written in Devanagari, the two languages use different writing systems. A token's Unicode script (Han, Kana, Hangul, Cyrillic, Arabic, Devanagari vs Latin) *is* its language. We tag these pairs by script, which is exact by construction and needs no model.

**Same-script pairs (31.3% of rows).** For Spanish, French, Italian, German, Vietnamese, Malay, Tagalog and romanized Hindi with English, and for German with French, both languages share the Latin alphabet. Only vocabulary and context tell them apart, and this is where all the difficulty lies.

### 1.3 What did not work for same-script pairs

We tried several taggers before settling on the final one. Early comparisons used a small development set of 324 tokens across 8 languages. Those labels were produced with model assistance, so they guided development but are not an independent benchmark (§1.4 replaces them).

- **Token-level language ID (fastText `lid.176` \cite{joulin2017fasttext}): 83%.** We restricted its 176-way output to the row's two languages. It scores each word *in isolation*, so short words shared across languages (Spanish *real*, Malay *bila*, *mas*) have nothing to disambiguate them. GlotLID \cite{kargaran2023glotlid}, a larger model with the same design, gained only about 6 points on a hard Malay/Tagalog sample, for a 1.6 GB model. We rejected both.
- **Dictionary membership: 52%,** close to chance. The rule "label a word English if it's in the English word list" fails because the word lists overlap heavily: 97% of French tokens and 81–88% of Spanish, Italian and German tokens appear in *both* lists (loanwords, shared vocabulary, names).
- **XLM-R fine-tuned for language ID on LinCE Hindi–English \cite{aguilar2020lince}: 42% on the other languages.** It reached 97.1% on the LinCE development set, so the fine-tune itself worked. But on every non-Hindi language it labelled almost every word English (Italian 4%, Malay 15%). Training taught it "the other language is romanized Hindi", not "the other language is anything but English". The training pair dominated the model's broad multilingual pretraining.
- **Character-level CANINE \cite{clark2022canine}: no gain.** We tested whether a model without a sub-word vocabulary would handle the inconsistent spelling of romanized Hindi (*nahi*/*nahin*/*nehi*) better. Under programmatic respelling of 6,890 tokens, both CANINE and mBERT stayed at about 97%, so there was no spelling problem for CANINE to fix.
- **Word-frequency comparison (dict-freq, using `wordfreq` \cite{speer2022wordfreq}).** Instead of asking whether a word is in a list, we compared how frequent it is in each language. This was much better than membership, and it served as our production tagger for a time. On human-annotated data, however, it fell short: 94.1% on LinCE Spanish–English and 80.5% on Hindi–English. Romanized Hindi has no standard spelling, so no word list can cover it.
- **mBERT \cite{devlin2019bert} classifiers fine-tuned on LinCE** were strong: 98.8% (Spanish–English) and 97.4% (Hindi–English). But each is tied to its training pair, transfer to other pairs was uneven (French 68% on the development set), and it was beaten by the tagger we adopted.

### 1.4 Final choice: AnE-LID for same-script pairs, Unicode for the rest

We replaced the self-labelled development set with evaluation on **human-annotated public data**: the LinCE development splits for Spanish–English and Hindi–English \cite{aguilar2020lince}, and the Denglisch German–English corpus \cite{sterner2023tongueswitcher}. On these sets, **AnE-LID** \cite{sterner_ane} was the most accurate tagger. It is an XLM-R-large token classifier trained on code-switched text to decide, for each word in context, whether it is English.

```latex
\begin{table}[t]
\centering
\small
\begin{tabular}{lrrrr}
\toprule
Human-annotated set & Words & dict-freq & mBERT & \textbf{AnE-LID} \\
\midrule
LinCE Spanish--English (dev) & 31,667 & 94.1\% & 98.8\% & \textbf{99.4\%} \\
LinCE Hindi--English (dev)   & 12,303 & 80.5\% & 97.4\% & \textbf{98.8\%} \\
Denglisch (German--English)  & 59,485 & 96.2\% & --     & 99.9\%$^{*}$ \\
\bottomrule
\end{tabular}
\caption{Word-level language-ID accuracy on human-annotated public data.
$^{*}$AnE-LID saw Denglisch in training, so this is an upper bound.}
\label{tab:tagger-accuracy}
\end{table}
```

AnE-LID only decides "English or not", so it can't separate German from French. For that one pair (German→French, 8,511 rows), we kept the word-frequency tagger, which compares the two named languages directly. No public German–French benchmark exists, so its accuracy there is untested; its closest measured case is 96.2% on Denglisch.

To check the labels on our own corpus, where no human labels exist, we ran the word-frequency tagger alongside AnE-LID on every AnE-tagged row. The two use entirely different evidence (a neural model vs word frequencies). They disagree on only 2–4% of tokens, and on the public sets, tokens where they agree were correct 99.3–100% of the time.

```latex
\begin{table}[t]
\centering
\small
\begin{tabular}{lll}
\toprule
Pairs & Tagger & Accuracy \\
\midrule
Cantonese, Chinese, Japanese, Korean, Russian, Arabic; & Unicode script & exact \\
\quad Devanagari-script Hindi rows & & \\
Spanish, French, Italian, German, Vietnamese, Malay, & AnE-LID & 98.8--99.4\% (LinCE) \\
\quad Tagalog, romanized Hindi (all with English) & & \\
German--French & word frequency & untested (96.2\% on Denglisch) \\
\bottomrule
\end{tabular}
\caption{Final routing: each pair goes to the most accurate tagger for it.}
\label{tab:routing}
\end{table}
```

### 1.5 Proper nouns: when does a name count as a switch?

Names do not belong cleanly to either language. Is *Messi* in a Spanish sentence a switch into English? A language tagger has to give it some label, and the choice matters. If every Latin-script name is treated as English, "Ayer Leo Messi marcó dos goles" gains two spurious one-word switches. Our targets are switches and run lengths, so such errors would inflate the switch count and shorten runs.

Names are first flagged by AnE-NER \cite{sterner_ane}, which marks name tokens without changing their language label. AnE-NER often splits a name into pieces ("Game [of] Thrones", "Severance [Season 2]"). A repair step joins a name back together across a single function word, and across adjacent capitalized words that aren't ordinary first-language words.

We then apply one rule: **a name counts as a switch only when there is visible evidence for it.**

- **In non-Latin rows, the script is the evidence.** A Latin-script name inside Arabic or Chinese text is English, because the writer chose to write it in Latin script. A name written in the row's own script, such as ترامب or 梅西, stays in the first language.
- **In Latin rows, the name's own words can be the evidence.** If its words are clearly more frequent in one of the two languages, by at least 1.0 on the Zipf frequency scale \cite{speer2022wordfreq}, that language wins. "Severance Season" is English, and "Union européenne" is French.
- **Otherwise the name takes the language of its context and cannot open a run of its own.** Most person and club names are opaque this way: *Messi*, *PSG* and *EU* have similar frequencies in both languages. Inside a single-language stretch, the name takes that language. At a real switch, it joins the side that isn't separated from it by punctuation. If that is still undecided, it keeps the tagger's label when it is one of the two sides, and otherwise takes the preceding language.

```latex
\begin{table}[t]
\centering
\small
\begin{tabular}{llll}
\toprule
Row & Name in context & Label & Evidence \\
\midrule
Spanish--English & Ayer \textit{Leo Messi} marc\'o \ldots & Spanish & context (opaque name) \\
Spanish--English & \ldots scored by \textit{Leo Messi} \ldots & English & context (opaque name) \\
Spanish--English & \ldots \textit{Severance Season 2} \ldots & English & the name's words \\
French--English  & \ldots l'\textit{Union europ\'eenne} \ldots & French & the name's words \\
Arabic--English  & \ldots \textit{Messi} \ldots & English & Latin script \\
Arabic--English  & \ldots ترامب \ldots & Arabic & Arabic script \\
Chinese--English & \ldots 梅西 \ldots & Chinese & Han script \\
\bottomrule
\end{tabular}
\caption{How names are labelled under our policy. Every case is a unit test.}
\label{tab:names}
\end{table}
```

*(LaTeX note: the Arabic and Chinese cells need XeLaTeX or LuaLaTeX with suitable fonts, e.g. `polyglossia`/`xeCJK` in Overleaf. Otherwise replace them with transliterations.)*

**Effect on the corpus.** On a sample of 30,000 rows (2,000 per pair), we compared this policy with two simpler ones.
- **Against skipping names entirely** (LinCE's convention \cite{aguilar2020lince}), our policy adds **5.9%** more switches. Those are the names with visible evidence.
- **Labelling every Latin-script name English** would add **13.6%**, and 20–28% in French, German, Spanish and Italian, where names are written in the same script as the first language.

Our policy adds only 2–6% in those four pairs, so most of that difference is spurious one-word switches that it avoids.

### 1.6 From word labels to prediction targets

The models read XLM-R sub-word pieces, so we re-tokenized the original text with the XLM-R tokenizer and gave each piece the language of the word its characters fall in. Only 155 of 21.7 M pieces straddle two languages. Punctuation, digits and bare spacing pieces get no language: they neither break nor extend a run. From this piece-level language sequence we derive, at each position $t$:

- $y^{\text{sw}}_t = 1$ if piece $t{+}1$ is in a different language from piece $t$;
- at each switch, the length of the new run, binned into **Small (1–2 pieces), Medium (3–6), Large (7+)**.

A run still open at the end of a text only counts once it reaches 7 pieces, because only then is its class certain. Durations are counted in pieces, so they are comparable within a script but not across scripts: Chinese, Japanese and Korean pieces are often single characters.

---

## 2. Models: from blind guessing to fine-tuned XLM-R

### 2.1 Task and evaluation

At each piece $t$, a model sees only pieces $1..t$ and predicts (a) whether piece $t{+}1$ switches language and (b) if a switch happens, which duration class the new run will fall in.

- **Splits.** Conversations are split by ID: 80% train, 10% validation, 10% test. The test split has 1.83 M scored positions, 189,756 of them switches.
- **Decision threshold.** Each model's switch threshold, and three duration offsets, are tuned on validation. Test is scored once.
- **Metrics.** Switches are rare (10.4% of positions), so we report switch F1 rather than accuracy: always predicting "no switch" would score about 90% accuracy and F1 0. We report F1 per pair and its mean over the 15 pairs, plus duration macro-F1 over the three classes at the true switch points.

### 2.2 The model ladder

We compare models that each add one source of knowledge:

1. **Blind guessing.** Guess each class at its own frequency, ignoring the input. For switches, guessing "switch" at the switch rate $p$ gives precision = recall = F1 = $p$ (0.104 pooled; 0.11 averaged over pairs). For duration, guessing at the class frequencies gives macro-F1 = 1/3 whatever the class mix.
2. **Rule baselines** (no neural model; the count-based ones are fitted on train):
   - *always switch*;
   - *naive current language*: predict a switch whenever the current piece is English;
   - *run-length hazard*: P(switch | current language, length of the current run, pair);
   - *piece lookup*: P(switch | current piece), one probability per XLM-R piece. This is the best that can be done from the current piece alone.
3. **Untrained XLM-R.** Pretrained `xlm-roberta-base` \cite{conneau2020xlmr} as downloaded, with randomly initialized switch and duration output layers and no training. Only the threshold and duration offsets are fitted on validation. This is the starting point of fine-tuning, and we expect it to score at chance.
4. **Linear probe.** XLM-R stays frozen, and only the two linear output layers (about 3,000 parameters, against 278 M in the frozen model) are trained, for 12,000 steps. The probe measures how much switch and duration information XLM-R's pretrained representations already hold.
5. **Fine-tuned XLM-R.** All 278 M parameters are trained for 17,500 steps at batch 16 (about 1.5 passes over the training conversations), with learning rates of $2{\times}10^{-5}$ for the body and $10^{-3}$ for the output layers, and class-balanced weights on the duration loss.

**Causality.** XLM-R was pretrained bidirectionally. We load it in decoder mode (`is_decoder=True`), which applies a lower-triangular attention mask in every layer, so position $t$ attends only to pieces $1..t$. Before every run we verify this directly. We replace all pieces after each position $t$ with random ones and check that no output at or before $t$ changes. The maximum change was 0.0 in all three runs. No model receives a language label as input: all read the text pieces only.

**Limitation.** The probe trained for 12,000 steps and the fine-tuned model for 17,500. The gap between them therefore combines the effect of adapting XLM-R itself with the effect of 5,500 extra training steps.

---

## 3. Results: XLM-R's progress

### 3.1 Summary

```latex
\begin{table}[t]
\centering
\small
\begin{tabular}{lccc}
\toprule
Model & Switch F1 & Switch F1 & Duration \\
      & (mean over pairs) & (pooled) & macro-F1 \\
\midrule
Blind guess (class frequencies)      & 0.110 & 0.104 & 0.333 \\
Always switch                         & 0.196 & 0.188 & -- \\
Naive (current language)              & 0.197 & 0.222 & -- \\
Run-length hazard                     & 0.259 & 0.266 & 0.225 \\
Piece lookup                          & 0.431 & 0.428 & 0.455 \\
\midrule
Untrained XLM-R                       & 0.165 & 0.160 & 0.324 \\
Linear probe (frozen XLM-R)           & 0.412 & 0.406 & 0.508 \\
\textbf{Fine-tuned XLM-R}             & \textbf{0.686} & \textbf{0.655} & \textbf{0.611} \\
\bottomrule
\end{tabular}
\caption{Test-set results on all 15 pairs. Duration macro-F1 is pooled over
pairs at the true switch points (val-tuned). The run-length hazard's duration
predictions use the most common class per pair and direction.}
\label{tab:ladder}
\end{table}
```

Each step of the ladder adds knowledge:

- **Untrained XLM-R is at chance, as expected.** A random output layer has no way to read the pretrained features. Its switch F1 (0.165) is below even "always switch" (0.196), and its duration macro-F1 (0.324) matches blind guessing (0.333).
- **The linear probe triples switch F1 (0.165 → 0.412).** Pretrained XLM-R already holds substantial switch information that a single linear layer can read. The probe roughly matches the piece lookup (0.431), the best possible from the current piece alone.
- **Fine-tuning adds +0.27 more (0.412 → 0.686)** and beats the piece lookup by +0.26. Everything above the piece lookup comes from reading the context before the current piece. Duration macro-F1 rises from 0.508 to 0.611, against 0.333 for blind guessing.

### 3.2 Per-pair switch and duration F1 (Figures `switch_f1`, `duration_f1`)

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/switch_f1.png}
\caption{Switch F1 per language pair on the test split, for untrained
XLM-R, the linear probe and fine-tuned XLM-R. Pairs are sorted by the
fine-tuned score.}
\label{fig:switch-f1}
\end{figure}
```

**Switch (Figure~\ref{fig:switch-f1}).**
- **Every pair improves, in the same order:** untrained (0.12–0.33) < probe (0.32–0.48) < fine-tuned (0.55–0.82). The gain from fine-tuning holds across all 15 pairs, not just on average.
- **Best pairs:** Korean (0.822), Japanese (0.750) and German→English (0.746).
- **Worst pairs:** Malay (0.545), Cantonese (0.626) and Vietnamese (0.631). The spread between best and worst is 0.277.
- **Hindi and Tagalog have high untrained scores** (0.33 and 0.28). That reflects their high switch rates (20.8% and 17.8% of positions), not knowledge: blind guessing scores the switch rate. Each pair should be compared with its own chance level.
- **German→French has the weakest probe** (0.320), the only pair whose second language isn't English. That's consistent with pretrained features that carry a general "English or not" signal (to confirm with an error analysis).

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/duration_f1.png}
\caption{Duration macro-F1 (Small 1--2 / Medium 3--6 / Large 7+ pieces) per
language pair, at the true switch points, with val-tuned decisions. Blind
guessing at the class frequencies scores 0.333.}
\label{fig:duration-f1}
\end{figure}
```

**Duration (Figure~\ref{fig:duration-f1}).**
- **Pairs differ less on duration than on switching.** Fine-tuned macro-F1 ranges from 0.518 to 0.650, a spread of 0.132, against 0.277 for switch F1.
- **Best pairs:** Spanish (0.650), French (0.641) and Italian (0.637).
- **Worst pairs:** Tagalog (0.518), Malay (0.552) and Russian (0.585), with German→French a near-tie at 0.585.
- **Malay is near the bottom on both tasks,** making it the hardest pair overall.

### 3.3 Where the errors are (Figure `confusion/all_pairs`)

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/all_pairs.png}
\caption{Confusion matrices of fine-tuned XLM-R, all 15 pairs pooled. Left:
switch (all test positions). Right: duration class at true switch points.
Cells show counts and the share of the actual row (recall). The line under
each matrix gives the blind-guess reference: guessing each class at its own
frequency, every row would repeat those frequencies.}
\label{fig:confusion}
\end{figure}
```

- **Switch:**
  - The model finds 65% of switches (recall 0.651), against 10.4% for blind guessing.
  - Its switch calls are right 66% of the time (precision 0.658).
  - Only 4% of non-switch positions are falsely flagged.
- **Duration:**
  - Small and Large runs are recognized well (recall 61% and 79%, against blind 27% and 51%).
  - **Medium (3–6 pieces) is the weak class** (44% recall, F1 0.415). Its errors split evenly between Small (28%) and Large (28%), while Small and Large are rarely confused with each other (6–13%). The model mostly confuses neighbouring classes: 93.4% of its duration predictions are within one class of the truth.
- **End to end,** the model predicts the switch *and* the right duration class at 45.2% of true switch points.

### 3.4 Direction: switching back is easier than switching in (Figure `direction_f1`)

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/direction_f1.png}
\caption{Switch F1 of fine-tuned XLM-R per pair, split by direction: L1$\to$L2
(the current piece is in the first language and the next is in the second) and
L2$\to$L1 (the reverse). The tick on each bar is blind guessing at that
direction's switch rate, whose F1 equals the rate.}
\label{fig:direction}
\end{figure}
```

Switching *back* into the first language is easier to predict than switching *into* the second.

- **L2→L1 is easier in 13 of 15 pairs.** The mean F1 over pairs is **0.754 for L2→L1 vs 0.626 for L1→L2.**
- **The two exceptions are Korean** (0.831 vs 0.812) **and German→French** (0.684 vs 0.652). Both directions are already high there.
- **Cantonese has by far the largest gap** (0.443 vs 0.786, +0.343). Its recall going into English is only 0.37. Vietnamese (+0.215) and Arabic (+0.187) follow.
- **The pooled gap is larger than the per-pair mean** (0.540 vs 0.771 pooled). Cantonese is 58% of the test positions and 55% of its switches, so its weak L1→L2 direction pulls the pooled number down. This is a reason to report per-pair means alongside pooled scores.
- **Both directions are far above blind guessing** in every pair. Pooled, the blind-guess F1s are 0.084 for L1→L2 and 0.147 for L2→L1, their switch rates.

One reading, still to be tested with an error analysis: the return to L1 often comes where an embedded L2 phrase is complete, which the preceding context signals. The start of an L2 insertion is harder to anticipate.

---

## References (placeholders: verify before adding to the .bib)

These entries have **not** yet been checked against a citation database; the scholarly-access connector requires sign-in in this session. Confirm authors, venues and years before adding them to the .bib.

- `switchlingua`: the SwitchLingua code-switching dataset paper (Hugging Face dataset `Shelton1013/SwitchLingua_text`). *Need the exact citation: authors, venue and year.*
- `conneau2020xlmr`: Conneau et al., "Unsupervised Cross-lingual Representation Learning at Scale", ACL 2020.
- `devlin2019bert`: Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", NAACL 2019.
- `aguilar2020lince`: Aguilar et al., "LinCE: A Centralized Benchmark for Linguistic Code-switching Evaluation", LREC 2020.
- `joulin2017fasttext`: Joulin et al., "Bag of Tricks for Efficient Text Classification", EACL 2017 (the fastText language-ID model `lid.176`).
- `kargaran2023glotlid`: Kargaran et al., "GlotLID: Language Identification for Low-Resource Languages", Findings of EMNLP 2023.
- `clark2022canine`: Clark et al., "CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation", TACL 2022.
- `speer2022wordfreq`: Speer, `wordfreq` (software).
- `sterner2023tongueswitcher`: Sterner and Teufel, "TongueSwitcher: Fine-Grained Identification of German–English Code-Switching", CALCS workshop 2023 (the Denglisch corpus).
- `sterner_ane`: the paper behind AnE-LID / AnE-NER (`igorsterner/AnE-LID`, `igorsterner/AnE-NER`). *Need the exact citation; check the model card.*
