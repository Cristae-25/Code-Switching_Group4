#!/usr/bin/env python
"""Score the Latin-script taggers on public, human-annotated LID sets.

``data/gold_eval.json`` is saturated (every tagger is at or near 100%) and was
labelled by a model, so it can no longer rank taggers. These sets were labelled
by people and are thousands of tokens each:

    lince_spaeng   LinCE Spanish-English LID, validation split (tweets)
    lince_hineng   LinCE romanized Hindi-English LID, validation split (tweets)
    denglisch      Denglisch German-English, manually annotated part (Reddit)

Only tokens whose gold label is one of the two languages are scored; ``ne``,
``other``, ``mixed``, ``ambiguous`` and Denglisch's category 3/4 words are
skipped, as ``NE`` is on the gold set.

Contamination - read before comparing columns:
  * dict-freq was never trained on anything, so it is independent on all three.
  * the mBERT LinCE models were trained on the LinCE *train* splits; the
    validation splits are held out, but in-domain for them.
  * AnE-LID was trained on LinCE train + Denglisch. Denglisch is therefore
    probably *seen data* for AnE, and its score there is an upper bound.

The second table is what makes corpus-wide consensus (``consensus_corpus.py``)
meaningful: it measures how accurate tokens are when AnE and dict-freq agree,
versus when they disagree. If agreement predicts correctness here, the corpus
agreement rate is a usable accuracy proxy with no hand labelling.

Data is fetched on first run into ``data/public/``.
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
PUBLIC = os.path.join(ROOT, "data", "public")

_LINCE = ("https://huggingface.co/datasets/lince-benchmark/lince/resolve/"
          "refs%2Fconvert%2Fparquet/{split}/validation/0000.parquet")
_DENGLISCH = ("https://raw.githubusercontent.com/HaifaCLG/Denglisch/main/"
              "corpus/Manu_corpus.csv.zip")


def _fetch(url: str, dst: str) -> str:
    if not os.path.exists(dst):
        os.makedirs(PUBLIC, exist_ok=True)
        print(f"downloading {url}", flush=True)
        urllib.request.urlretrieve(url, dst)
    return dst


def load_lince(split: str) -> list[tuple[list[str], list[str]]]:
    """LinCE: lang1 = English, lang2 = the other language."""
    import pandas as pd
    path = _fetch(_LINCE.format(split=split), os.path.join(PUBLIC, f"{split}_validation.parquet"))
    df = pd.read_parquet(path)
    lab = {"lang1": "EN", "lang2": "L1"}
    return [(list(w), [lab.get(x, "-") for x in l]) for w, l in zip(df.words, df.lid)]


def load_denglisch() -> list[tuple[list[str], list[str]]]:
    """Denglisch: category 1 = English, 2 = German; ``<...>`` rows are markup."""
    import pandas as pd
    csv = os.path.join(PUBLIC, "Manu_corpus.csv")
    if not os.path.exists(csv):
        z = _fetch(_DENGLISCH, os.path.join(PUBLIC, "Manu_corpus.csv.zip"))
        with zipfile.ZipFile(z) as f:
            f.extract("Manu_corpus.csv", PUBLIC)
    df = pd.read_csv(csv, keep_default_na=False)
    df = df[~df.categ.str.startswith("<")]
    lab = {"1": "EN", "2": "L1"}
    return [(g.token.tolist(), [lab.get(c, "-") for c in g.categ])
            for _, g in df.groupby(["sen_id", "sen_num"], sort=False)]


SETS = {
    "lince_spaeng": ("Spanish", lambda: load_lince("lid_spaeng"), "spaeng"),
    "lince_hineng": ("Hindi", lambda: load_lince("lid_hineng"), "hineng"),
    "denglisch": ("German", load_denglisch, "spaeng"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="sentences per set (0 = all)")
    args = ap.parse_args()

    from src import ane_tagger
    from src.bert_tagger import _label_words
    from src.dict_tagger import tag_frequency
    from src.metrics import wilson
    from src.routing import MODEL_L1_LABEL

    from src.lexical_override import override_tagged

    names = ["dict-freq", "mBERT", "AnE", "AnE+dict", "AnE+override"]
    n_flipped = {}
    rows_acc, rows_agree = [], []

    for set_name in args.sets.split(","):
        lang, loader, mkey = SETS[set_name]
        data = loader()
        if args.limit:
            data = data[:args.limit]
        l1 = MODEL_L1_LABEL[mkey]
        sc = {k: 0 for k in names}
        n = 0
        agree_n = agree_ok = split_n = split_ok_ane = split_ok_dict = 0

        for b in range(0, len(data), args.batch):
            batch = data[b:b + args.batch]
            ane = ane_tagger.word_probs_batch([w for w, _ in batch])
            for (words, gold), ane_row in zip(batch, ane):
                df = [("EN" if l == "English" else ("?" if l == "other" else "L1"))
                      for _, l in tag_frequency(words, lang, "English")]
                mb = [("EN" if t == "en" else ("L1" if t == l1 else "?"))
                      for t in _label_words(mkey, words)]
                an = [("EN" if t == "English" else ("L1" if t in ("notEnglish", "Mixed") else "?"))
                      for t, _ in ane_row]
                hy = [a if a != "?" and c >= args.threshold else d
                      for a, (_, c), d in zip(an, ane_row, df)]
                # production: AnE, then the lexical override (src/lexical_override.py)
                code = {"EN": "English", "L1": lang}
                ov, changed = override_tagged([(w, code.get(a, "other")) for w, a in zip(words, an)],
                                              lang, "English")
                ov = [a if not c else ("EN" if l == "English" else "L1")
                      for a, (_, l), c in zip(an, ov, changed)]
                n_flipped[set_name] = n_flipped.get(set_name, 0) + sum(
                    c for c, g in zip(changed, gold) if g != "-")
                preds = {"dict-freq": df, "mBERT": mb, "AnE": an, "AnE+dict": hy, "AnE+override": ov}
                for i, g in enumerate(gold):
                    if g == "-":
                        continue
                    n += 1
                    for k in names:
                        sc[k] += preds[k][i] == g
                    if an[i] == df[i]:
                        agree_n += 1
                        agree_ok += an[i] == g
                    else:
                        split_n += 1
                        split_ok_ane += an[i] == g
                        split_ok_dict += df[i] == g
            print(f"  {set_name}: {min(b + args.batch, len(data))}/{len(data)} sentences",
                  end="\r", flush=True)
        print(" " * 60, end="\r")
        rows_acc.append((set_name, n, {k: sc[k] / n for k in names}))
        lo, hi = wilson(agree_ok, agree_n)
        rows_agree.append((set_name, agree_n / n, agree_ok / max(agree_n, 1), lo, hi,
                           split_ok_ane / max(split_n, 1), split_ok_dict / max(split_n, 1)))

    print("=== 1. token accuracy on human-annotated sets (L1 / EN tokens only) ===")
    print(f"{'set':14}{'n':>7}" + "".join(f"{k:>14}" for k in names))
    for s, n, acc in rows_acc:
        print(f"{s:14}{n:>7}" + "".join(f"{acc[k]:>13.1%} " for k in names))
    print("  (denglisch is probably in AnE's training data - treat its AnE score as an upper bound)")
    print("  AnE+override: tokens flipped per set "
          + ", ".join(f"{s} {n_flipped.get(s, 0)}" for s, _, _ in rows_acc)
          + " (lince_hineng is out of scope: always identical to AnE)")

    print("\n=== 2. does AnE / dict-freq agreement predict correctness? ===")
    print(f"{'set':14}{'agree':>8}{'acc|agree':>11}{'95% CI':>17}"
          f"{'AnE|split':>11}{'dict|split':>11}")
    for s, rate, acc, lo, hi, a_s, d_s in rows_agree:
        print(f"{s:14}{rate:>7.1%}{acc:>10.1%}   [{lo:.1%}, {hi:.1%}]{a_s:>10.1%}{d_s:>11.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
