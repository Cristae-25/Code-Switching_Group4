#!/usr/bin/env python
"""Tag a stratified sample and write out every token with its language label.

No notebook and no kernel selection - the script pins its own interpreter:

    ./.venv/bin/python scripts/labelling/export_sample.py                 # 150/language = 2100 rows
    ./.venv/bin/python scripts/labelling/export_sample.py -n 20           # quick look
    ./.venv/bin/python scripts/labelling/export_sample.py --lang Spanish  # one language
    ./.venv/bin/python scripts/labelling/export_sample.py --no-color      # plain text

Writes three files to data/ and prints a colour-coded preview:

    sample_tokens.csv   one row PER TOKEN   (the main artifact)
    sample_rows.csv     one row per sampled row, with run lengths
    sample_tagged.txt   human-readable, token/LABEL inline
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
BLUE, ORANGE, GREY, BOLD, OFF = "\033[34m", "\033[33m", "\033[90m", "\033[1m", "\033[0m"


def preflight() -> None:
    """Fail in seconds with a clear message, not minutes into the run."""
    try:
        import torch, transformers  # noqa: F401
    except ImportError as exc:
        sys.exit(f"FATAL: {exc}\nRun with ./.venv/bin/python, not plain `python`.")
    import torch, transformers
    if int(transformers.__version__.split(".")[0]) < 5:
        sys.exit(f"FATAL: transformers {transformers.__version__} segfaults on "
                 f"torch {torch.__version__}.\nUse ./.venv/bin/python.")


def load_corpus():
    import glob
    import pandas as pd
    files = sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--Shelton1013--SwitchLingua_text/"
        "snapshots/*/*.csv")))
    if not files:
        sys.exit("No cached CSVs found. Run the notebook's download cell first.")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--per-language", type=int, default=150)
    ap.add_argument("--lang", default=None, help="restrict to one first_language")
    ap.add_argument("--preview", type=int, default=2, help="rows to print per language")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()
    preflight()

    import pandas as pd
    from src.dispatch import tag_tokens
    from src.routing import route, tagger_quality
    from src.segments import absorb_other, l2_word_runs, parse_utterances, to_segments
    from src.tokenizer import tokenize_mixed

    colour = sys.stdout.isatty() and not args.no_color
    df = load_corpus()
    if args.lang:
        df = df[df.first_language == args.lang]
        if df.empty:
            sys.exit(f"No rows for first_language={args.lang!r}")
    n = min(args.per_language, df.first_language.value_counts().min())
    sample = (df.groupby("first_language", group_keys=False)
                .sample(n=n, random_state=0).reset_index(drop=True))
    print(f"sampling {len(sample):,} rows "
          f"({n} x {sample.first_language.nunique()} languages)\n")

    token_rows, row_rows, text_blocks = [], [], []
    for rid, r in enumerate(sample.itertuples()):
        l1, l2 = r.first_language, r.second_language
        utterances = parse_utterances(r.data_generation_result)

        tokens, origin = [], []
        for ui, utt in enumerate(utterances):
            toks = tokenize_mixed(utt)
            tokens.extend(toks)
            origin.extend([ui] * len(toks))
        if not tokens:
            continue

        tagged = tag_tokens(tokens, l1, l2)
        absorbed = absorb_other(tagged)
        segments = to_segments(absorbed)
        runs = l2_word_runs(absorbed, l2)

        for ti, ((tok, raw), (_, lang)) in enumerate(zip(tagged, absorbed)):
            token_rows.append(dict(
                row_id=rid, first_language=l1, second_language=l2,
                utterance_idx=origin[ti], token_idx=ti, token=tok,
                label=lang,                       # after punctuation absorption
                raw_label=raw,                    # before absorption
                is_l2=int(lang == l2),
                tier=route(l1, l2), quality=tagger_quality(l1, l2)))

        row_rows.append(dict(
            row_id=rid, first_language=l1, second_language=l2,
            tier=route(l1, l2), quality=tagger_quality(l1, l2),
            n_tokens=len(tokens), n_switches=len(runs),
            l2_runs=" ".join(map(str, runs)),
            max_run=max(runs) if runs else 0, total_l2=sum(runs),
            segments=" ".join(f"{lg}:{ln}" for lg, ln in segments)))

        text_blocks.append(
            f"### row {rid} | {l1} -> {l2} | tier={route(l1, l2)} "
            f"| quality={tagger_quality(l1, l2)} | runs={runs}\n"
            + " ".join(f"{t}/{'L2' if lg == l2 else ('.' if lg == 'other' else ('NE' if lg == 'NE' else 'L1'))}"
                       for t, lg in absorbed))

    os.makedirs(DATA, exist_ok=True)
    tok_df, row_df = pd.DataFrame(token_rows), pd.DataFrame(row_rows)
    tok_path = os.path.join(DATA, "sample_tokens.csv")
    row_path = os.path.join(DATA, "sample_rows.csv")
    txt_path = os.path.join(DATA, "sample_tagged.txt")
    tok_df.to_csv(tok_path, index=False)
    row_df.to_csv(row_path, index=False)
    with open(txt_path, "w") as fh:
        fh.write("\n\n".join(text_blocks))

    # ---- colour-coded preview -------------------------------------------
    for lang, g in row_df.groupby("first_language"):
        for rid in g.row_id.head(args.preview):
            sub = tok_df[tok_df.row_id == rid]
            l2 = sub.second_language.iloc[0]
            head = (f"{BOLD}{lang} -> {l2}{OFF}" if colour else f"{lang} -> {l2}")
            meta = g[g.row_id == rid].iloc[0]
            print(f"{head}  [{meta.tier}/{meta.quality}]  runs={meta.l2_runs or '-'}")
            out = []
            for t in sub.itertuples():
                if not colour:
                    out.append(f"{t.token}/{'L2' if t.is_l2 else ('.' if t.label=='other' else 'L1')}")
                elif t.is_l2:
                    out.append(f"{ORANGE}{t.token}{OFF}")
                elif t.label == "other":
                    out.append(f"{GREY}{t.token}{OFF}")
                else:
                    out.append(f"{BLUE}{t.token}{OFF}")
            print("  " + " ".join(out[:70]) + ("" if len(out) <= 70 else " ..."))
            print()

    if colour:
        print(f"legend: {BLUE}first language{OFF}  {ORANGE}second language{OFF}  {GREY}other{OFF}\n")
    print(f"wrote {tok_path}  ({len(tok_df):,} tokens)")
    print(f"wrote {row_path}  ({len(row_df):,} rows)")
    print(f"wrote {txt_path}")
    print("\ntokens by label tier:")
    print(tok_df.groupby(["tier", "quality"]).size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
