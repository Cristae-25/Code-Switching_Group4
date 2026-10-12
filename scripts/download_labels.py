#!/usr/bin/env python
"""Download our labels from the private Hugging Face dataset.

    ./.venv/bin/python scripts/download_labels.py            # the per-token training targets
    ./.venv/bin/python scripts/download_labels.py --all      # plus the word-language files

Needs HF_TOKEN in .env (a read token whose account was given access to the
dataset). Files land in data/final/; with --all, tagged.parquet is also copied
to data/tagged.parquet, where the training scripts look for it.

    token_labels_word.parquet      per word:     lang, y_sw, y_len, y_dur, censored, dur_known
    token_labels_subword.parquet   per XLM-R piece: the same, plus pieces / piece_ids
    tagged.parquet                 word languages in the pipeline's format      (--all)
    switchlingua_tagged_rows.csv   word languages, one CSV line per conversation (--all)
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = "CodeSwitchGroup4/switchlingua-tagged-final"
TARGETS = ["token_labels_word.parquet", "token_labels_subword.parquet"]
EXTRA = ["tagged.parquet", "switchlingua_tagged_rows.csv", "README.md"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="also the word-language files")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "data", "final"))
    args = ap.parse_args()

    from dotenv import load_dotenv
    from huggingface_hub import hf_hub_download
    load_dotenv(os.path.join(ROOT, ".env"))
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN is missing: add HF_TOKEN=hf_... to .env", file=sys.stderr)
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    for name in TARGETS + (EXTRA if args.all else []):
        path = hf_hub_download(REPO, name, repo_type="dataset", token=token,
                               local_dir=args.out_dir)
        print(f"{name:32} {os.path.getsize(path) / 1e6:6.0f} MB -> {path}")
    if args.all:
        dst = os.path.join(ROOT, "data", "tagged.parquet")
        shutil.copyfile(os.path.join(args.out_dir, "tagged.parquet"), dst)
        print(f"copied tagged.parquet -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
