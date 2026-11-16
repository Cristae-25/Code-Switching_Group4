#!/usr/bin/env python
"""Upload the final labels in data/final/ to the private Hugging Face dataset.

    ./.venv/bin/python scripts/labelling/upload_labels.py --message "v2: lexical override"
    ./.venv/bin/python scripts/labelling/upload_labels.py --dry-run

One commit with the five files below. Uses a write token from .env (``--token-name``,
default ``HF_FULL``; it needs write access to the org repo) (``HF_TOKEN`` is read-only). The repo stays private: the labels derive
from a gated dataset. Earlier versions stay in the repo history; the dataset
card names the commit of each.
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = "CodeSwitchGroup4/switchlingua-tagged-final"
FILES = ["README.md", "tagged.parquet", "switchlingua_tagged_rows.csv",
         "token_labels_word.parquet", "token_labels_subword.parquet"]


def token(name: str = "HF_FULL") -> str:
    for line in open(os.path.join(ROOT, ".env")):
        if line.startswith(name + "="):
            return line.split("=", 1)[1].strip()
    raise SystemExit(f"{name} not found in .env")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ROOT, "data", "final"))
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--message", required="--dry-run" not in sys.argv)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--token-name", default="HF_FULL",
                    help=".env key of a token with write access to the org repo")
    a = ap.parse_args()
    from huggingface_hub import CommitOperationAdd, HfApi

    paths = [os.path.join(a.src, f) for f in FILES]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"missing: {missing}")
    api = HfApi(token=token(a.token_name))
    info = api.dataset_info(a.repo)
    if not info.private:
        raise SystemExit(f"{a.repo} is public - refusing to upload labels derived from a gated dataset")
    print(f"{a.repo} (private) at {info.sha[:10]}")
    for f, p in zip(FILES, paths):
        print(f"  {f:32s} {os.path.getsize(p) / 1e6:8.1f} MB")
    if a.dry_run:
        return 0
    ops = [CommitOperationAdd(path_in_repo=f, path_or_fileobj=p) for f, p in zip(FILES, paths)]
    c = api.create_commit(a.repo, ops, commit_message=a.message, repo_type="dataset")
    print(f"committed {c.oid[:10]}: {c.commit_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
