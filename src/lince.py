"""LinCE as an out-of-domain test set for the anticipatory task.

LinCE (https://ritual.uh.edu/lince/) has tweets with **human** word-level language tags.
Two of its pairs are SwitchLingua pairs: ``spaeng`` (Spanish->English) and ``hineng``
(romanized Hindi->English). The tags are gold, so the switch and duration labels come
straight from them, with no tagger involved. The tweets only need to be cut into this
project's tokens and then into each model's pieces (``subword.build_frame``), so the
evaluation matches SwitchLingua's exactly.

Tag mapping (LinCE: ``lang1`` = English, ``lang2`` = the other language):

* ``lang1`` -> English (L2), ``lang2`` -> Spanish / Hindi (L1).
* ``ne``, ``mixed``, ``ambiguous``, ``fw``, ``unk`` and lettered ``other`` (@handles,
  #hashtags, URLs) -> ``NE``. That is a language-less word: it neither breaks nor
  extends a run. This is LinCE's own convention and our ``transparent`` names
  policy. ``absorb_other`` would otherwise fold a handle into its neighbour's run.
* Tokens with no letters (punctuation, emoji) -> ``other``, as in the corpus.

The ``validation`` and ``train`` splits both have public labels, and no model here was
trained on LinCE, so both are fair test data. The ``test`` labels are hidden.
"""
from __future__ import annotations

import os
import urllib.request

import numpy as np
import pandas as pd

from .anticipation import is_word
from .segments import NE
from .tokenizer import tokenize_mixed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUBLIC = os.path.join(ROOT, "data", "public")
LINCE_DIR = os.path.join(ROOT, "data", "lince")
URL = ("https://huggingface.co/datasets/lince-benchmark/lince/resolve/"
       "refs%2Fconvert%2Fparquet/lid_{pair}/{split}/0000.parquet")
PAIRS = {"spaeng": "Spanish", "hineng": "Hindi"}          # LinCE config -> first_language
SPLITS = ("validation", "train")
# row ids far above SwitchLingua's 234,172: pair block + split block + tweet index
_PAIR_BASE = {"spaeng": 10_000_000, "hineng": 20_000_000}
_SPLIT_BASE = {"validation": 0, "train": 1_000_000}


def row_id(pair: str, split: str, idx: int) -> int:
    return _PAIR_BASE[pair] + _SPLIT_BASE[split] + int(idx)


def describe_row(rid: int) -> tuple[str, str, int]:
    """``(pair, split, index)`` of a LinCE row id."""
    pair = next(p for p, b in _PAIR_BASE.items() if b <= rid < b + 10_000_000)
    rest = rid - _PAIR_BASE[pair]
    split = next(s for s, b in sorted(_SPLIT_BASE.items(), key=lambda kv: -kv[1]) if rest >= b)
    return pair, split, rest - _SPLIT_BASE[split]


def load(pair: str, split: str) -> pd.DataFrame:
    """A LinCE LID split (``idx``, ``words``, ``lid``), downloaded once into data/public/."""
    path = os.path.join(PUBLIC, f"lid_{pair}_{split}.parquet")
    if not os.path.exists(path):
        os.makedirs(PUBLIC, exist_ok=True)
        print(f"downloading LinCE lid_{pair}/{split}", flush=True)
        urllib.request.urlretrieve(URL.format(pair=pair, split=split), path)
    return pd.read_parquet(path)


def tweet_tokens(words, lid, first_language: str) -> tuple[str, list[str], list[str]]:
    """``(text, tokens, token_lang)`` for one tweet.

    The text is the LinCE words joined by spaces. Each project token takes the gold
    tag of the LinCE word its characters fall in.
    """
    words = [str(w) for w in words]
    text = " ".join(words)
    owner = np.full(len(text), -1, dtype=np.int64)          # character -> LinCE word
    pos = 0
    for k, w in enumerate(words):
        owner[pos:pos + len(w)] = k
        pos += len(w) + 1
    tokens, langs, at = [], [], 0
    for t in tokenize_mixed(text):
        s = text.find(t, at)
        if s < 0:
            raise ValueError(f"token {t!r} not found in {text!r}")
        at = s + len(t)
        tag = str(lid[owner[s]]) if owner[s] >= 0 else "other"
        tokens.append(t)
        langs.append(map_tag(tag, t, first_language))
    return text, tokens, langs


def map_tag(tag: str, token: str, first_language: str) -> str:
    if not is_word(token):
        return "other"                       # punctuation, digits, emoji: as in the corpus
    if tag == "lang1":
        return "English"
    if tag == "lang2":
        return first_language
    return NE                                # ne / mixed / ambiguous / fw / unk / @ # URLs


def to_tagged(df: pd.DataFrame, pair: str, split: str) -> tuple[pd.DataFrame, dict]:
    """A tagged-like frame plus ``texts``: the input ``subword.build_frame`` takes."""
    l1 = PAIRS[pair]
    rows, texts = [], {}
    for i, (words, lid) in enumerate(zip(df["words"], df["lid"])):
        if len(words) == 0:
            continue
        text, tokens, langs = tweet_tokens(words, lid, l1)
        rid = row_id(pair, split, int(df["idx"].iloc[i]) if "idx" in df else i)
        rows.append(dict(row_id=rid, first_language=l1, second_language="English",
                         quality="gold", tokens=tokens, token_lang=langs))
        texts[rid] = [text]
    return pd.DataFrame(rows), texts


def parquet_path(backbone: str, pair: str, split: str) -> str:
    safe = backbone.replace("/", "_")
    return os.path.join(LINCE_DIR, f"subword_{safe}_{pair}_{split}.parquet")


def build(pair: str, split: str, backbone: str, rebuild: bool = False) -> tuple[str, dict]:
    """Write (or reuse) the LinCE piece-level parquet for one pair, split and tokenizer."""
    from transformers import AutoTokenizer

    from .subword import build_frame, write_parquet
    path = parquet_path(backbone, pair, split)
    if os.path.exists(path) and not rebuild:
        return path, {}
    frame, texts = to_tagged(load(pair, split), pair, split)
    tok = AutoTokenizer.from_pretrained(backbone)
    pieces, stats = build_frame(frame, texts, tok, max_fail=0.01)
    os.makedirs(LINCE_DIR, exist_ok=True)
    write_parquet(pieces, path)
    stats["switches"] = int(sum((np.asarray(y) == 1).sum() for y in pieces.y_sw))
    stats["tweets_with_switch"] = int(sum((np.asarray(y) == 1).any() for y in pieces.y_sw))
    return path, stats
