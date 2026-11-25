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


def mix_path(weight: int, data_dir: str = os.path.join(ROOT, "data")) -> str:
    return os.path.join(data_dir, f"subword_mix_lince{weight}.parquet")


def mix(sl_path: str, backbone: str, weight: int, out: str | None = None, seed: int = 0,
        pairs=tuple(PAIRS), lince_paths: dict | None = None) -> tuple[str, dict]:
    """SwitchLingua pieces + LinCE **train** tweets, for multi-dataset training.

    LinCE *validation* is never included: it stays the held-out LinCE test. The
    streaming split (``row_id % 10``) also divides LinCE train 80/10/10, so only
    its training rows (``row_id % 10 < 8``) are repeated ``weight`` times. Its val
    and test rows appear once, and duplicates never cross a split. The rows are
    shuffled (``seed``) so streaming is not a curriculum by dataset.

    ``lince_paths`` maps pair -> an already-built LinCE train parquet (tests).
    Returns ``(path, stats)``; ``stats`` gives rows and pieces per source and the
    LinCE share of training-split pieces.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    if weight < 1:
        raise ValueError("weight must be >= 1 (0 means: do not mix)")
    out = out or mix_path(weight)
    sl = pq.read_table(sl_path)
    parts, stats = [sl], {"weight": int(weight)}

    def n_pieces(t):
        return int(pc.sum(pc.list_value_length(t["piece_ids"])).as_py() or 0)

    def train_rows(t):                              # streaming split: row_id % 10 in 0..7
        return pa.array(t["row_id"].to_numpy() % 10 < 8)

    sl_train = sl.filter(train_rows(sl))
    stats["switchlingua"] = dict(rows=sl.num_rows, train_pieces=n_pieces(sl_train))
    lince_train_pieces = 0
    for pair in pairs:
        path = (lince_paths or {}).get(pair) or build(pair, "train", backbone)[0]
        t = pq.read_table(path).cast(sl.schema)
        if (pc.min(t["row_id"]).as_py() - _PAIR_BASE[pair]) < _SPLIT_BASE["train"]:
            raise ValueError(f"{path} holds LinCE validation rows; only train may be mixed")
        tr = t.filter(train_rows(t))
        parts += [t] + [tr] * (weight - 1)          # val/test rows once, train rows x weight
        lince_train_pieces += weight * n_pieces(tr)
        stats[pair] = dict(rows=t.num_rows, train_rows=tr.num_rows, rows_written=t.num_rows + (weight - 1) * tr.num_rows)
    table = pa.concat_tables(parts)
    perm = np.random.default_rng(seed).permutation(table.num_rows)
    table = table.take(pa.array(perm))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    pq.write_table(table, out, compression="snappy", row_group_size=8192)
    stats["rows"] = table.num_rows
    stats["lince_share_of_train_pieces"] = lince_train_pieces / (lince_train_pieces + stats["switchlingua"]["train_pieces"])
    return out, stats
