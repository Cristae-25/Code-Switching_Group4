"""Sub-word pieces for the anticipation task.

Each XLM-R piece takes the language of the word it sits in. The word labels
are the final per-token labels (``token_lang`` after the names policy), and
the original text comes from the SwitchLingua CSVs, joined on ``row_id``. The
stored tokens were produced by ``tokenize_mixed`` from that text, so an
in-order search recovers each token's character span exactly.

A piece's language is the majority slot of the word characters it covers
(ties go to L1; a piece covering two languages is counted, not an error). A
piece covering no word characters - punctuation, digits, spaces, or the bare
SentencePiece "▁" that XLM-R emits with the next character's offsets - is
NONE and transparent to the labels. Merged CJK pieces such as 東京 need no
special case: their characters share a language.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import regex

from .anticipation import L1, L2, NONE, anticipatory_labels, token_slots
from .tokenizer import tokenize_mixed

XLMR_VOCAB = 250_002
# Space markers: SentencePiece (XLM-R) "▁", byte-level BPE (RoBERTa) "Ġ" and "Ċ".
# WordPiece (mBERT) has none; its "##" continuations are ordinary pieces.
MARKERS = ("▁", "Ġ", "Ċ")
_CJK = regex.compile(r"[\p{Han}\p{Hiragana}\p{Katakana}\p{Hangul}]")


def is_blank(piece: str) -> bool:
    """A piece that is only a space marker carries no characters of its own."""
    for m in MARKERS:
        piece = piece.replace(m, "")
    return piece == ""
LABEL_KEYS = ("y_sw", "y_len", "y_dur", "censored", "dur_known")


def align_words(utterances: list[str], tokens: list[str]):
    """``(utterance, start, end)`` for every stored token, in order, or None."""
    spans, k = [], 0
    for u, text in enumerate(utterances):
        pos = 0
        for t in tokenize_mixed(text):
            if k >= len(tokens) or tokens[k] != t:
                return None
            j = text.find(t, pos)
            if j < 0:
                return None
            spans.append((u, j, j + len(t)))
            pos = j + len(t)
            k += 1
    return spans if k == len(tokens) else None


def piece_languages(offsets, blank, char_slots: np.ndarray):
    """Slot per piece from the word slots of the characters it covers."""
    langs, mixed = [], 0
    for (s, e), is_blank in zip(offsets, blank):
        if is_blank:
            langs.append(NONE)
            continue
        seg = char_slots[s:e]
        n1, n2 = int((seg == L1).sum()), int((seg == L2).sum())
        if n1 == 0 and n2 == 0:
            langs.append(NONE)
            continue
        mixed += int(n1 > 0 and n2 > 0)
        langs.append(L2 if n2 > n1 else L1)
    return langs, mixed


def subword_row(utterances, tokens, token_lang, first_language: str,
                second_language: str, tokenizer, is_entity=None):
    """Piece ids and per-piece language slots for one row, or None if unaligned."""
    tokens, token_lang = list(tokens), list(token_lang)
    spans = align_words(utterances, tokens)
    if spans is None:
        return None
    slots = token_slots(tokens, token_lang, first_language, second_language,
                        None if is_entity is None else list(is_entity))
    char_slots = [np.full(len(u), -1, dtype=np.int8) for u in utterances]
    char_tok = [np.full(len(u), -1, dtype=np.int64) for u in utterances]
    for k, ((u, s, e), slot) in enumerate(zip(spans, slots)):
        char_tok[u][s:e] = k
        if slot != NONE:
            char_slots[u][s:e] = slot
    # Word ends: the last piece of each non-CJK word. Every tokenizer we use
    # splits at whitespace, so these are the same words for all of them - the
    # positions on which models with different tokenizers can be compared.
    shared = {k for k, (t, s) in enumerate(zip(tokens, slots)) if s != NONE and not _CJK.search(t)}
    ids, pieces, langs, word_end, mixed = [], [], [], [], 0
    for u, text in enumerate(utterances):
        if not text:
            continue
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        strs = tokenizer.convert_ids_to_tokens(enc["input_ids"])
        blank = [is_blank(p) for p in strs]
        lg, m = piece_languages(enc["offset_mapping"], blank, char_slots[u])
        last = {}
        for i, ((s, e), b) in enumerate(zip(enc["offset_mapping"], blank)):
            if not b:
                for k in set(char_tok[u][s:e].tolist()) & shared:
                    last[k] = len(ids) + i
        ends = set(last.values())
        word_end.extend(len(ids) + i in ends for i in range(len(strs)))
        ids.extend(enc["input_ids"])
        pieces.extend(strs)
        langs.extend(lg)
        mixed += m
    return dict(piece_ids=ids, pieces=pieces, piece_lang=langs, word_end=word_end,
                mixed_pieces=mixed)


def build_frame(frame: pd.DataFrame, texts: dict, tokenizer, max_fail: float = 0.001):
    """Sub-word rows for every tagged row; misaligned rows are skipped and counted."""
    out, failed, mixed = [], 0, 0
    has_ent = "is_entity" in frame.columns
    for r in frame.itertuples(index=False):
        res = subword_row(texts[int(r.row_id)], r.tokens, r.token_lang,
                          r.first_language, r.second_language, tokenizer,
                          is_entity=r.is_entity if has_ent else None)
        if res is None:
            failed += 1
            continue
        mixed += res["mixed_pieces"]
        lab = anticipatory_labels(np.asarray(res["piece_lang"], dtype=np.int8))
        out.append(dict(row_id=int(r.row_id), first_language=r.first_language,
                        second_language=r.second_language, quality=r.quality, **res,
                        **{k: lab[k].tolist() for k in LABEL_KEYS}))
    rate = failed / max(len(frame), 1)
    if rate > max_fail:
        raise RuntimeError(f"{failed:,} of {len(frame):,} rows failed to align "
                           f"({rate:.2%} > {max_fail:.2%})")
    stats = dict(rows=len(out), failed=failed, mixed_pieces=mixed,
                 pieces=sum(len(x["piece_ids"]) for x in out))
    return pd.DataFrame(out), stats


def write_parquet(frame: pd.DataFrame, path: str, row_group_size: int = 8192) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    schema = pa.schema([
        ("row_id", pa.int64()), ("first_language", pa.string()),
        ("second_language", pa.string()), ("quality", pa.string()),
        ("pieces", pa.list_(pa.string())), ("piece_ids", pa.list_(pa.int32())),
        ("piece_lang", pa.list_(pa.int8())),
        ("y_sw", pa.list_(pa.int8())), ("y_len", pa.list_(pa.int32())),
        ("y_dur", pa.list_(pa.int8())), ("censored", pa.list_(pa.bool_())),
        ("dur_known", pa.list_(pa.bool_())), ("word_end", pa.list_(pa.bool_())),
        ("mixed_pieces", pa.int32()),
    ])
    table = pa.Table.from_pandas(frame[schema.names], schema=schema, preserve_index=False)
    pq.write_table(table, path, compression="snappy", row_group_size=row_group_size)
