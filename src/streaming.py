"""Streaming DataLoader for the anticipatory switch task.

The tagged corpus is 22.3M tokens. Materialising it as Python lists of strings
costs roughly 3 GB, so nothing here ever holds more than one Arrow batch. The
parquet is read with ``ParquetFile.iter_batches``, which decodes column pages
incrementally: a full pass over the six columns the task needs peaks at ~175 MB
RSS and takes 0.3 s.

Four things this layer has to get right:

1. **Splits must be row-level, never token-level.** Two windows from the same
   conversation share vocabulary, topic and speaker; splitting tokens would
   leak. ``row_id % 10`` assigns 0-7 train / 8 val / 9 test. The row ids are
   sequential and the source CSVs are contiguous per language, so a modulo is
   an even interleave - every language lands in every split in proportion,
   with no shuffle and no state to persist.
2. **Long rows must be windowed without amputating context.** A conversation
   longer than the window is cut into overlapping windows: the first
   ``window - stride`` positions of every continuation window are *context
   only*, fed to the model but masked out of the loss, so no position is ever
   scored on a truncated prefix.
3. **Workers must not duplicate data.** Each DataLoader worker opens its own
   handle and keeps only the Arrow batches where ``batch_index % num_workers
   == worker_id``.
4. **Shuffling must not require the whole corpus.** A bounded reservoir buffer
   shuffles the stream locally; combined with the interleaved split this is
   enough to decorrelate consecutive windows from the same conversation.
"""
from __future__ import annotations

import random

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .anticipation import N_DUR, NONE, row_example, subword_example
from .evaluation import PAIR_KEYS, pair_key

PAD, UNK, BOS = 0, 1, 2
_SPECIALS = ["<pad>", "<unk>", "<bos>"]
PAD_LANG = 2          # padded positions must not look like first-language
assert NONE == PAD_LANG          # NONE pieces and padding share the "no language" slot

_SUBWORD_COLUMNS = ["row_id", "first_language", "second_language", "quality",
                    "piece_ids", "piece_lang", "word_end", "piece_word", "piece_utt"]
COLUMNS = {"word": None, "subword": _SUBWORD_COLUMNS}      # None = word defaults
DEFAULT_PATHS = {"word": "data/tagged.parquet", "subword": "data/subword.parquet"}
VOCAB_PATHS = {"word": "data/vocab_anticipation.json",
               "subword": "data/vocab_subword.json"}


def example_for(unit: str):
    return {"word": row_example, "subword": subword_example}[unit]

_COLUMNS = ["row_id", "tokens", "token_lang", "first_language",
            "second_language", "quality"]

SPLITS = {"train": set(range(8)), "val": {8}, "test": {9}}


def in_split(row_id: int, split: str) -> bool:
    return split == "all" or (row_id % 10) in SPLITS[split]


def iter_rows(path: str, split: str = "train", batch_rows: int = 512,
              languages=None, qualities=None, columns=None, row_group_order=None):
    """Yield one dict per row, streaming. Never holds more than one batch.

    ``row_group_order`` reads the file's row groups in that order instead of
    file order (the per-epoch reshuffle); every worker must get the same order,
    because workers shard by the running batch index.
    """
    info = get_worker_info()
    wid, nworkers = (info.id, info.num_workers) if info else (0, 1)

    pf = pq.ParquetFile(path)
    # is_entity only exists in parquets tagged after the names-policy change.
    names = pf.schema_arrow.names
    cols = [c for c in (columns or _COLUMNS + ["is_entity"]) if c in names]   # optional columns
    if row_group_order is None:
        batches = pf.iter_batches(batch_size=batch_rows, columns=cols)
    else:
        batches = (b for g in row_group_order
                   for b in pf.iter_batches(batch_size=batch_rows, row_groups=[int(g)], columns=cols))
    for bi, batch in enumerate(batches):
        if bi % nworkers != wid:
            continue
        for row in batch.to_pylist():
            if not in_split(row["row_id"], split):
                continue
            if languages and row["first_language"] not in languages:
                continue
            if qualities and row["quality"] not in qualities:
                continue
            yield row


def build_vocab(path: str, max_size: int = 50_000, split: str = "train",
                min_count: int = 2) -> dict[str, int]:
    """Word vocabulary from the TRAIN split only - val/test words stay unseen."""
    from collections import Counter

    counts: Counter[str] = Counter()
    for row in iter_rows(path, split=split):
        ex = row_example(row)
        if ex:
            counts.update(w.lower() for w in ex["words"])
    vocab = {tok: i for i, tok in enumerate(_SPECIALS)}
    for word, c in counts.most_common():
        if c < min_count or len(vocab) >= max_size:
            break
        vocab[word] = len(vocab)
    return vocab


def build_subword_vocab(path: str, max_size: int = 100_000, split: str = "train",
                        min_count: int = 2) -> dict[str, int]:
    """Compact vocabulary of XLM-R piece ids seen in the TRAIN split.

    Keys are ``str(xlmr_id)`` (JSON keys must be strings). The compact table
    keeps the from-scratch model's embedding matrix small; a pretrained XLM-R
    backbone uses the raw ids instead.
    """
    from .subword import XLMR_VOCAB
    counts = np.zeros(XLMR_VOCAB, dtype=np.int64)
    for row in iter_rows(path, split=split, columns=_SUBWORD_COLUMNS):
        counts += np.bincount(np.asarray(row["piece_ids"], dtype=np.int64),
                              minlength=XLMR_VOCAB)
    vocab = {tok: i for i, tok in enumerate(_SPECIALS)}
    for pid in np.argsort(-counts, kind="stable"):
        if counts[pid] < min_count or len(vocab) >= max_size:
            break
        vocab[str(int(pid))] = len(vocab)
    return vocab


def encode(words, vocab) -> np.ndarray:
    return np.fromiter((vocab.get(w.lower(), UNK) for w in words),
                       dtype=np.int64, count=len(words))


def encode_pieces(ids, vocab, sep_id: int | None = None) -> np.ndarray:
    """Compact ids for the from-scratch model; ``vocab=None`` keeps raw piece ids.

    Id -1 marks an inserted turn separator: the tokenizer's separator
    (``sep_id``) in raw mode, ``<bos>`` in the compact vocabulary.
    """
    if vocab is None:
        out = np.asarray(ids, dtype=np.int64).copy()
        out[out == -1] = BOS if sep_id is None else sep_id
        return out
    return np.fromiter((BOS if int(i) == -1 else vocab.get(str(int(i)), UNK) for i in ids),
                       dtype=np.int64, count=len(ids))


class SwitchWindowDataset(IterableDataset):
    """Streams fixed-length causal windows of the anticipatory task.

    Each yielded window is a dict of length-``window`` arrays. ``sw_mask``
    marks positions that are scored; a position is scored only when it has a
    defined label AND a full prefix inside this window.

    ``unit="subword"`` streams ``data/subword.parquet``: ``word_ids`` hold compact
    piece ids, and NONE pieces are unscored inputs.
    """

    def __init__(self, path: str, vocab: dict[str, int], split: str = "train",
                 window: int = 256, stride: int = 192, shuffle_buffer: int = 4096,
                 seed: int = 0, languages=None, qualities=None,
                 batch_rows: int = 512, pair_index: dict[str, int] | None = None,
                 unit: str = "word", predict_at: str = "piece", lang_input: str = "current",
                 turn_sep: bool = False, sep_id: int | None = None, reshuffle: bool = False):
        assert predict_at in ("piece", "word_end") and lang_input in ("current", "prev_word")
        assert 0 < stride <= window
        self.path, self.vocab, self.split = path, vocab, split
        self.window, self.stride = window, stride
        self.shuffle_buffer = shuffle_buffer if split == "train" else 0
        self.seed = seed
        self.languages, self.qualities = languages, qualities
        self.batch_rows = batch_rows
        self.pair_index = pair_index or {}
        self.unit = unit
        self.predict_at, self.lang_input = predict_at, lang_input
        self.turn_sep, self.sep_id = turn_sep, sep_id
        # reshuffle: a new order every epoch (--reshuffle-epochs). Off, every epoch
        # repeats epoch 0's order, as all runs before 2026-11-01 did. Opt-in because a
        # 12k-step run is 1.02 epochs on 15 pairs and 1.5 on the 6 script pairs, so
        # turning it on changes those runs. The epoch is counted inside __iter__, so it
        # also advances in persistent worker copies.
        self.reshuffle = reshuffle
        self._epochs = 0

    def _windows(self, ex):
        w, s = self.window, self.stride
        ids = (encode_pieces(ex["ids"], self.vocab, self.sep_id) if self.unit == "subword"
               else encode(ex["words"], self.vocab))
        if self.lang_input == "prev_word":        # the last finished word's language
            if ex.get("lang_in") is None:
                raise ValueError("--prev-lang needs piece_word: rebuild the labels")
            langs = ex["lang_in"].astype(np.int64)
        else:
            langs = ex["langs"].astype(np.int64)
        n = len(ids)
        pair = self.pair_index.get(ex["first_language"], 0)
        report = PAIR_KEYS.index(pair_key(ex["first_language"], ex["second_language"]))
        warmup = w - s

        start = 0
        first = True
        while start < n:
            stop = min(start + w, n)
            k = stop - start
            out = {name: np.zeros(w, dtype=dt) for name, dt in
                   (("word_ids", np.int64), ("lang_ids", np.int64),
                    ("y_sw", np.int64), ("y_dur", np.int64))}
            out["sw_mask"] = np.zeros(w, dtype=bool)
            out["dur_mask"] = np.zeros(w, dtype=bool)

            out["word_ids"][:k] = ids[start:stop]
            out["lang_ids"][:] = PAD_LANG
            out["lang_ids"][:k] = langs[start:stop]
            out["cur_lang"] = np.full(w, PAD_LANG, dtype=np.int64)   # for reporting direction
            out["cur_lang"][:k] = ex["langs"][start:stop]
            out["y_sw"][:k] = np.maximum(ex["y_sw"][start:stop], 0)
            out["y_dur"][:k] = np.maximum(ex["y_dur"][start:stop], 0)

            scored = np.zeros(w, dtype=bool)
            scored[:k] = ex["y_sw"][start:stop] >= 0
            if self.predict_at == "word_end":     # only the last piece of each word
                if ex.get("unit_end") is None:
                    raise ValueError("--predict-at word_end needs piece_word: rebuild the labels")
                scored[:k] &= ex["unit_end"][start:stop]
            if not first:
                scored[:warmup] = False          # context-only warm-up
            out["sw_mask"] = scored
            dur = np.zeros(w, dtype=bool)
            dur[:k] = ex["dur_known"][start:stop]
            out["dur_mask"] = dur & scored
            sh = np.zeros(w, dtype=bool)
            if "shared" in ex:                   # hand-built examples may omit it
                sh[:k] = ex["shared"][start:stop]
            out["shared_mask"] = sh & scored     # positions comparable across tokenizers

            out["pair_id"] = np.int64(pair)
            out["report_pair"] = np.int64(report)
            out["row_id"] = np.int64(ex["row_id"])
            out["start"] = np.int64(start)        # window offset in the row: position = start + i
            yield out

            if stop == n:
                break
            start += s
            first = False

    def epoch_order(self, epoch: int):
        """Row-group order for an epoch: file order in epoch 0 (so the first epoch is
        unchanged), then a seeded permutation, the same in every worker."""
        if epoch == 0 or not self.reshuffle:
            return None
        n = pq.ParquetFile(self.path).metadata.num_row_groups
        return np.random.default_rng([self.seed, epoch]).permutation(n).tolist()

    def _stream(self, epoch: int = 0):
        make = example_for(self.unit)
        for row in iter_rows(self.path, self.split, self.batch_rows,
                             self.languages, self.qualities, columns=COLUMNS[self.unit],
                             row_group_order=self.epoch_order(epoch)):
            ex = make(row, turn_sep=True) if self.turn_sep else make(row)
            if ex is None:
                continue
            yield from self._windows(ex)

    def __iter__(self):
        if not self.shuffle_buffer:              # val / test: always file order
            yield from self._stream()
            return
        epoch, self._epochs = self._epochs, self._epochs + 1
        if not self.reshuffle:
            epoch = 0
        stream = self._stream(epoch)
        info = get_worker_info()
        wid = info.id if info else 0
        # Epoch 0 keeps the original seed exactly; later epochs get their own buffer order.
        rng = random.Random(self.seed + wid if epoch == 0 else f"{self.seed}-{wid}-{epoch}")
        buf = []
        for item in stream:
            buf.append(item)
            if len(buf) >= self.shuffle_buffer:
                j = rng.randrange(len(buf))
                buf[j], buf[-1] = buf[-1], buf[j]
                yield buf.pop()
        rng.shuffle(buf)
        yield from buf


def collate(batch):
    out = {}
    for key in ("word_ids", "lang_ids", "cur_lang", "y_sw", "y_dur"):
        out[key] = torch.from_numpy(np.stack([b[key] for b in batch]))
    for key in ("sw_mask", "dur_mask", "shared_mask"):
        out[key] = torch.from_numpy(np.stack([b[key] for b in batch]))
    for key in ("pair_id", "report_pair", "row_id", "start"):
        out[key] = torch.tensor([int(b[key]) for b in batch])
    return out


def make_loader(path, vocab, split="train", batch_size=32, num_workers=0, **kw):
    ds = SwitchWindowDataset(path, vocab, split=split, **kw)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers, collate_fn=collate,
        persistent_workers=bool(num_workers),
    )
