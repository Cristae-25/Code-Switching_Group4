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

from .anticipation import N_DUR, row_example

PAD, UNK, BOS = 0, 1, 2
_SPECIALS = ["<pad>", "<unk>", "<bos>"]
PAD_LANG = 2          # padded positions must not look like first-language

_COLUMNS = ["row_id", "tokens", "token_lang", "first_language",
            "second_language", "quality"]

SPLITS = {"train": set(range(8)), "val": {8}, "test": {9}}


def in_split(row_id: int, split: str) -> bool:
    return split == "all" or (row_id % 10) in SPLITS[split]


def iter_rows(path: str, split: str = "train", batch_rows: int = 512,
              languages=None, qualities=None):
    """Yield one dict per row, streaming. Never holds more than one batch."""
    info = get_worker_info()
    wid, nworkers = (info.id, info.num_workers) if info else (0, 1)

    pf = pq.ParquetFile(path)
    # is_entity only exists in parquets tagged after the names-policy change.
    cols = _COLUMNS + (["is_entity"] if "is_entity" in pf.schema_arrow.names else [])
    for bi, batch in enumerate(pf.iter_batches(batch_size=batch_rows, columns=cols)):
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


def encode(words, vocab) -> np.ndarray:
    return np.fromiter((vocab.get(w.lower(), UNK) for w in words),
                       dtype=np.int64, count=len(words))


class SwitchWindowDataset(IterableDataset):
    """Streams fixed-length causal windows of the anticipatory task.

    Each yielded window is a dict of length-``window`` arrays. ``sw_mask``
    marks positions that are scored; a position is scored only when it has a
    defined label AND a full prefix inside this window.
    """

    def __init__(self, path: str, vocab: dict[str, int], split: str = "train",
                 window: int = 256, stride: int = 192, shuffle_buffer: int = 4096,
                 seed: int = 0, languages=None, qualities=None,
                 batch_rows: int = 512, pair_index: dict[str, int] | None = None):
        assert 0 < stride <= window
        self.path, self.vocab, self.split = path, vocab, split
        self.window, self.stride = window, stride
        self.shuffle_buffer = shuffle_buffer if split == "train" else 0
        self.seed = seed
        self.languages, self.qualities = languages, qualities
        self.batch_rows = batch_rows
        self.pair_index = pair_index or {}

    def _windows(self, ex):
        w, s = self.window, self.stride
        ids = encode(ex["words"], self.vocab)
        langs = ex["langs"].astype(np.int64)
        n = len(ids)
        pair = self.pair_index.get(ex["first_language"], 0)
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
            out["y_sw"][:k] = np.maximum(ex["y_sw"][start:stop], 0)
            out["y_dur"][:k] = np.maximum(ex["y_dur"][start:stop], 0)

            scored = np.zeros(w, dtype=bool)
            scored[:k] = ex["y_sw"][start:stop] >= 0
            if not first:
                scored[:warmup] = False          # context-only warm-up
            out["sw_mask"] = scored
            dur = np.zeros(w, dtype=bool)
            dur[:k] = ex["dur_known"][start:stop]
            out["dur_mask"] = dur & scored

            out["pair_id"] = np.int64(pair)
            out["row_id"] = np.int64(ex["row_id"])
            yield out

            if stop == n:
                break
            start += s
            first = False

    def _stream(self):
        for row in iter_rows(self.path, self.split, self.batch_rows,
                             self.languages, self.qualities):
            ex = row_example(row)
            if ex is None:
                continue
            yield from self._windows(ex)

    def __iter__(self):
        stream = self._stream()
        if not self.shuffle_buffer:
            yield from stream
            return
        info = get_worker_info()
        rng = random.Random(self.seed + (info.id if info else 0))
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
    for key in ("word_ids", "lang_ids", "y_sw", "y_dur"):
        out[key] = torch.from_numpy(np.stack([b[key] for b in batch]))
    for key in ("sw_mask", "dur_mask"):
        out[key] = torch.from_numpy(np.stack([b[key] for b in batch]))
    for key in ("pair_id", "row_id"):
        out[key] = torch.tensor([int(b[key]) for b in batch])
    return out


def make_loader(path, vocab, split="train", batch_size=32, num_workers=0, **kw):
    ds = SwitchWindowDataset(path, vocab, split=split, **kw)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers, collate_fn=collate,
        persistent_workers=bool(num_workers),
    )
