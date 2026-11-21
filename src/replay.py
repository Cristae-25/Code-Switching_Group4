"""Replay an anticipation model's predictions on one row, decision point by decision point.

Back end of ``scripts/analysis/replay_server.py``. Each run is identified by its tag:
``data/results/anticipation_<tag>.json`` holds its settings and threshold, and two
sources can supply its predictions:

* ``saved``: ``data/preds/<tag>.npz`` (or the same arrays unpacked into
  ``data/preds/<tag>/``), written by ``train_anticipation.py --save-preds``. Every scored test position has a switch logit, but the duration
  head was saved only at ``dur_mask`` positions (true switches whose class is
  known), and without their positions. Those positions are rebuilt here from the
  row's windows, which come out in the order evaluation wrote them.
* ``live``: ``models/anticipation_<tag>.pt``, the checkpoint. The model runs on CPU,
  and the duration head is read at every scored position, so a prediction is
  available wherever the model fires.

Both sources go through the training code's own windowing
(``SwitchWindowDataset._windows``), so a row's decision points are exactly the
positions that were scored. Saved mode checks that and raises on any mismatch,
rather than showing predictions against the wrong pieces.

Duration is shown as **Short (1-6) vs Large (7+)** by default. A binary head
(``--dur-classes 2``) is read directly, with its val-tuned class offsets. A 3-class
head is collapsed: P(short) = P(small) + P(medium), and the prediction is Large when
P(large) > the val-tuned cut in ``data/results/duration_binary_<tag>.json`` (0.5 if
there is none).

``dataset="lince"`` replays LinCE tweets (``src/lince.py``) instead: human gold
labels, saved predictions from ``eval_lince.py --save-preds data/preds/lince_<tag>.npz``.
"""
from __future__ import annotations

import json
import os
import unicodedata
from dataclasses import dataclass, field

import numpy as np
import pyarrow.parquet as pq

from .anticipation import DUR_LABELS, NONE
from .evaluation import PAIR_KEYS, pair_key
from .streaming import SwitchWindowDataset, example_for, in_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RUNS = {"xlmr": "xlmr_prev_gap_36k_rsv2", "mbert": "mbert_prev_gap_36k_rsv2",
                "xlmr_bin": "xlmr_bin_prev_gap_36k_rsv2", "mbert_bin": "mbert_bin_prev_gap_36k_rsv2"}
DATASETS = ("switchlingua", "lince")
ROW_COLUMNS = ["row_id", "first_language", "second_language", "quality", "pieces",
               "piece_ids", "piece_lang", "word_end", "piece_word", "piece_utt"]
TWO_CLASS_LABELS = ["short", "large"]
LARGE = 2                                  # index of Large in the 3-class labels


def _p(path: str, root: str) -> str:
    return path if os.path.isabs(path) else os.path.join(root, path)


@dataclass
class Run:
    """One trained model: its settings, its threshold and where its predictions are."""
    name: str
    tag: str
    args: dict
    tau: float
    root: str = ROOT
    dataset: str = "switchlingua"
    dur_offsets: list | None = None        # binary head: val-tuned class offsets
    _blank: list | None = field(default=None, repr=False)

    @classmethod
    def from_tag(cls, name: str, tag: str, root: str = ROOT, dataset: str = "switchlingua") -> "Run":
        res = json.load(open(os.path.join(root, "data", "results", f"anticipation_{tag}.json")))
        off = res.get("duration_tuned", {}).get("offsets")
        return cls(name, tag, res["args"], float(res["tau"]), root, dataset, off)

    @property
    def backbone(self) -> str:
        from .xlmr_model import BACKBONES
        return self.args.get("backbone") or BACKBONES.get(self.model, BACKBONES["xlmr"])

    def path_for(self, row_id: int) -> str:
        """The parquet holding ``row_id``: the run's training file, or a LinCE file."""
        if self.dataset == "lince":
            from .lince import describe_row, parquet_path
            pair, split, _ = describe_row(int(row_id))
            return parquet_path(self.backbone, pair, split)
        return self.path

    @property
    def preds_path(self) -> str | None:
        """Saved predictions: an unpacked directory or an ``.npz``; None if neither exists."""
        name = f"lince_{self.tag}" if self.dataset == "lince" else self.tag
        base = os.path.join(self.root, "data", "preds", name)
        if os.path.isdir(base):
            return base
        return base + ".npz" if os.path.exists(base + ".npz") else None

    @property
    def pred_splits(self) -> tuple[str, ...]:
        return ("validation", "train") if self.dataset == "lince" else ("test",)

    @property
    def large_cut(self) -> float:
        """3-class heads: P(large) above which the Short/Large decision is Large."""
        f = os.path.join(self.root, "data", "results", f"duration_binary_{self.tag}.json")
        if os.path.exists(f):
            return float(json.load(open(f))["two_class"]["cut"])
        return 0.5

    def short_large(self, probs) -> tuple[list[float], int]:
        """``([P(short), P(large)], prediction)`` from this head's duration probabilities."""
        probs = np.asarray(probs, dtype=np.float64)
        if self.n_dur == 2:
            off = np.asarray(self.dur_offsets if self.dur_offsets else [0.0, 0.0])
            pred = int((np.log(np.clip(probs, 1e-12, None)) + off).argmax())
            return [float(probs[0]), float(probs[1])], pred
        pl = float(probs[LARGE])
        return [float(probs[:LARGE].sum()), pl], int(pl > self.large_cut)

    @property
    def model(self) -> str:
        return self.args.get("model", "scratch")

    @property
    def path(self) -> str:
        return _p(self.args["path"], self.root)

    @property
    def ckpt(self) -> str:
        return _p(self.args.get("ckpt") or f"models/anticipation_{self.tag}.pt", self.root)

    @property
    def n_dur(self) -> int:
        return int(self.args.get("dur_classes") or 3)

    @property
    def dur_labels(self) -> list[str]:
        return TWO_CLASS_LABELS if self.n_dur == 2 else DUR_LABELS

    def modes(self) -> list[str]:
        out = []
        if self.preds_path:
            out.append("saved")
        if os.path.exists(self.ckpt):
            out.append("live")
        return out

    def blank_ids(self):
        if self.args.get("decide_at", "word") != "gap":
            return None
        if self._blank is None:
            from .subword import blank_piece_ids
            self._blank = blank_piece_ids(self.backbone)
        return self._blank


def discover_runs(extra: dict[str, str] | None = None, root: str = ROOT,
                  dataset: str = "switchlingua") -> dict[str, Run]:
    """The default runs plus ``extra`` (name -> tag) that have a result json and a source."""
    if dataset not in DATASETS:
        raise ValueError(f"dataset is one of {DATASETS}")
    runs = {}
    for name, tag in {**DEFAULT_RUNS, **(extra or {})}.items():
        if os.path.exists(os.path.join(root, "data", "results", f"anticipation_{tag}.json")):
            run = Run.from_tag(name, tag, root, dataset)
            if run.modes():
                runs[name] = run
    return runs


# --------------------------------------------------------------------------- rows
def load_row(path: str, row_id: int) -> dict:
    names = pq.read_schema(path).names
    cols = [c for c in ROW_COLUMNS if c in names]
    rows = pq.read_table(path, columns=cols, filters=[("row_id", "=", int(row_id))]).to_pylist()
    if not rows:
        raise KeyError(f"row {row_id} is not in {path}")
    return rows[0]


def row_windows(run: Run, row: dict, blank_ids=None) -> tuple[dict | None, list[dict]]:
    """The row's training example and its windows, built exactly as evaluation built them."""
    decide_at = run.args.get("decide_at", "word")
    kw = {}
    if decide_at == "gap":
        kw = dict(decide_at="gap", blank_ids=blank_ids if blank_ids is not None else run.blank_ids())
    if run.args.get("turn_sep"):
        raise NotImplementedError("replay of --turn-sep runs is not supported")
    ex = example_for("subword")(row, **kw)
    if ex is None:
        return None, []
    ds = SwitchWindowDataset(run.path, None, split="test", window=int(run.args["window"]),
                             stride=int(run.args["stride"]), unit="subword",
                             predict_at=run.args.get("predict_at", "piece"),
                             lang_input="prev_word" if run.args.get("prev_lang") else "current",
                             decide_at=decide_at, blank_ids=kw.get("blank_ids"))
    return ex, list(ds._windows(ex))


def scored(windows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Row positions of every scored position, and of every ``dur_mask`` position, in order."""
    sw = [w["start"] + np.flatnonzero(w["sw_mask"]) for w in windows]
    du = [w["start"] + np.flatnonzero(w["dur_mask"]) for w in windows]
    cat = lambda xs: np.concatenate(xs).astype(np.int64) if xs else np.empty(0, np.int64)
    return cat(sw), cat(du)


# --------------------------------------------------------------------------- sources
class SavedPreds:
    """A run's saved predictions (one or more splits), sliced per row."""

    KEYS = ("row", "pos", "logit", "pair", "drow", "dprob")

    def __init__(self, source: str, splits=("test",)):
        if isinstance(splits, str):
            splits = (splits,)
        if os.path.isdir(source):
            def ld(split, k):
                f = os.path.join(source, f"{split}_{k}.npy")
                return np.load(f, mmap_mode="r") if os.path.exists(f) else None
        else:
            z = np.load(source)
            ld = lambda split, k: z[f"{split}_{k}"] if f"{split}_{k}" in z.files else None  # noqa: E731
        arrays = {k: [] for k in self.KEYS}
        for split in splits:
            got = {k: ld(split, k) for k in self.KEYS}
            if got["row"] is None:
                continue
            for k in self.KEYS:
                arrays[k].append(got[k])
        if not arrays["row"]:
            raise FileNotFoundError(f"no {'/'.join(splits)} predictions in {source}")
        cat = (lambda xs: xs[0]) if len(arrays["row"]) == 1 else np.concatenate  # noqa: E731
        self.row, self.pos, self.logit = cat(arrays["row"]), cat(arrays["pos"]), cat(arrays["logit"])
        self.pair = cat(arrays["pair"])
        self.drow, self.dprob = cat(arrays["drow"]), cat(arrays["dprob"])
        self._idx = self._index(np.asarray(self.row))
        self._didx = self._index(np.asarray(self.drow))

    @staticmethod
    def _index(rows: np.ndarray) -> dict[int, tuple[int, int]]:
        if not len(rows):
            return {}
        cut = np.flatnonzero(np.r_[True, rows[1:] != rows[:-1], True])
        out = {}
        for a, b in zip(cut[:-1], cut[1:]):
            r = int(rows[a])
            if r in out:
                raise ValueError(f"row {r} is not contiguous in the saved predictions")
            out[r] = (int(a), int(b))
        return out

    def row_ids(self, pair: str | None = None) -> np.ndarray:
        ids = np.fromiter(self._idx.keys(), dtype=np.int64)
        if pair:
            k = PAIR_KEYS.index(pair)
            starts = np.fromiter((v[0] for v in self._idx.values()), dtype=np.int64)
            ids = ids[np.asarray(self.pair)[starts] == k]
        return ids

    def predict(self, row_id: int, sw_pos: np.ndarray, dur_pos: np.ndarray):
        """``(p_switch at sw_pos, {position: duration probs})`` for one row."""
        if row_id not in self._idx:
            raise KeyError(f"row {row_id} has no saved predictions")
        a, b = self._idx[row_id]
        pos = np.asarray(self.pos[a:b])
        if not np.array_equal(pos, sw_pos):
            raise ValueError(f"row {row_id}: saved positions differ from the rebuilt windows "
                             f"({len(pos)} saved, {len(sw_pos)} rebuilt)")
        p = 1 / (1 + np.exp(-np.asarray(self.logit[a:b], dtype=np.float64)))
        da, db = self._didx.get(row_id, (0, 0))
        if db - da != len(dur_pos):
            raise ValueError(f"row {row_id}: {db - da} saved duration predictions, "
                             f"{len(dur_pos)} duration positions rebuilt")
        probs = np.asarray(self.dprob[da:db], dtype=np.float64)
        return p, {int(q): probs[i] for i, q in enumerate(dur_pos)}


class LivePredictor:
    """Runs a checkpoint on CPU. The model is built once and kept."""

    def __init__(self, run: Run):
        from .models import load_checkpoint
        self.model, ck = load_checkpoint(run.ckpt, fallback_args=run.args)
        self.tau = float(ck.get("tau", run.tau))

    @classmethod
    def from_model(cls, model, tau: float) -> "LivePredictor":
        self = cls.__new__(cls)
        self.model, self.tau = model.eval(), tau
        return self

    def predict(self, windows: list[dict]):
        """``(positions, p_switch, duration probs)`` at every scored position."""
        import torch

        from .causal_model import duration_probs
        from .models import pair_ids
        from .streaming import collate
        if not windows:
            return np.empty(0, np.int64), np.empty(0), np.empty((0, 0))
        b = collate(windows)
        with torch.no_grad():
            sw, dur = self.model(b["word_ids"], b["lang_ids"], pair_ids(self.model, b))
        m = b["sw_mask"]
        pos = (b["start"].unsqueeze(1) + torch.arange(m.shape[1])).expand_as(m)[m].numpy()
        p = torch.sigmoid(sw.float()[m]).numpy().astype(np.float64)
        probs = duration_probs(dur.float()[m], getattr(self.model, "dur_ordinal", False))
        return pos, p, probs.numpy().astype(np.float64)


# --------------------------------------------------------------------------- display
def _cjk(ch: str) -> bool:
    return unicodedata.name(ch, "").split(" ")[0] in ("CJK", "HIRAGANA", "KATAKANA", "HANGUL")


def display_pieces(pieces: list[str], words: list[int] | None) -> list[str]:
    """Each piece's text with the space that precedes it (``▁`` / ``##`` resolved).

    The source text is not stored, so this is a reconstruction: SentencePiece
    marks word starts with ``▁``; WordPiece marks continuations with ``##`` and
    otherwise gets a space between words, except between CJK characters and
    before punctuation.
    """
    sentencepiece = any(p.startswith("▁") for p in pieces)
    out = []
    for i, p in enumerate(pieces):
        if sentencepiece:
            out.append((" " if i and p.startswith("▁") else "") + p.lstrip("▁"))
        elif p.startswith("##"):
            out.append(p[2:])
        elif i == 0:
            out.append(p)
        else:
            prev = pieces[i - 1]
            same = words is not None and words[i] == words[i - 1]
            tight = (same or not p or not prev or (_cjk(p[0]) and _cjk(prev[-1]))
                     or unicodedata.category(p[0])[0] == "P")
            out.append(p if tight else " " + p)
    return out


# --------------------------------------------------------------------------- replay
def replay_row(run: Run, row_id: int, mode: str = "saved", saved: SavedPreds | None = None,
               live: LivePredictor | None = None, row: dict | None = None) -> dict:
    """Everything the UI needs to replay one row: pieces, truth and every decision point."""
    row = row if row is not None else load_row(run.path_for(row_id), row_id)
    ex, windows = row_windows(run, row)
    pieces = list(row["pieces"])
    words = row.get("piece_word")
    text = display_pieces(pieces, words)
    out = dict(row_id=int(row["row_id"]), run=run.name, tag=run.tag, mode=mode,
               model=run.model, first_language=row["first_language"],
               second_language=row["second_language"],
               pair=pair_key(row["first_language"], row["second_language"]),
               pieces=text, piece_lang=[int(x) for x in row["piece_lang"]],
               dur_labels=run.dur_labels, n_dur=run.n_dur, dataset=run.dataset,
               large_cut=round(run.large_cut, 4) if run.n_dur == 3 else None, decisions=[])
    if ex is None:
        out.update(tau=run.tau, summary=_summary([]))
        return out
    sw_pos, dur_pos = scored(windows)
    if mode == "saved":
        if saved is None:
            raise ValueError("saved mode needs the run's SavedPreds")
        p, dprobs = saved.predict(int(row["row_id"]), sw_pos, dur_pos)
        tau = run.tau
    elif mode == "live":
        if live is None:
            raise ValueError("live mode needs a LivePredictor")
        pos, p, probs = live.predict(windows)
        if not np.array_equal(pos, sw_pos):
            raise ValueError("live positions differ from the rebuilt windows")
        dprobs = {int(q): probs[i] for i, q in enumerate(pos)}
        tau = live.tau
    else:
        raise ValueError(f"mode is 'saved' or 'live', not {mode!r}")

    decisions = []
    for i, q in enumerate(sw_pos):
        q = int(q)
        y = int(ex["y_sw"][q])
        fired = bool(p[i] >= tau)
        ylen = int(ex["y_len"][q]) if y == 1 else None
        dp = dprobs.get(q)
        known = bool(y and ex["dur_known"][q])
        y3 = int(ex["y_dur"][q]) if known else None           # the stored 3-class label
        y_bin = None if y3 is None else int(y3 == LARGE)      # Short (0) / Large (1)
        bp, bpred = run.short_large(dp) if dp is not None else (None, None)
        decisions.append(dict(
            pos=q, p=round(float(p[i]), 4), fired=fired, y=y,
            outcome=("TP" if fired else "FN") if y else ("FP" if fired else "TN"),
            next_lang=_next_lang(ex["cur_lang"] if "cur_lang" in ex else ex["langs"], q, y),
            y_len=ylen, censored=bool(ex["censored"][q]) if y else None,
            dur_known=bool(ex["dur_known"][q]) if y else None,
            y_dur=y_bin if run.n_dur == 2 else y3,               # in the head's own classes
            dur_probs=None if dp is None else [round(float(x), 4) for x in dp],
            dur_pred=None if dp is None else int(np.argmax(dp)),
            y_bin=y_bin, bin_probs=None if bp is None else [round(x, 4) for x in bp],
            bin_pred=bpred,
        ))
    out.update(tau=round(float(tau), 4), decisions=decisions, summary=_summary(decisions))
    return out


def _next_lang(cur: np.ndarray, q: int, y: int) -> int | None:
    """The language the next word is in when the label is a switch (the opposite of
    the word the label belongs to), else the current language."""
    c = int(cur[q])
    if c == NONE:
        return None
    return 1 - c if y else c


def _summary(decisions: list[dict]) -> dict:
    c = {k: sum(d["outcome"] == k for d in decisions) for k in ("TP", "FP", "FN", "TN")}
    tp, fp, fn = c["TP"], c["FP"], c["FN"]
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None
    return dict(**c, f1=None if f1 is None else round(f1, 4), n=len(decisions))


def candidate_paths(run: Run) -> list[str]:
    """Parquets live mode samples from: the training file, or the LinCE files for this
    tokenizer (the validation split is built on first use)."""
    if run.dataset != "lince":
        return [run.path]
    from . import lince
    paths = [lince.parquet_path(run.backbone, p, s) for p in lince.PAIRS for s in lince.SPLITS]
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        paths = [lince.build(p, "validation", run.backbone)[0] for p in lince.PAIRS]
    return paths


def random_row(rng: np.random.Generator, saved: SavedPreds | None, run: Run,
               pair: str | None = None) -> int:
    """A random row: from the saved predictions if there are any, else from the parquet
    (SwitchLingua: test split only, so a live model never replays a row it trained on)."""
    if saved is not None:
        ids = saved.row_ids(pair)
    else:
        cols = ["row_id", "first_language", "second_language"]
        chunks = []
        for path in candidate_paths(run):
            t = pq.read_table(path, columns=cols)
            ids = np.asarray(t.column("row_id"))
            keep = np.ones(len(ids), dtype=bool)
            if run.dataset != "lince":
                keep = np.fromiter((in_split(int(r), "test") for r in ids), dtype=bool, count=len(ids))
            if pair:
                l1, l2 = pair.split("->")
                keep &= (np.asarray(t.column("first_language")) == l1) & \
                        (np.asarray(t.column("second_language")) == l2)
            chunks.append(ids[keep])
        ids = np.concatenate(chunks) if chunks else np.empty(0, np.int64)
    if not len(ids):
        raise KeyError(f"no rows for {pair or 'any pair'}")
    return int(rng.choice(ids))
