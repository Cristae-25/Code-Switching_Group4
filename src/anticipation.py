"""Anticipatory labels: will the NEXT token switch language, and for how long?

The tagging pipeline answers "what language is token t?". This module turns
that into the *prediction* task: standing at token t with only tokens 1..t
visible, predict two things about t+1.

    y_sw[t]  = 1 if lang(t+1) != lang(t)          binary, defined for t < n-1
    y_dur[t] = class of the length of the run t+1 starts (Small 1-2 | Medium 3-6 | Large 7+)

Both labels are derived, never annotated - they are a deterministic function of
the token-language sequence produced by ``src.dispatch``. That means label
quality is exactly tagger quality, which is why ``quality`` travels with every
row (``exact`` for script-disjoint pairs, ``provisional`` otherwise).

Three decisions that materially change the numbers, made explicit:

1. **Punctuation is dropped, not labelled.** ``l2_word_runs`` treats
   non-alphabetic tokens as transparent - they neither break a run nor count
   toward it. The anticipatory stream uses the same rule, so a run length here
   means the same thing as a run length in the descriptive statistics. Feeding
   punctuation to the model as a predictable "switch" would inflate the
   positive class with tokens nobody wants predicted.
2. **The final token carries no label.** There is no t+1 to predict, so
   position n-1 is masked out of the loss rather than being assigned a
   convenient 0. Assigning 0 would add one free true-negative per row -
   234,172 of them - and quietly lift precision.
3. **Durations at the sequence end are right-censored.** A run still in
   progress when the text ends is *at least* its observed length. It is
   flagged ``censored``, and it is used as a duration target only when that
   lower bound already fixes the class (7+ is Large).
"""
from __future__ import annotations

import numpy as np
import regex

from .segments import DEFAULT_NAMES, absorb_other, resolve_names

# Chinese, Japanese and Korean characters. Every tokenizer we use splits at
# whitespace, so non-CJK words are the same units for all of them; CJK is not.
CJK = regex.compile(r"[\p{Han}\p{Hiragana}\p{Katakana}\p{Hangul}]")

# Language slots. The stream is binarised relative to the row's own pair, so a
# model trained across pairs sees "first vs second language", not "Spanish vs
# English" - that is what makes the universality claim testable.
L1, L2 = 0, 1

# A piece or token that carries no language: punctuation, digits, a bare
# SentencePiece "▁". It stays in the input but is transparent to the labels -
# it neither breaks a run nor counts toward one. Equal to streaming.PAD_LANG.
NONE = 2

# Duration classes from the project requirement: Small 1-2, Medium 3-6,
# Large 7+ tokens. The integer length is kept as y_len; the class is derived.
DUR_EDGES = (2, 6)                     # inclusive upper bounds of small, medium
DUR_LABELS = ["small", "medium", "large"]
N_DUR = len(DUR_LABELS)
LARGE_MIN = DUR_EDGES[-1] + 1          # a cut-off run this long is certainly large

def is_word(token: str) -> bool:
    """A token counts toward a run only if it contains a letter."""
    return any(ch.isalpha() for ch in token)


# How named entities count - see segments.py. Scripts set this from --names.
NAMES = DEFAULT_NAMES


def token_slots(tokens, token_lang, first_language: str, second_language: str,
                is_entity=None, names: str | None = None) -> list[int]:
    """Language slot (L1 / L2 / NONE) for every stored token of a row.

    Applies the names policy and ``absorb_other`` exactly as the word stream
    always has; a token with no letters, or no resolvable language, is NONE.
    """
    tagged = resolve_names(list(zip(tokens, token_lang)), is_entity,
                           first_language, second_language, names or NAMES)
    out = []
    for tok, lang in absorb_other(tagged):
        if not is_word(tok):
            out.append(NONE)
        elif lang == second_language:
            out.append(L2)
        elif lang == first_language:
            out.append(L1)
        else:
            out.append(NONE)                # NE, or unresolved: drop rather than guess
    return out


def word_stream(tokens, token_lang, first_language: str, second_language: str,
                is_entity=None, names: str | None = None):
    """Collapse a tagged row into the (words, language-slot) stream to model.

    Keeps only tokens with a language slot, so punctuation and digits are
    dropped from the stream entirely.
    """
    slots = token_slots(tokens, token_lang, first_language, second_language,
                        is_entity, names)
    keep = [i for i, s in enumerate(slots) if s != NONE]
    return ([tokens[i] for i in keep],
            np.asarray([slots[i] for i in keep], dtype=np.int8))

def run_starts_and_lengths(langs: np.ndarray):
    """Index where each run begins, and its length. Length n == 0 -> empty."""
    n = len(langs)
    if n == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    boundary = np.flatnonzero(langs[1:] != langs[:-1]) + 1
    starts = np.concatenate([[0], boundary])
    ends = np.concatenate([boundary, [n]])
    return starts, ends - starts


def dur_class(length: int) -> int:
    """Map an integer run length to its DUR_LABELS index."""
    for i, upper in enumerate(DUR_EDGES):
        if length <= upper:
            return i
    return N_DUR - 1


def anticipatory_labels(langs: np.ndarray) -> dict:
    """Per-position labels for a language-slot sequence; NONE slots are transparent.

    Position ``t`` describes the next language-bearing position after it.

    * ``y_sw``      0/1 switch indicator (-1 where undefined: NONE slots, last one)
    * ``y_len``     integer length of the run the switch starts (0 elsewhere)
    * ``y_dur``     ``dur_class(y_len)`` (-1 elsewhere)
    * ``censored``  that run is still open at the end of the sequence
    * ``dur_known`` a duration target exists and its class is certain
    """
    n = len(langs)
    y_sw = np.full(n, -1, dtype=np.int8)
    y_len = np.zeros(n, dtype=np.int32)
    y_dur = np.full(n, -1, dtype=np.int8)
    censored = np.zeros(n, dtype=bool)
    dur_known = np.zeros(n, dtype=bool)
    out = dict(y_sw=y_sw, y_len=y_len, y_dur=y_dur, censored=censored,
               dur_known=dur_known)
    idx = np.flatnonzero(np.asarray(langs) != NONE)
    if len(idx) < 2:
        return out
    lb = np.asarray(langs)[idx]
    y_sw[idx[:-1]] = (lb[1:] != lb[:-1]).astype(np.int8)
    starts, lengths = run_starts_and_lengths(lb)
    m = len(lb)
    # A run starting at language-bearing index s>0 is anticipated at idx[s-1].
    for s, ln in zip(starts[1:], lengths[1:]):
        t = idx[s - 1]
        cut = bool(s + ln == m)
        y_len[t] = ln
        y_dur[t] = dur_class(int(ln))
        censored[t] = cut
        dur_known[t] = (not cut) or ln >= LARGE_MIN
    return out

def row_example(row) -> dict | None:
    """Build one word-level training example from a ``load_tagged``-shaped row."""
    words, langs = word_stream(
        row["tokens"], row["token_lang"],
        row["first_language"], row["second_language"],
        is_entity=row.get("is_entity"),
    )
    if len(words) < 2:
        return None
    return dict(
        row_id=int(row["row_id"]),
        words=words,
        langs=langs,
        shared=np.array([not CJK.search(w) for w in words], dtype=bool),
        first_language=row["first_language"],
        second_language=row["second_language"],
        quality=row.get("quality", "unknown"),
        **anticipatory_labels(langs),
    )


def _word_features(words, langs):
    """``unit_end`` (last piece of each word) and ``lang_in`` (the language of the
    last *finished* word) for every piece.

    ``lang_in`` at piece t only uses words that ended before t's own word
    began, so it is a legal causal input: a word's language is known once all
    of its pieces have been read. NONE (the pad slot) where no word has finished.
    """
    n = len(langs)
    unit_end = np.zeros(n, dtype=bool)
    lang_in = np.full(n, NONE, dtype=np.int8)
    wlang, cur, done = {}, None, NONE
    for i in range(n):
        w = int(words[i])
        if w >= 0 and w != cur:
            if cur is not None and cur in wlang:
                done = wlang[cur]
            cur = w
        if w >= 0 and langs[i] != NONE:
            wlang.setdefault(w, langs[i])
        lang_in[i] = done
    nxt = None
    for i in range(n - 1, -1, -1):
        if words[i] >= 0:
            unit_end[i] = words[i] != nxt
            nxt = words[i]
    return unit_end, lang_in


def subword_example(row, turn_sep: bool = False) -> dict | None:
    """Build one sub-word training example from a ``data/subword.parquet`` row.

    ``turn_sep`` inserts a separator piece (id -1, language NONE, so no label
    moves) before the first piece of every utterance after the first.
    """
    ids = np.asarray(row["piece_ids"], dtype=np.int64)
    langs = np.asarray(row["piece_lang"], dtype=np.int8)
    n = len(ids)
    shared = (np.asarray(row["word_end"], dtype=bool) if row.get("word_end") is not None
              else np.zeros(n, dtype=bool))
    words = np.asarray(row["piece_word"], dtype=np.int64) if row.get("piece_word") is not None else None
    utts = np.asarray(row["piece_utt"], dtype=np.int64) if row.get("piece_utt") is not None else None
    if turn_sep:
        if utts is None:
            raise ValueError("--turn-sep needs piece_utt: rebuild the labels with build_subword.py")
        at = np.flatnonzero(np.r_[False, utts[1:] != utts[:-1]]) if n else np.empty(0, int)
        ids, langs = np.insert(ids, at, -1), np.insert(langs, at, NONE)
        shared = np.insert(shared, at, False)
        if words is not None:
            words = np.insert(words, at, -1)
    if int((langs != NONE).sum()) < 2:
        return None
    unit_end, lang_in = _word_features(words, langs) if words is not None else (None, None)
    return dict(
        row_id=int(row["row_id"]),
        ids=ids,
        langs=langs,
        shared=shared,
        unit_end=unit_end,
        lang_in=lang_in,
        first_language=row["first_language"],
        second_language=row["second_language"],
        quality=row.get("quality", "unknown"),
        **anticipatory_labels(langs),
    )
