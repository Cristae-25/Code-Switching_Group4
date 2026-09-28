# Sub-word Token Labels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Label the corpus at the XLM-R sub-word token level. Every piece takes its word's final language. Per piece, store the switch label, the integer duration of the run a switch starts, and its class (Small 1–2, Medium 3–6, Large 7+), all in `data/subword.parquet`. No model work: streaming, model inputs and training on these labels come later (Week 5).

**Architecture:**
- **Build step.** A one-off script aligns the original SwitchLingua text with the stored per-word labels, tokenizes it with XLM-R, and writes `data/subword.parquet` (piece ids plus a per-piece language slot, in shuffled row order).
- **Shared labels.** `src/anticipation.py` builds the labels for both units from a language-slot array in which slot `NONE` (punctuation, digits, a bare `▁`) is transparent.
- **Stored labels.** The parquet stores the labels themselves (`y_sw`, `y_len`, `y_dur`, `censored`, `dur_known`) next to the readable `pieces`, so they can be used and inspected without code.

**Tech Stack:** Python 3.13, numpy, pandas, pyarrow, torch 2.14, transformers 5.17 (the `xlm-roberta-base` fast tokenizer, already cached in `~/.cache/huggingface/hub/models--xlm-roberta-base`), pytest.

**Spec:** `docs/design/2026-09-27-subword-labels-design.md`

## Global Constraints

- **Environment:**
  - Always run `./.venv/bin/python` from the repo root.
  - Add no dependencies (`requirements.txt` stays unchanged).
- **Tokenizer:** the `xlm-roberta-base` fast tokenizer, `add_special_tokens=False`, `return_offsets_mapping=True`. Tests load it with `local_files_only=True` and skip if it isn't cached.
- **Duration classes:** `0` Small (1–2), `1` Medium (3–6), `2` Large (7+). Store the integer length as `y_len`.
- **Cut-off runs:** a run still open at the end of the row counts for duration only if its class is certain: `dur_known = switch and (not censored or y_len >= 7)`.
- **Splits:** row-level `row_id % 10`: 0–7 train, 8 val, 9 test. Never split a row.
- **Names:** resolved at the word level with `resolve_names(..., DEFAULT_NAMES)` (`"script_words"`) before being mapped to pieces. A name piece is `NONE` only if it has no letters.
- **Language slots:** `L1 = 0`, `L2 = 1`, `NONE = 2`. `NONE` must equal `streaming.PAD_LANG` (`2`).
- **Git:**
  - Data outputs stay gitignored (`data/*.parquet` already is; add `data/vocab_subword.json`).
  - Commit with `git -c user.name="Cristae-25" -c user.email="teh.c@northeastern.edu" commit`.
  - End every message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Out of scope:** streaming, model inputs, training, and report results. Existing models must keep loading (Task 1).

## Review Focus

1. **Characters `tokenize_mixed` drops** (emoji, zero-width joiners, Devanagari combining marks). Alignment must still find every stored token in order. Tested in Task 2 (`test_align_survives_emoji_zwj_and_combining_marks`).
2. **Empty utterances inside a row** (`["", "hola"]`). These must align and tokenize to zero pieces, not crash. Tested in Task 2 (`test_empty_utterance_contributes_no_pieces`).
3. **A bare `▁` piece whose offsets point at the next character** (XLM-R emits `('▁', '昨', (0,1))` before `('昨天', (0,2))`, and `('▁', '，')` before `','`). It must be `NONE`, or a character is counted twice. Tested in Task 2 (`test_bare_space_marker_piece_is_none`).
4. **Rows with fewer than 2 language-bearing pieces** (for example `"Hola!"`). They must be kept, with every label -1, not silently dropped. Tested in Task 2 (`test_rows_too_short_to_label_are_kept_with_empty_labels`).
5. **Existing 7-class checkpoints** after the switch to 3 classes. `eval_anticipation_pair.py` must still load them by reading the duration-head size from the state dict. Tested in Task 1 (`test_seven_class_checkpoint_loads_with_matching_n_dur`).

---

### Task 1: Integer run lengths, 3 duration classes, transparent `NONE` slots

**Files:**
- Modify: `src/anticipation.py`: the duration constants, `anticipatory_labels`, `word_stream`, `row_example`; add `NONE`, `dur_class`, `token_slots`.
- Modify: `src/streaming.py:151` (the duration mask).
- Modify: `scripts/baseline_anticipation.py:38,72` (imports and the duration mask).
- Modify: `src/causal_model.py` (the `n_dur` parameter); `scripts/eval_anticipation_pair.py` (read `n_dur` from the checkpoint).
- Test: `tests/test_anticipation.py`

**Interfaces:**
- Produces:
  - `NONE: int = 2`
  - `DUR_EDGES = (2, 6)`, `DUR_LABELS = ["small", "medium", "large"]`, `N_DUR = 3`, `LARGE_MIN = 7`
  - `dur_class(length: int) -> int`
  - `anticipatory_labels(langs: np.ndarray) -> dict` with the keys:
    - `y_sw` (int8, -1 where undefined)
    - `y_len` (int32, 0 where no switch)
    - `y_dur` (int8 class, -1 where no switch)
    - `dur_known` (bool)
    - `censored` (bool)
  - `token_slots(tokens, token_lang, first_language, second_language, is_entity=None, names=None) -> list[int]`
  - `row_example(row)` returns `row_id, words, langs, first_language, second_language, quality` plus every key of `anticipatory_labels`.
- Produces: `CausalSwitchModel(..., n_dur: int = N_DUR)`.
- Removes: `bucket`, `DUR_BUCKETS`, and the `dur_valid` key.

- [ ] **Step 1: Rewrite the label tests in `tests/test_anticipation.py`**

Replace the import block and the whole `# --- labels ---` section (everything from line 1 through `test_single_token_row_yields_no_labels`) with:

```python
"""Label-generation, streaming and causality tests for the anticipatory task."""
import numpy as np
import pytest

from src.anticipation import (LARGE_MIN, N_DUR, NONE, anticipatory_labels, dur_class,
                              row_example, run_starts_and_lengths, token_slots,
                              word_stream)
from src.metrics import prf
from src.streaming import SwitchWindowDataset, in_split


# --- labels -----------------------------------------------------------------

def test_switch_label_is_about_the_next_token():
    langs = np.array([0, 0, 1, 1, 0], dtype=np.int8)
    assert list(anticipatory_labels(langs)["y_sw"]) == [0, 1, 0, 1, -1]


def test_final_position_has_no_label():
    lab = anticipatory_labels(np.array([0, 1], dtype=np.int8))
    assert lab["y_sw"][-1] == -1 and lab["y_dur"][-1] == -1 and not lab["dur_known"][-1]


def test_duration_is_the_integer_length_of_the_run_that_starts_next():
    lab = anticipatory_labels(np.array([0, 1, 1, 1, 0, 0], dtype=np.int8))
    assert lab["y_len"][0] == 3 and lab["y_dur"][0] == 1      # 3 -> medium
    assert lab["y_len"][3] == 2 and lab["y_dur"][3] == 0      # 2 -> small
    assert list(np.flatnonzero(lab["y_dur"] >= 0)) == [0, 3]


def test_duration_classes_cut_at_2_and_6():
    assert [dur_class(n) for n in (1, 2, 3, 6, 7, 1040)] == [0, 0, 1, 1, 2, 2]
    assert N_DUR == 3 and LARGE_MIN == 7


def test_censored_run_counts_only_once_its_class_is_certain():
    large = anticipatory_labels(np.array([0] * 3 + [1] * 7, dtype=np.int8))
    assert large["censored"][2] and large["y_len"][2] == 7 and large["dur_known"][2]
    medium = anticipatory_labels(np.array([0] * 3 + [1] * 6, dtype=np.int8))
    assert medium["censored"][2] and not medium["dur_known"][2]
    closed = anticipatory_labels(np.array([0, 0, 1, 1, 0], dtype=np.int8))
    assert not closed["censored"][1] and closed["dur_known"][1]


def test_none_slots_are_transparent():
    # language-bearing positions 0,2,3,5,6 carry L1 L1 L2 L2 L1
    langs = np.array([0, NONE, 0, 1, NONE, 1, 0], dtype=np.int8)
    lab = anticipatory_labels(langs)
    assert list(lab["y_sw"]) == [0, -1, 1, 0, -1, 1, -1]
    assert lab["y_len"][2] == 2 and lab["dur_known"][2]       # L2 run of 2 pieces
    assert lab["y_len"][5] == 1 and lab["censored"][5]        # final L1 run, cut off
    assert not lab["dur_known"][5]


def test_single_token_row_yields_no_labels():
    lab = anticipatory_labels(np.array([1], dtype=np.int8))
    assert list(lab["y_sw"]) == [-1] and not lab["dur_known"].any()


def _previous_labels(langs):
    """The pre-change implementation: y_sw, raw run length, censored."""
    n = len(langs)
    y_sw = np.full(n, -1)
    lens = np.zeros(n, dtype=int)
    cens = np.zeros(n, dtype=bool)
    y_sw[:-1] = langs[1:] != langs[:-1]
    starts, lengths = run_starts_and_lengths(langs)
    for s, ln in zip(starts[1:], lengths[1:]):
        lens[s - 1] = ln
        cens[s - 1] = s + ln == n
    return y_sw, lens, cens


def test_word_level_labels_match_the_previous_implementation():
    rng = np.random.default_rng(0)
    for _ in range(200):
        langs = (rng.random(rng.integers(2, 60)) < 0.3).astype(np.int8)
        y_sw, lens, cens = _previous_labels(langs)
        lab = anticipatory_labels(langs)
        assert (lab["y_sw"] == y_sw).all()
        assert (lab["y_len"] == lens).all()
        assert (lab["censored"] == cens).all()


def test_token_slots_mark_non_words_none():
    assert token_slots(["hola", ",", "hello", "2"], ["Spanish", "other", "English", "other"],
                       "Spanish", "English") == [0, NONE, 1, NONE]


def test_seven_class_checkpoint_loads_with_matching_n_dur():
    pytest.importorskip("torch")
    from src.causal_model import CausalSwitchModel
    old = CausalSwitchModel(200, d_model=32, n_layers=2, n_heads=2, d_ff=64, n_dur=7)
    state = old.state_dict()
    new = CausalSwitchModel(200, d_model=32, n_layers=2, n_heads=2, d_ff=64,
                            n_dur=state["head_dur.weight"].shape[0])
    new.load_state_dict(state)
    assert N_DUR == 3 and new.head_dur.out_features == 7
```

In the `# --- streaming ---` section, replace each of the three two-line blocks

```python
    ex["y_sw"], ex["y_dur"], ex["dur_valid"], ex["censored"] = \
        anticipatory_labels(ex["langs"])
```

with

```python
    ex.update(anticipatory_labels(ex["langs"]))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./.venv/bin/python -m pytest tests/test_anticipation.py -q`
Expected: collection ERROR, `ImportError: cannot import name 'LARGE_MIN'`.

- [ ] **Step 3: Implement in `src/anticipation.py`**

Replace the duration-bucket block (the `DUR_BUCKETS` comment through `N_DUR = len(DUR_LABELS)`) with:

```python
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
```

Replace `word_stream` with `token_slots` plus a thin `word_stream`:

```python
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
```

Replace `anticipatory_labels` and `bucket` with:

```python
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
```

Replace `row_example`:

```python
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
        first_language=row["first_language"],
        second_language=row["second_language"],
        quality=row.get("quality", "unknown"),
        **anticipatory_labels(langs),
    )
```

In the module docstring, change the `y_dur` line to
`    y_dur[t] = class of the length of the run t+1 starts (Small 1-2 | Medium 3-6 | Large 7+)`,
and change decision 3 to say: "A run still in progress when the text ends is *at least* its observed length. It is flagged `censored`, and it is used as a duration target only when that lower bound already fixes the class (7+ is Large)."

- [ ] **Step 4: Update the two callers**

In `src/streaming.py`, the `_windows` duration mask (line 151), replace

```python
            dur[:k] = ex["dur_valid"][start:stop] & ~ex["censored"][start:stop]
```
with
```python
            dur[:k] = ex["dur_known"][start:stop]
```

In `scripts/baseline_anticipation.py`, change the import on line 38 to
```python
from src.anticipation import L2, N_DUR, DUR_LABELS, row_example  # noqa: E402
```
and line 72 to
```python
        dur_ok.append(ex["dur_known"][m])
```

In `src/causal_model.py`:
- Change the constructor signature's last line to `dropout: float = 0.1, n_pairs: int = 16, use_pair: bool = False, n_dur: int = N_DUR):`.
- Change `self.head_dur = nn.Linear(d_model, N_DUR)` to `self.head_dur = nn.Linear(d_model, n_dur)`.

In `scripts/eval_anticipation_pair.py`, add this argument to the `CausalSwitchModel(...)` call:
```python
                              n_dur=ck["state"]["head_dur.weight"].shape[0],
```

- [ ] **Step 5: Run the full suite**

Run: `./.venv/bin/python -m pytest tests/ -q`
Expected: all pass. Six label tests become ten, so **214 passed**.

Then check that an existing checkpoint still loads (it was trained with 7 classes):
`./.venv/bin/python scripts/eval_anticipation_pair.py --ckpt models/anticipation_zeroshot.pt --l1 German --l2 French`
Expected: `German -> French ... F1 0.243`, unchanged.

- [ ] **Step 6: Commit**

```bash
git add src/anticipation.py src/streaming.py src/causal_model.py scripts/baseline_anticipation.py scripts/eval_anticipation_pair.py tests/test_anticipation.py
git -c user.name="Cristae-25" -c user.email="teh.c@northeastern.edu" commit -m "Integer run lengths and 3 duration classes; NONE slots are transparent

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `src/subword.py`: align words, give each piece its word's language, build the frame

**Files:**
- Create: `src/subword.py`
- Test: `tests/test_subword.py`

**Interfaces:**
- Consumes: `NONE`, `L1`, `L2`, `token_slots` (Task 1); `src.tokenizer.tokenize_mixed`.
- Produces:
  - `align_words(utterances: list[str], tokens: list[str]) -> list[tuple[int, int, int]] | None`: `(utterance_index, start, end)` per stored token, or `None` if the text doesn't reproduce the tokens.
  - `piece_languages(offsets: list[tuple[int,int]], blank: list[bool], char_slots: np.ndarray) -> tuple[list[int], int]`: a slot per piece, plus the number of mixed pieces.
  - `subword_row(utterances, tokens, token_lang, first_language, second_language, tokenizer, is_entity=None) -> dict | None` with the keys `piece_ids: list[int]`, `pieces: list[str]`, `piece_lang: list[int]`, `mixed_pieces: int`.
  - `build_frame(frame: pd.DataFrame, texts: dict[int, list[str]], tokenizer, max_fail: float = 0.001) -> tuple[pd.DataFrame, dict]`.
    - Output columns: `row_id, first_language, second_language, quality, pieces, piece_ids, piece_lang, y_sw, y_len, y_dur, censored, dur_known, mixed_pieces`. The labels come from `anticipatory_labels(piece_lang)`.
    - Stats keys: `rows, failed, mixed_pieces, pieces`.
    - It raises `RuntimeError` if `failed / len(frame) > max_fail`.
  - `write_parquet(frame: pd.DataFrame, path: str, row_group_size: int = 8192) -> None`
  - `XLMR_VOCAB = 250_002`

- [ ] **Step 1: Write the failing tests `tests/test_subword.py`**

```python
"""Sub-word pieces take the language of the word they sit in."""
import importlib.util
import json
import os

import numpy as np
import pandas as pd
import pytest

from src.anticipation import L1, L2, NONE, token_slots
from src.segments import repair_spans
from src.subword import (align_words, build_frame, piece_languages, subword_row,
                         write_parquet)
from src.tokenizer import tokenize_mixed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(ROOT, "tests", "fixtures", "name_cases.json")


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("xlm-roberta-base",
                                                          local_files_only=True)
    except OSError:
        pytest.skip("xlm-roberta-base tokenizer not cached")


def _pieces(tok, text):
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    return tok.convert_ids_to_tokens(enc["input_ids"]), enc["offset_mapping"]


# --- alignment ---------------------------------------------------------------

def test_align_handles_repeated_substrings():
    assert align_words(["the theme the"], ["the", "theme", "the"]) == \
        [(0, 0, 3), (0, 4, 9), (0, 10, 13)]


def test_align_spans_multiple_utterances():
    assert align_words(["a b", "c"], ["a", "b", "c"]) == [(0, 0, 1), (0, 2, 3), (1, 0, 1)]


def test_align_rejects_tokens_the_text_does_not_produce():
    assert align_words(["hola amigo"], ["hola", "amiga"]) is None
    assert align_words(["hola"], ["hola", "extra"]) is None


def test_align_survives_emoji_zwj_and_combining_marks():
    for text in ["hola 😀 hello", "ok 👩‍💻 fine", "नमस्ते दोस्त yaar", "مرحبا friend"]:
        toks = tokenize_mixed(text)
        spans = align_words([text], toks)
        assert spans is not None, text
        assert [text[a:b] for _, a, b in spans] == toks


def test_empty_utterance_contributes_no_pieces(tok):
    assert align_words(["", "hola"], ["hola"]) == [(1, 0, 4)]
    row = subword_row(["", "hola"], ["hola"], ["Spanish"], "Spanish", "English", tok)
    assert row["piece_lang"] == [L1] * len(row["piece_ids"]) and row["piece_ids"]


# --- piece languages -----------------------------------------------------------

def test_mixed_piece_takes_the_majority_and_ties_go_to_l1():
    slots = np.array([L1, L1, L2, L1, L2, L2, L2], dtype=np.int8)
    langs, mixed = piece_languages([(0, 3), (3, 5), (4, 7)], [False] * 3, slots)
    assert langs == [L1, L1, L2] and mixed == 2          # (4,7) is all L2


def test_piece_over_no_word_characters_is_none():
    slots = np.array([-1, -1, L1], dtype=np.int8)
    assert piece_languages([(0, 2), (2, 3)], [False, False], slots) == ([NONE, L1], 0)


def test_bare_space_marker_piece_is_none():
    slots = np.array([L1, L1], dtype=np.int8)
    # XLM-R emits ('▁', offsets of the next char) before a CJK or fullwidth piece
    assert piece_languages([(0, 1), (0, 2)], [True, False], slots) == ([NONE, L1], 0)


def test_cjk_merged_pieces_take_the_character_language(tok):
    text = "昨天我去參觀了中關村論壇"
    toks = tokenize_mixed(text)
    row = subword_row([text], toks, ["Chinese"] * len(toks), "Chinese", "English", tok)
    pieces, offs = _pieces(tok, text)
    merged = [p for p, (a, b) in zip(pieces, offs) if b - a > 1]
    assert merged, "expected XLM-R to merge some characters"
    for p, lang in zip(pieces, row["piece_lang"]):
        assert lang == (NONE if p.replace("▁", "") == "" else L1), p


def test_punctuation_pieces_are_none_and_words_keep_their_language(tok):
    text = "Hola, ¿qué tal? I am fine."
    toks = tokenize_mixed(text)
    spanish = {"Hola", "qué", "tal"}
    labs = ["Spanish" if t in spanish else "English" if t.isalpha() else "other"
            for t in toks]
    row = subword_row([text], toks, labs, "Spanish", "English", tok)
    by_piece = dict(zip(_pieces(tok, text)[0], row["piece_lang"]))
    assert by_piece[","] == NONE and by_piece["?"] == NONE and by_piece["."] == NONE
    assert by_piece["▁Hola"] == L1 and by_piece["▁fine"] == L2


# --- names ----------------------------------------------------------------------

def _case_texts():
    spec = importlib.util.spec_from_file_location(
        "cns", os.path.join(ROOT, "scripts", "compare_name_strategies.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {cid: s for cid, _, _, s in mod.CASES}


def test_every_piece_of_a_name_takes_its_words_resolved_language(tok):
    from src.dispatch import tag_tokens
    texts = _case_texts()
    for c in json.load(open(FIXTURE)):
        text, toks, l1, l2 = texts[c["id"]], c["tokens"], c["l1"], c["l2"]
        labs = [lg for _, lg in tag_tokens(toks, l1, l2, ane_labels=c["ane"],
                                          is_entity=c["is_entity"])]
        row = subword_row([text], toks, labs, l1, l2, tok, is_entity=c["is_entity"])
        assert row is not None, c["id"]
        slots = token_slots(toks, labs, l1, l2, c["is_entity"])
        spans = align_words([text], toks)
        ent = repair_spans(toks, c["is_entity"], l1, l2)
        pieces, offs = _pieces(tok, text)
        for p, (s, e), lang in zip(pieces, offs, row["piece_lang"]):
            if p.replace("▁", "") == "":
                continue
            hit = [i for i, (_, a, b) in enumerate(spans) if a < e and s < b]
            names = [i for i in hit if ent[i] and slots[i] != NONE]
            if names:
                assert lang in {slots[i] for i in names}, (c["id"], text[s:e])
                assert lang != NONE, (c["id"], text[s:e])
        # a nonzero count here means a real mixed piece: inspect it before relaxing
        assert row["mixed_pieces"] == 0, c["id"]


# --- build frame ------------------------------------------------------------------

def _fixture_rows():
    texts, rows = {}, []
    sentences = ["Hola amigo, how are you?", "Yo quiero ir , but I can't today .",
                 "Vamos a la playa this weekend", "Me gusta mucho the new album !"]
    spanish = {"Hola", "amigo", "Yo", "quiero", "ir", "Vamos", "a", "la", "playa",
               "Me", "gusta", "mucho"}
    for rid in range(20):
        text = sentences[rid % len(sentences)]
        toks = tokenize_mixed(text)
        labs = ["Spanish" if t in spanish else "English" if any(ch.isalpha() for ch in t)
                else "other" for t in toks]
        texts[rid] = [text]
        rows.append(dict(row_id=rid, first_language="Spanish", second_language="English",
                         quality="provisional", tokens=toks, token_lang=labs,
                         is_entity=[False] * len(toks)))
    return pd.DataFrame(rows), texts


def test_build_frame_counts_and_skips_misaligned_rows(tok):
    frame, texts = _fixture_rows()
    texts[7] = ["completely different text"]
    out, stats = build_frame(frame, texts, tok, max_fail=0.1)
    assert stats["rows"] == 19 and stats["failed"] == 1
    assert 7 not in set(out.row_id)
    assert stats["pieces"] == sum(len(p) for p in out.piece_ids)
    with pytest.raises(RuntimeError):
        build_frame(frame, texts, tok)                    # default max_fail 0.1%


def test_stored_labels_match_the_label_function(tok):
    from src.anticipation import anticipatory_labels
    frame, texts = _fixture_rows()
    out, _ = build_frame(frame, texts, tok)
    for r in out.itertuples():
        lab = anticipatory_labels(np.asarray(r.piece_lang, dtype=np.int8))
        for k in ("y_sw", "y_len", "y_dur", "censored", "dur_known"):
            assert list(getattr(r, k)) == lab[k].tolist(), k
        assert len(r.pieces) == len(r.piece_ids) == len(r.y_sw)


def test_rows_too_short_to_label_are_kept_with_empty_labels(tok):
    frame = pd.DataFrame([dict(row_id=0, first_language="Spanish", second_language="English",
                               quality="provisional", tokens=["Hola", "!"],
                               token_lang=["Spanish", "other"], is_entity=[False, False])])
    out, stats = build_frame(frame, {0: ["Hola!"]}, tok)
    assert stats["rows"] == 1 and set(out.y_sw.iloc[0]) == {-1}


def test_write_parquet_round_trips_types(tok, tmp_path):
    import pyarrow.parquet as pq
    frame, texts = _fixture_rows()
    out, _ = build_frame(frame, texts, tok)
    path = str(tmp_path / "subword.parquet")
    write_parquet(out, path)
    schema = pq.read_schema(path)
    assert str(schema.field("piece_ids").type) == "list<element: int32>"
    assert str(schema.field("piece_lang").type) == "list<element: int8>"
    assert str(schema.field("pieces").type) == "list<element: string>"
    assert str(schema.field("y_len").type) == "list<element: int32>"
    assert str(schema.field("dur_known").type) == "list<element: bool>"
    back = pd.read_parquet(path)
    assert list(back.piece_lang.iloc[0]) == list(out.piece_lang.iloc[0])
    assert list(back.y_sw.iloc[0]) == list(out.y_sw.iloc[0])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./.venv/bin/python -m pytest tests/test_subword.py -q`
Expected: collection ERROR, `ModuleNotFoundError: No module named 'src.subword'`.

- [ ] **Step 3: Implement `src/subword.py`**

```python
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

from .anticipation import L1, L2, NONE, anticipatory_labels, token_slots
from .tokenizer import tokenize_mixed

XLMR_VOCAB = 250_002
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
    for (u, s, e), slot in zip(spans, slots):
        if slot != NONE:
            char_slots[u][s:e] = slot
    ids, pieces, langs, mixed = [], [], [], 0
    for u, text in enumerate(utterances):
        if not text:
            continue
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        strs = tokenizer.convert_ids_to_tokens(enc["input_ids"])
        blank = [p.replace("▁", "") == "" for p in strs]
        lg, m = piece_languages(enc["offset_mapping"], blank, char_slots[u])
        ids.extend(enc["input_ids"])
        pieces.extend(strs)
        langs.extend(lg)
        mixed += m
    return dict(piece_ids=ids, pieces=pieces, piece_lang=langs, mixed_pieces=mixed)


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
        ("dur_known", pa.list_(pa.bool_())),
        ("mixed_pieces", pa.int32()),
    ])
    table = pa.Table.from_pandas(frame[schema.names], schema=schema, preserve_index=False)
    pq.write_table(table, path, compression="snappy", row_group_size=row_group_size)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `./.venv/bin/python -m pytest tests/test_subword.py -q`
Expected: 15 passed.

If `test_every_piece_of_a_name...` fails on `mixed_pieces`, print the piece and its overlapping tokens before changing anything. A real mixed piece belongs in the report, not in a relaxed assertion.

- [ ] **Step 5: Run the full suite, then commit**

Run: `./.venv/bin/python -m pytest tests/ -q`. Expected: 229 passed.

```bash
git add src/subword.py tests/test_subword.py
git -c user.name="Cristae-25" -c user.email="teh.c@northeastern.edu" commit -m "Map final word labels onto XLM-R pieces

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `scripts/build_subword.py` and `data/subword.parquet`

**Files:**
- Create: `scripts/build_subword.py`

**Interfaces:**
- Consumes: `build_frame`, `write_parquet` (Task 2); `run_tagging.load_corpus()` (returns the 234,172-row CSV DataFrame whose positional index is `row_id`); `segments.parse_utterances`.
- Produces: `data/subword.parquet`, with the columns from Task 2, rows in `np.random.default_rng(0).permutation` order.

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python
"""Write data/subword.parquet: XLM-R pieces with each piece's word language.

    ./.venv/bin/python scripts/build_subword.py              # full corpus, ~10 min CPU
    ./.venv/bin/python scripts/build_subword.py --limit 2000 --dst /tmp/sw.parquet

Joins the original SwitchLingua text (the cached CSVs, positional index =
row_id) with the final per-word labels in data/tagged.parquet, then writes the
rows in shuffled order (seed 0) so streaming it is not a curriculum by
language. The row_id % 10 split assignment travels with each row.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from run_tagging import load_corpus  # noqa: E402
from src.segments import parse_utterances  # noqa: E402
from src.subword import build_frame, write_parquet  # noqa: E402

COLUMNS = ["row_id", "first_language", "second_language", "quality",
           "tokens", "token_lang", "is_entity"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/tagged.parquet")
    ap.add_argument("--dst", default="data/subword.parquet")
    ap.add_argument("--tokenizer", default="xlm-roberta-base")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="random sample of rows (0 = all)")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    t0 = time.time()
    frame = pd.read_parquet(args.src, columns=COLUMNS)
    if args.limit:
        frame = frame.sample(n=args.limit, random_state=args.seed)
    corpus = load_corpus()
    cells = corpus["data_generation_result"]
    texts = {int(r): parse_utterances(cells.iloc[int(r)]) for r in frame.row_id}
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    out, stats = build_frame(frame, texts, tok)
    out = out.iloc[np.random.default_rng(args.seed).permutation(len(out))]
    write_parquet(out, args.dst)

    print(f"{args.dst}: {stats['rows']:,} rows, {stats['pieces']:,} pieces "
          f"in {(time.time() - t0) / 60:.1f} min")
    print(f"alignment failures {stats['failed']:,} | mixed-language pieces "
          f"{stats['mixed_pieces']:,} ({stats['mixed_pieces'] / max(stats['pieces'], 1):.4%})")
    lang = np.concatenate([np.asarray(x, dtype=np.int8) for x in out.piece_lang])
    print(f"piece slots: L1 {np.mean(lang == 0):.1%} | L2 {np.mean(lang == 1):.1%} "
          f"| NONE {np.mean(lang == 2):.1%}")
    y_sw = np.concatenate([np.asarray(x) for x in out.y_sw])
    y_dur = np.concatenate([np.asarray(x) for x in out.y_dur])
    known = np.concatenate([np.asarray(x) for x in out.dur_known])
    print(f"scored pieces {int((y_sw >= 0).sum()):,} | switches {int((y_sw == 1).sum()):,} "
          f"({(y_sw == 1).sum() / max((y_sw >= 0).sum(), 1):.2%})")
    counts = np.bincount(y_dur[known], minlength=3)
    print("duration classes (known): " + " | ".join(
        f"{lab} {c:,} ({c / max(counts.sum(), 1):.1%})"
        for lab, c in zip(["small", "medium", "large"], counts)))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke run**

Run: `./.venv/bin/python scripts/build_subword.py --limit 2000 --dst /private/tmp/claude-501/-Users-alphastar-Northeastern-GitHub-code-switch/d4912f19-d032-44b5-960b-e703d5ca4c32/scratchpad/sw_smoke.parquet`
Expected:
- about 2,000 rows
- `alignment failures 0`
- mixed-language pieces under 0.1%
- NONE share between 5% and 30%
- duration classes printed for all three

- [ ] **Step 3: Full build**

Run: `./.venv/bin/python scripts/build_subword.py 2>&1 | tee data/logs/build_subword.log`
Expected:
- 234,172 rows (minus any alignment failures, which must be 0 or close to it)
- roughly 28–35 M pieces
- a runtime under 20 min

Keep the log; the report quotes the counts.

- [ ] **Step 4: Commit**

```bash
git add scripts/build_subword.py
git -c user.name="Cristae-25" -c user.email="teh.c@northeastern.edu" commit -m "Add build_subword.py: token-level labels on XLM-R pieces

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Document the token-level labels

**Files:**
- Modify: `CLAUDE.md`, `TIMELINE.md`, `groupUpdate1.md`
- Regenerate: `kaggle/label_switchlingua.ipynb` (`scripts/build_kaggle_notebook.py` embeds `src/`, which gained `subword.py`)

- [ ] **Step 1: `CLAUDE.md`**
  - Under **Commands**, add:
    ```
    ./.venv/bin/python scripts/build_subword.py                          # -> data/subword.parquet: XLM-R pieces + per-piece language, y_sw, y_len, y_dur
    ```
  - Under **Anticipatory task**, add a bullet: "Duration is stored as the integer `y_len`, with the classes Small 1–2 / Medium 3–6 / Large 7+ (`anticipation.dur_class`). A run cut off by the end of the text counts only once it is 7+ (`dur_known`). Slot `NONE` (punctuation, digits, a bare `▁`) is transparent. `src/subword.py` gives each XLM-R piece the language of its word via character offsets on the original text. The models still train on word units; sub-word training is Week 5."

- [ ] **Step 2: `TIMELINE.md`**

Change Week 4 to: "🟡 Word- and sub-word-level labels done (XLM-R pieces, `data/subword.parquet`, 3-class duration); training on sub-words is Week 5." Add an "Update" bullet with the counts from `data/logs/build_subword.log`: pieces, switch rate, and the class shares.

- [ ] **Step 3: `groupUpdate1.md`**
  - **§4.4:** add a lead sentence: "These 7-bucket numbers are historical. Duration now uses the 3 required classes (Small 1–2, Medium 3–6, Large 7+), and results in those classes come with the next training round."
  - **§5.1, "Planned fix":** add a paragraph with the token-level labels as built:
    - the tokenizer comparison table from the spec
    - the 24–51% of CJK pieces that merge characters
    - the counts from `data/logs/build_subword.log`
    - the note that CJK run lengths are still not comparable across scripts

- [ ] **Step 4: Rebuild the notebook, run the tests, commit**

```bash
./.venv/bin/python scripts/build_kaggle_notebook.py
./.venv/bin/python -m pytest tests/ -q
git add CLAUDE.md TIMELINE.md groupUpdate1.md kaggle/label_switchlingua.ipynb
git -c user.name="Cristae-25" -c user.email="teh.c@northeastern.edu" commit -m "Document the sub-word token labels

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
Expected: 229 passed.

---

## Deferred to the model work (Week 5, not this plan)

- **Streaming:** sub-word streaming (`unit="subword"` windows of 384/288, a compact piece vocabulary).
- **Language input:** the model gets no language input on sub-words, because a word's label depends on pieces not yet read; plus `--no-lang` for a fair word-model comparison.
- **Scripts:** `--unit` flags in the train, baseline and pair-eval scripts.
- **Metrics:** 3-class duration metrics (macro-F1, confusion matrix).
- **Runs and write-up:** all training runs and the results write-up.
