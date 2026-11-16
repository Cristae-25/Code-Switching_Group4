"""Sub-word pieces take the language of the word they sit in."""
import importlib.util
import json
import os

import numpy as np
import pandas as pd
import pytest

from src.anticipation import L1, L2, NONE, token_slots
from src.segments import repair_spans
from src.subword import (align_words, build_frame, is_blank, piece_languages, subword_row,
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
        "cns", os.path.join(ROOT, "scripts", "labelling", "compare_name_strategies.py"))
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


# --- other tokenizers: RoBERTa (byte-level BPE) and mBERT (WordPiece) --------

def _load(name):
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(name, local_files_only=True)
    except OSError:
        pytest.skip(f"{name} tokenizer not cached")


@pytest.mark.parametrize("name", ["roberta-base", "bert-base-multilingual-cased"])
def test_other_tokenizers_label_every_piece(name):
    tk = _load(name)
    text = "Hola, ¿qué tal? I'm fine. Привет friend"
    toks = tokenize_mixed(text)
    labs = ["Spanish" if t in {"Hola", "qué", "tal", "Привет"} else "English" if t.isalpha() else "other"
            for t in toks]
    row = subword_row([text], toks, labs, "Spanish", "English", tk)
    assert row is not None and row["mixed_pieces"] == 0
    pieces, offs = _pieces(tk, text)
    spans = align_words([text], toks)
    char_lang = {}                                   # expected: the language of the word it sits in
    for (_, a, b), slot in zip(spans, token_slots(toks, labs, "Spanish", "English")):
        for c in range(a, b):
            char_lang[c] = None if slot == NONE else slot
    for p, (s, e), lang in zip(pieces, offs, row["piece_lang"]):
        got = {char_lang.get(c) for c in range(s, e)} - {None}
        assert lang == (got.pop() if got else NONE), (name, p, text[s:e])


def test_roberta_space_marker_piece_is_none():
    assert piece_languages([(0, 1)], [True], np.array([L1], dtype=np.int8)) == ([NONE], 0)
    from src.subword import is_blank
    assert is_blank("Ġ") and is_blank("▁") and is_blank("Ċ") and not is_blank("Ġfine") and not is_blank("##la")


# --- word ends: the positions shared by every tokenizer ------------------------

@pytest.mark.parametrize("name", ["xlm-roberta-base", "roberta-base", "bert-base-multilingual-cased"])
def test_word_end_marks_the_last_piece_of_each_non_cjk_word(name):
    tk = _load(name)
    text = "Hola amigo, 你好 friend"
    toks = tokenize_mixed(text)                      # Hola amigo , 你 好 friend
    labs = ["Spanish", "Spanish", "other", "Chinese", "Chinese", "English"]
    row = subword_row([text], toks, labs, "Spanish", "English", tk)
    pieces, offs = _pieces(tk, text)
    ends = [text[s:e] for (s, e), f in zip(offs, row["word_end"]) if f]
    assert [x.strip() for x in ends] == ["o", "o", "d"] or len(ends) == 3, (name, ends)
    assert sum(row["word_end"]) == 3                  # Hola, amigo, friend - not the CJK characters
    last = [max(i for i, (s, e) in enumerate(offs) if s < b and a < e)
            for a, b in [(0, 4), (5, 10), (17, 23)]]
    assert [i for i, f in enumerate(row["word_end"]) if f] == last


@pytest.mark.parametrize("name", ["xlm-roberta-base", "roberta-base", "bert-base-multilingual-cased"])
def test_word_and_piece_streams_score_the_same_shared_positions(name):
    from src.anticipation import row_example, subword_example
    tk = _load(name)
    text = "Ayer vi a Leo , and it was amazing 你好 honestly bueno"
    toks = tokenize_mixed(text)
    labs = ["Spanish"] * 4 + ["other"] + ["English"] * 4 + ["Chinese", "Chinese", "English", "Spanish"]
    base = dict(row_id=0, first_language="Spanish", second_language="English", quality="p")
    word = row_example(dict(base, tokens=toks, token_lang=labs, is_entity=[False] * len(toks)))
    sw = subword_row([text], toks, labs, "Spanish", "English", tk)
    piece = subword_example(dict(base, piece_ids=sw["piece_ids"], piece_lang=sw["piece_lang"],
                                 word_end=sw["word_end"]))
    w_shared = word["shared"] & (word["y_sw"] >= 0)
    p_shared = piece["shared"] & (piece["y_sw"] >= 0)
    assert w_shared.sum() == p_shared.sum() == 9            # 10 non-CJK words, last one unlabelled
    assert list(word["y_sw"][w_shared]) == list(piece["y_sw"][p_shared])   # same questions, same answers


# --- word and utterance index per piece (for the tuning flags) ---------------

def test_pieces_record_their_word_and_utterance(tok):
    utts = ["Hola amigo", "你好 friend"]
    toks = [t for u in utts for t in tokenize_mixed(u)]          # Hola amigo | 你 好 friend
    labs = ["Spanish", "Spanish", "Chinese", "Chinese", "English"]
    row = subword_row(utts, toks, labs, "Spanish", "English", tok)
    pieces = [p for u in utts for p in tok.convert_ids_to_tokens(
        tok(u, add_special_tokens=False)["input_ids"])]
    assert len(row["piece_word"]) == len(row["piece_utt"]) == len(pieces)
    words = [w for w in row["piece_word"] if w >= 0]
    assert words == sorted(words) and set(words) <= set(range(len(toks)))
    assert row["piece_word"][-1] == 4                             # last piece belongs to "friend"
    assert list(row["piece_utt"]) == sorted(row["piece_utt"]) and row["piece_utt"][-1] == 1
    for p, w in zip(pieces, row["piece_word"]):
        if p.replace("▁", "") == "":
            assert w == -1                                        # a bare space marker has no word


# --- prefix consistency: does the prefix alone reproduce each piece? -----------

class _ShiftyTokenizer:
    """Segments "abc" as ab|c, but its prefix "ab" as a|b: a leak to flag."""
    table = {"abc": [("ab", 0, 2), ("c", 2, 3)], "ab": [("a", 0, 1), ("b", 1, 2)],
             "a": [("a", 0, 1)]}
    vocab = {"a": 1, "b": 2, "ab": 3, "c": 4}

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        if isinstance(text, list):
            return {"input_ids": [self(t)["input_ids"] for t in text]}
        segs = self.table[text]
        out = {"input_ids": [self.vocab[p] for p, _, _ in segs]}
        if return_offsets_mapping:
            out["offset_mapping"] = [(s, e) for _, s, e in segs]
        return out

    def convert_ids_to_tokens(self, ids):
        inv = {v: k for k, v in self.vocab.items()}
        return [inv[i] for i in ids]


def test_a_piece_the_prefix_would_not_produce_is_flagged():
    from src.subword import prefix_consistency
    ids, last_ok, all_ok = prefix_consistency("abc", _ShiftyTokenizer())
    assert ids == [3, 4]
    assert last_ok == [False, True]          # "ab" alone ends in "b", not "ab"
    assert all_ok == [False, True]           # the last prefix is the whole text


@pytest.mark.parametrize("text", ["東京都に行く", "我喜欢music很多", "meetings를 했어",
                                  "Это очень интересно", "I love 香港人"])
def test_xlmr_pieces_are_prefix_consistent(tok, text):
    """Unigram SentencePiece decodes by Viterbi, whose best path through a piece
    boundary is also the best path to it, so the prefix reproduces the pieces."""
    from src.subword import prefix_consistency
    _, last_ok, all_ok = prefix_consistency(text, tok)
    assert all(last_ok) and all(all_ok)


def test_blank_pieces_are_never_scored(tok):
    """A bare "▁" carries the next character's offsets: it must be NONE, i.e. no label."""
    from src.anticipation import anticipatory_labels
    enc = tok("東京都に行く", add_special_tokens=False)
    strs = tok.convert_ids_to_tokens(enc["input_ids"])
    assert strs[0] == "▁" and is_blank(strs[0])
    langs = np.array([NONE, 0, 0], dtype=np.int8)
    assert anticipatory_labels(langs)["y_sw"][0] == -1
