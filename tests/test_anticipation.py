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


# --- word stream ------------------------------------------------------------

def test_punctuation_is_transparent_not_a_switch():
    toks = ["hola", ",", "hello"]
    langs = ["Spanish", "other", "English"]
    words, slots = word_stream(toks, langs, "Spanish", "English")
    assert words == ["hola", "hello"]
    assert list(slots) == [0, 1]


def test_run_lengths_match_the_descriptive_pipeline():
    from src.segments import absorb_other, l2_word_runs
    toks = ["uno", "dos", "one", "two", "three", ",", "tres"]
    langs = ["Spanish"] * 2 + ["English"] * 3 + ["other", "Spanish"]
    _, slots = word_stream(toks, langs, "Spanish", "English")
    starts, lengths = run_starts_and_lengths(slots)
    l2_here = [int(n) for s, n in zip(starts, lengths) if slots[s] == 1]
    assert l2_here == l2_word_runs(absorb_other(list(zip(toks, langs))), "English")


# --- streaming --------------------------------------------------------------

def test_splits_are_disjoint_and_cover_everything():
    for rid in range(50):
        assert sum(in_split(rid, s) for s in ("train", "val", "test")) == 1


def test_continuation_windows_mask_their_warm_up():
    ex = dict(row_id=0, words=[f"w{i}" for i in range(30)],
              langs=np.array([i % 2 for i in range(30)], dtype=np.int8),
              first_language="Spanish", second_language="English", quality="exact")
    ex.update(anticipatory_labels(ex["langs"]))
    ds = SwitchWindowDataset("unused", {"<pad>": 0, "<unk>": 1}, window=16, stride=12)
    wins = list(ds._windows(ex))
    assert len(wins) > 1
    assert wins[0]["sw_mask"][:4].all()          # first window scores from 0
    assert not wins[1]["sw_mask"][:4].any()      # continuation warm-up masked


def test_every_position_is_scored_exactly_once_across_windows():
    n = 100
    ex = dict(row_id=0, words=[f"w{i}" for i in range(n)],
              langs=np.array([i % 3 == 0 for i in range(n)], dtype=np.int8),
              first_language="Spanish", second_language="English", quality="exact")
    ex.update(anticipatory_labels(ex["langs"]))
    ds = SwitchWindowDataset("unused", {"<pad>": 0, "<unk>": 1}, window=32, stride=24)
    seen = []
    for i, w in enumerate(ds._windows(ex)):
        offset = 0 if i == 0 else 24 * i
        seen.extend(offset + j for j in np.flatnonzero(w["sw_mask"]))
    assert seen == sorted(seen)
    assert len(seen) == len(set(seen)) == n - 1   # every position but the last


def test_padded_positions_never_look_like_a_real_language():
    from src.streaming import PAD_LANG
    ex = dict(row_id=0, words=["a", "b", "c"],
              langs=np.array([0, 1, 0], dtype=np.int8),
              first_language="Spanish", second_language="English", quality="exact")
    ex.update(anticipatory_labels(ex["langs"]))
    ds = SwitchWindowDataset("unused", {"<pad>": 0, "<unk>": 1}, window=8, stride=8)
    w = next(iter(ds._windows(ex)))
    assert (w["lang_ids"][3:] == PAD_LANG).all()
    assert not w["sw_mask"][3:].any()


# --- causality --------------------------------------------------------------

def test_model_cannot_see_the_future():
    torch = pytest.importorskip("torch")
    from src.causal_model import CausalSwitchModel, check_causality
    m = CausalSwitchModel(500, d_model=32, n_layers=2, n_heads=2, d_ff=64)
    report = check_causality(m, 500, seq_len=24)
    assert report["causal"], report
    assert report["max_abs_logit_drift"] == 0.0


def test_bidirectional_model_would_fail_the_same_check():
    """The causality test has teeth: an uncausal model must fail it."""
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    from src.causal_model import CausalSwitchModel, check_causality

    m = CausalSwitchModel(500, d_model=32, n_layers=2, n_heads=2, d_ff=64)
    original = m.encoder.forward

    def bidirectional(x, mask=None, is_causal=None, **kw):
        return original(x, mask=None, is_causal=False)

    m.encoder.forward = bidirectional
    assert not check_causality(m, 500, seq_len=24)["causal"]


# --- metrics ----------------------------------------------------------------

def test_never_predicting_a_switch_scores_zero_f1_at_high_accuracy():
    y = np.array([0] * 95 + [1] * 5)
    s = prf(y, np.zeros_like(y))
    assert s["accuracy"] == 0.95 and s["f1"] == 0.0


# --- sub-word streaming -------------------------------------------------------

from src.anticipation import subword_example  # noqa: E402
from src.streaming import build_subword_vocab, encode_pieces  # noqa: E402

_SW_VOCAB = {"<pad>": 0, "<unk>": 1, "<bos>": 2, **{str(i): i for i in range(3, 60)}}


def _sw_row(ids, langs, row_id=0):
    return dict(row_id=row_id, piece_ids=list(ids), piece_lang=list(langs),
                first_language="Spanish", second_language="English", quality="provisional")


def test_subword_example_labels_follow_piece_lang():
    ex = subword_example(_sw_row([5, 6, 7, 8, 9, 10, 11], [0, NONE, 0, 1, NONE, 1, 0]))
    assert list(ex["y_sw"]) == [0, -1, 1, 0, -1, 1, -1]
    assert ex["ids"].dtype == np.int64 and ex["langs"].dtype == np.int8


def test_subword_example_needs_two_language_pieces():
    assert subword_example(_sw_row([5, 6, 7], [NONE, 0, NONE])) is None


def test_encode_pieces_maps_xlmr_ids_through_the_compact_vocab():
    vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "17": 3}
    assert list(encode_pieces([17, 99], vocab)) == [3, 1]


def test_subword_windows_score_only_language_pieces():
    ex = subword_example(_sw_row([5, 6, 7, 8, 9], [0, NONE, 1, 1, 0]))
    ds = SwitchWindowDataset("unused", _SW_VOCAB, window=8, stride=8, unit="subword")
    w = next(iter(ds._windows(ex)))
    assert list(w["sw_mask"][:5]) == [True, False, True, True, False]
    assert list(w["word_ids"][:5]) == [5, 6, 7, 8, 9]


def test_long_subword_row_scores_every_language_piece_once():
    rng = np.random.default_rng(1)
    n = 1000
    langs = rng.choice([0, 1, NONE], size=n, p=[0.6, 0.25, 0.15]).astype(np.int8)
    ex = subword_example(_sw_row(rng.integers(3, 60, n), langs))
    ds = SwitchWindowDataset("unused", _SW_VOCAB, window=384, stride=288, unit="subword")
    seen = []
    for i, w in enumerate(ds._windows(ex)):
        seen.extend(288 * i + j for j in np.flatnonzero(w["sw_mask"]))
    assert seen == sorted(set(seen))
    assert seen == list(np.flatnonzero(ex["y_sw"] >= 0))


def test_subword_window_inputs_depend_only_on_earlier_pieces():
    rng = np.random.default_rng(2)
    ids = rng.integers(3, 60, 40)
    langs = rng.choice([0, 1, NONE], size=40).astype(np.int8)
    ds = SwitchWindowDataset("unused", _SW_VOCAB, window=64, stride=48, unit="subword")
    first = next(iter(ds._windows(subword_example(_sw_row(ids, langs)))))
    for t in range(39):
        ids2, langs2 = ids.copy(), langs.copy()
        ids2[t + 1:] = rng.integers(3, 60, 39 - t)
        langs2[t + 1:] = rng.choice([0, 1], size=39 - t)
        ex2 = subword_example(_sw_row(ids2, langs2))
        if ex2 is None:
            continue
        w2 = next(iter(ds._windows(ex2)))
        assert (first["word_ids"][:t + 1] == w2["word_ids"][:t + 1]).all()


def test_build_subword_vocab_uses_the_train_split_only(tmp_path):
    import pandas as pd
    rows = [dict(row_id=0, first_language="Spanish", second_language="English",
                 quality="provisional", piece_ids=[5, 5, 6], piece_lang=[0, 0, 1]),
            dict(row_id=8, first_language="Spanish", second_language="English",
                 quality="provisional", piece_ids=[7, 7, 7], piece_lang=[0, 1, 0])]  # row 8 is val
    path = str(tmp_path / "sw.parquet")
    pd.DataFrame(rows).to_parquet(path, index=False)
    vocab = build_subword_vocab(path, min_count=2)
    assert "5" in vocab and "6" not in vocab and "7" not in vocab
    assert vocab["<pad>"] == 0 and vocab["<unk>"] == 1
