"""LinCE adapter (src/lince.py): gold tags -> project tokens -> anticipation labels."""
import os

import numpy as np
import pandas as pd
import pytest

from src.anticipation import anticipatory_labels, token_slots
from src.lince import describe_row, map_tag, row_id, to_tagged, tweet_tokens
from src.segments import NE


def test_tag_mapping():
    assert map_tag("lang1", "hello", "Spanish") == "English"
    assert map_tag("lang2", "hola", "Spanish") == "Spanish"
    for tag in ("ne", "mixed", "ambiguous", "fw", "unk", "other"):
        assert map_tag(tag, "Messi", "Spanish") == NE        # lettered, language-less
    assert map_tag("other", "!!", "Spanish") == "other"       # no letters: punctuation
    assert map_tag("lang1", "😂", "Spanish") == "other"


def test_tokens_inherit_their_word_tag():
    words = ["@_easanti", "Loved", "the", "chai-tea", "#yum", "pero", "muy", "caro", ";)"]
    lid = ["other", "lang1", "lang1", "mixed", "other", "lang2", "lang2", "lang2", "other"]
    text, tokens, langs = tweet_tokens(words, lid, "Spanish")
    assert text == " ".join(words)
    got = dict(zip(tokens, langs))
    assert got["@"] == "other" and got["easanti"] == NE      # handle: transparent
    assert got["chai"] == NE and got["tea"] == NE            # "mixed" word
    assert got["Loved"] == "English" and got["pero"] == "Spanish"
    assert got["yum"] == NE


def test_labels_from_gold_tags():
    # EN EN | ES ES ES | EN: a switch into Spanish (3 words), then one back (1 word, cut off)
    words = ["I", "love", "pero", "no", "puedo", "dude"]
    lid = ["lang1", "lang1", "lang2", "lang2", "lang2", "lang1"]
    _, tokens, langs = tweet_tokens(words, lid, "Spanish")
    slots = np.asarray(token_slots(tokens, langs, "Spanish", "English"), dtype=np.int8)
    lab = anticipatory_labels(slots)
    assert lab["y_sw"].tolist() == [0, 1, 0, 0, 1, -1]
    assert lab["y_len"][1] == 3 and lab["dur_known"][1]
    assert lab["censored"][4] and not lab["dur_known"][4]   # cut-off run of 1 is uncertain


def test_names_neither_break_nor_extend_a_run():
    words = ["vamos", "con", "Messi", "y", "Neymar"]
    lid = ["lang2", "lang2", "ne", "lang2", "ne"]
    _, tokens, langs = tweet_tokens(words, lid, "Spanish")
    slots = token_slots(tokens, langs, "Spanish", "English")
    assert slots == [0, 0, 2, 0, 2]                           # NE -> NONE (transparent)


def test_row_ids_round_trip():
    for pair in ("spaeng", "hineng"):
        for split in ("validation", "train"):
            assert describe_row(row_id(pair, split, 1234)) == (pair, split, 1234)
    assert row_id("spaeng", "validation", 0) > 234_172


def test_to_tagged_frame():
    df = pd.DataFrame(dict(idx=[0, 1], words=[["hola", "amigo"], []], lid=[["lang2", "lang2"], []]))
    frame, texts = to_tagged(df, "spaeng", "validation")
    assert len(frame) == 1 and frame.first_language.iloc[0] == "Spanish"
    assert texts[frame.row_id.iloc[0]] == ["hola amigo"]


@pytest.mark.skipif(not os.path.exists("data/public/lid_spaeng_validation.parquet"),
                    reason="needs the LinCE validation file")
def test_real_validation_aligns():
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    from src.lince import load
    from src.subword import build_frame
    df = load("spaeng", "validation").iloc[:300]
    frame, texts = to_tagged(df, "spaeng", "validation")
    pieces, stats = build_frame(frame, texts, AutoTokenizer.from_pretrained("xlm-roberta-base"),
                                max_fail=0.001)
    assert stats["failed"] == 0 and len(pieces) == len(frame)
    assert sum((np.asarray(y) == 1).sum() for y in pieces.y_sw) > 0


def _piece_frame(row_ids, first_language="Spanish", n=3):
    k = len(row_ids)
    return pd.DataFrame(dict(
        row_id=list(row_ids), first_language=[first_language] * k, second_language=["English"] * k,
        quality=["gold"] * k, pieces=[["▁a"] * n] * k, piece_ids=[[5] * n] * k, piece_lang=[[0] * n] * k,
        y_sw=[[0] * n] * k, y_len=[[1] * n] * k, y_dur=[[0] * n] * k, censored=[[False] * n] * k,
        dur_known=[[True] * n] * k, word_end=[[True] * n] * k, piece_word=[list(range(n))] * k,
        piece_utt=[[0] * n] * k, mixed_pieces=[0] * k))


def test_mix_repeats_only_lince_training_rows_and_never_adds_validation(tmp_path):
    import pyarrow.parquet as pq

    from src.lince import mix
    from src.subword import write_parquet
    sl = tmp_path / "sl.parquet"
    write_parquet(_piece_frame(range(20), "Korean"), str(sl))
    tr_ids = [row_id("spaeng", "train", i) for i in range(10)]          # idx % 10: 0..9
    lp = tmp_path / "spaeng_train.parquet"
    write_parquet(_piece_frame(tr_ids), str(lp))
    out, stats = mix(str(sl), "unused", 3, out=str(tmp_path / "mix.parquet"), pairs=("spaeng",),
                     lince_paths={"spaeng": str(lp)})
    rid = pq.read_table(out, columns=["row_id"]).column(0).to_numpy()
    li = rid[rid >= 10_000_000]
    counts = {int(r): int((li == r).sum()) for r in tr_ids}
    assert all(counts[r] == 3 for r in tr_ids if r % 10 < 8)            # training split: x weight
    assert all(counts[r] == 1 for r in tr_ids if r % 10 >= 8)           # val / test: once
    assert (rid < 10_000_000).sum() == 20 and stats["rows"] == len(rid) == 20 + 8 * 3 + 2
    assert not any(describe_row(int(r))[1] == "validation" for r in li)
    assert 0 < stats["lince_share_of_train_pieces"] < 1
    again, _ = mix(str(sl), "unused", 3, out=str(tmp_path / "mix2.parquet"), pairs=("spaeng",),
                   lince_paths={"spaeng": str(lp)})
    assert (pq.read_table(again, columns=["row_id"]).column(0).to_numpy() == rid).all()   # seeded
    assert not (rid[:-1] <= rid[1:]).all()                                                # shuffled


def test_mix_refuses_lince_validation(tmp_path):
    from src.lince import mix
    from src.subword import write_parquet
    sl, lp = tmp_path / "sl.parquet", tmp_path / "val.parquet"
    write_parquet(_piece_frame(range(4)), str(sl))
    write_parquet(_piece_frame([row_id("spaeng", "validation", i) for i in range(4)]), str(lp))
    with pytest.raises(ValueError):
        mix(str(sl), "unused", 1, out=str(tmp_path / "m.parquet"), pairs=("spaeng",),
            lince_paths={"spaeng": str(lp)})
    with pytest.raises(ValueError):
        mix(str(sl), "unused", 0, out=str(tmp_path / "m.parquet"))
