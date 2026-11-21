"""Replay dashboard back end (src/replay.py): positions line up with training's windows."""
import os

import numpy as np
import pytest

from src.replay import (LivePredictor, Run, SavedPreds, display_pieces, replay_row,
                        row_windows, scored)

L1, L2, NONE = 0, 1, 2


def _row(n_words=12, row_id=19):
    """A hand-built sub-word row: L1 words, a 3-word L2 run, L1 again, with punctuation."""
    langs = [L1] * 4 + [NONE] + [L2] * 3 + [L1] * (n_words - 7)
    pieces = [("▁w%d" % i if lg != NONE else ",") for i, lg in enumerate(langs)]
    n = len(langs)
    return dict(row_id=row_id, first_language="Spanish", second_language="English",
                quality="provisional", pieces=pieces, piece_ids=list(range(10, 10 + n)),
                piece_lang=langs, word_end=[True] * n, piece_word=list(range(n)),
                piece_utt=[0] * n)


def _run(tmp_path, window=8, stride=6, decide_at="word"):
    return Run("toy", "toy", dict(model="xlmr", path=str(tmp_path / "x.parquet"),
                                  window=window, stride=stride, prev_lang=True,
                                  decide_at=decide_at), tau=0.5, root=str(tmp_path))


def _saved_dir(tmp_path, run, rows):
    """Write predictions the way train_anticipation.evaluate() orders them."""
    d = tmp_path / "data" / "preds" / run.tag
    d.mkdir(parents=True)
    out = {k: [] for k in ("row", "pos", "logit", "pair", "drow", "dprob")}
    for r in rows:
        _, w = row_windows(run, r)
        sw, du = scored(w)
        out["row"] += [r["row_id"]] * len(sw)
        out["pos"] += list(sw)
        out["logit"] += list(np.linspace(-3, 3, len(sw)))
        out["pair"] += [13] * len(sw)                 # Spanish->English
        out["drow"] += [r["row_id"]] * len(du)
        out["dprob"] += [[0.2, 0.5, 0.3]] * len(du)
    for k, v in out.items():
        np.save(d / f"test_{k}.npy", np.asarray(v))
    return str(d)


def test_windows_score_each_position_once_and_skip_the_last(tmp_path):
    run = _run(tmp_path)
    ex, w = row_windows(run, _row(n_words=20))
    assert len(w) > 1                                  # long row: several windows
    sw, du = scored(w)
    assert len(sw) == len(set(sw.tolist()))            # overlap is context only
    assert (np.diff(sw) > 0).all()
    labelled = np.flatnonzero(ex["y_sw"] >= 0)
    assert np.array_equal(sw, labelled)
    assert len(ex["ids"]) - 1 not in sw                # final token has no label
    assert set(du.tolist()) <= set(sw.tolist())


def test_saved_replay_lines_up(tmp_path):
    run = _run(tmp_path)
    rows = [_row(row_id=19), _row(n_words=15, row_id=29)]
    sp = SavedPreds(_saved_dir(tmp_path, run, rows))
    assert sorted(sp.row_ids().tolist()) == [19, 29]
    assert sp.row_ids("Spanish->English").size == 2 and sp.row_ids("Hindi->English").size == 0
    r = replay_row(run, 29, "saved", saved=sp, row=rows[1])
    ex, w = row_windows(run, rows[1])
    sw, du = scored(w)
    assert [d["pos"] for d in r["decisions"]] == sw.tolist()
    sw_true = [d for d in r["decisions"] if d["y"] == 1]
    assert sw_true and all(d["next_lang"] in (L1, L2) for d in sw_true)
    with_dur = [d["pos"] for d in r["decisions"] if d["dur_probs"] is not None]
    assert with_dur == du.tolist()
    s = r["summary"]
    assert s["TP"] + s["FP"] + s["FN"] + s["TN"] == len(sw)
    assert all(d["fired"] == (d["p"] >= 0.5) for d in r["decisions"])


def test_saved_mismatch_raises(tmp_path):
    run = _run(tmp_path)
    rows = [_row()]
    sp = SavedPreds(_saved_dir(tmp_path, run, rows))
    replay_row(run, 19, "saved", saved=sp, row=rows[0])        # aligned: fine
    pos = np.array(sp.pos)
    sp.pos = pos + 1                                           # shifted positions
    with pytest.raises(ValueError, match="positions differ"):
        replay_row(run, 19, "saved", saved=sp, row=rows[0])
    sp.pos = pos
    assert scored(row_windows(run, rows[0])[1])[1].size        # the row has a duration point
    sp._didx = {}                                              # ...whose prediction is lost
    with pytest.raises(ValueError, match="duration"):
        replay_row(run, 19, "saved", saved=sp, row=rows[0])


def test_live_replay_on_tiny_model(tmp_path):
    transformers = pytest.importorskip("transformers")
    from src.models import build_model
    cfg = transformers.XLMRobertaConfig(vocab_size=300, hidden_size=32, num_hidden_layers=2,
                                        num_attention_heads=2, intermediate_size=64,
                                        max_position_embeddings=64)
    model = build_model(dict(model="xlmr", prev_lang=True), 300, xlmr_config=cfg)
    run = _run(tmp_path)
    row = _row(n_words=20)
    r = replay_row(run, 19, "live", live=LivePredictor.from_model(model, 0.5), row=row)
    sw, _ = scored(row_windows(run, row)[1])
    assert [d["pos"] for d in r["decisions"]] == sw.tolist()
    assert all(d["dur_probs"] is not None and abs(sum(d["dur_probs"]) - 1) < 1e-3
               for d in r["decisions"])                # live: duration everywhere


def test_display_pieces():
    assert "".join(display_pieces(["▁Hola", "▁ami", "go", ",", "▁you", "▁know"], None)) == \
        "Hola amigo, you know"
    assert "".join(display_pieces(["本", "人", "##いる", "It", "'", "s", "really", "##ly", "."],
                                  [0, 1, 2, 3, 3, 3, 4, 4, 5])) == "本人いる It's reallyly."


def test_short_large_three_class_collapses_with_cut(tmp_path):
    run = _run(tmp_path)                                       # 3-class, no binary json: cut 0.5
    probs, pred = run.short_large([0.3, 0.3, 0.4])
    assert probs == pytest.approx([0.6, 0.4]) and pred == 0
    (tmp_path / "data" / "results").mkdir(parents=True)
    (tmp_path / "data" / "results" / "duration_binary_toy.json").write_text(
        '{"two_class": {"cut": 0.3}}')
    assert run.short_large([0.3, 0.3, 0.4])[1] == 1           # the tuned cut decides


def test_short_large_binary_head_uses_offsets(tmp_path):
    run = _run(tmp_path)
    run.args["dur_classes"] = 2
    assert run.short_large([0.6, 0.4]) == ([0.6, 0.4], 0)
    run.dur_offsets = [0.0, 1.0]                               # log-offset favouring Large
    assert run.short_large([0.6, 0.4])[1] == 1
    assert run.dur_labels == ["short", "large"]


def test_saved_preds_read_npz_and_decisions_carry_short_large(tmp_path):
    run = _run(tmp_path)
    rows = [_row(row_id=19)]
    d = _saved_dir(tmp_path, run, rows)
    np.savez(tmp_path / "p.npz", **{f"test_{k}": np.load(os.path.join(d, f"test_{k}.npy"))
                                    for k in SavedPreds.KEYS})
    sp = SavedPreds(str(tmp_path / "p.npz"))
    r = replay_row(run, 19, "saved", saved=sp, row=rows[0])
    for dec in r["decisions"]:
        if dec["dur_probs"] is not None:                       # [0.2, 0.5, 0.3] saved
            assert dec["bin_probs"] == pytest.approx([0.7, 0.3]) and dec["bin_pred"] == 0
            assert dec["y_bin"] == int(dec["y_dur"] == 2)


REAL = os.path.exists("data/preds/xlmr_prev_gap_36k_rsv2/test_row.npy") and \
    os.path.exists("data/subword.parquet")


@pytest.mark.skipif(not REAL, reason="needs the saved XLM-R predictions and data/subword.parquet")
def test_real_saved_predictions_align():
    from src.replay import discover_runs, random_row
    runs = discover_runs()
    rng = np.random.default_rng(0)
    for run in runs.values():
        sp = SavedPreds(run.preds_path, run.pred_splits)
        for _ in range(10):
            r = replay_row(run, random_row(rng, sp, run), "saved", saved=sp)   # raises on mismatch
            assert r["decisions"]
