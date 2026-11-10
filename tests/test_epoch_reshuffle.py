"""Per-epoch reshuffle of the training stream, and the --eval-every curve."""
import argparse
import importlib.util
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

torch = pytest.importorskip("torch")

from src.streaming import SwitchWindowDataset, iter_rows  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _parquet(tmp_path, n_rows=240, group=20):
    rng = np.random.default_rng(0)
    rows = []
    for r in range(n_rows):
        rid = r * 10 + (r % 8)                         # every row in the train split
        n = int(rng.integers(4, 10))
        rows.append(dict(row_id=rid, first_language="Spanish", second_language="English", quality="p",
                         piece_ids=rng.integers(5, 50, n).tolist(), piece_lang=rng.integers(0, 2, n).tolist(),
                         piece_word=list(range(n)), piece_utt=[0] * n, word_end=[True] * n))
    path = str(tmp_path / "sw.parquet")
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False), path,
                   row_group_size=group)
    return path


def _ds(path, reshuffle=True, **kw):
    return SwitchWindowDataset(path, None, split="train", window=8, stride=6, shuffle_buffer=16,
                               seed=0, unit="subword", batch_rows=10, reshuffle=reshuffle, **kw)


def _order(ds):
    return [(int(x["row_id"]), int(x["start"])) for x in ds]


def test_first_epoch_is_unchanged_and_later_epochs_reshuffle(tmp_path):
    path = _parquet(tmp_path)
    ds = _ds(path)
    e0, e1, e2 = _order(ds), _order(ds), _order(ds)
    assert e0 == _order(_ds(path))                       # epoch 0 = a fresh dataset's order
    assert e1 != e0 and e2 != e1                         # later epochs differ ...
    assert sorted(e0) == sorted(e1) == sorted(e2)        # ... over exactly the same windows
    fresh = _ds(path)
    _order(fresh)
    assert _order(fresh) == e1                           # and are reproducible from the seed


def test_epoch_zero_matches_the_original_file_order_stream(tmp_path):
    path = _parquet(tmp_path)
    plain = [r["row_id"] for r in iter_rows(path, "train", 10, columns=["row_id", "first_language",
                                                                       "quality"])]
    perm = [r["row_id"] for r in iter_rows(path, "train", 10, columns=["row_id", "first_language", "quality"],
                                          row_group_order=list(range(12)))]
    assert plain == perm                                 # identity order = the old reader
    shuffled = [r["row_id"] for r in iter_rows(path, "train", 10, columns=["row_id", "first_language",
                                                                          "quality"],
                                              row_group_order=_ds(path).epoch_order(1))]
    assert shuffled != plain and sorted(shuffled) == sorted(plain)


def test_reshuffle_is_off_by_default_so_old_runs_reproduce(tmp_path):
    path = _parquet(tmp_path)
    ds = _ds(path, reshuffle=False)
    e0 = _order(ds)
    assert _order(ds) == e0 == _order(_ds(path, reshuffle=True))   # epoch 0 is the same either way
    assert SwitchWindowDataset(path, None).reshuffle is False


def test_val_stream_never_reshuffles(tmp_path):
    path = _parquet(tmp_path)
    ds = SwitchWindowDataset(path, None, split="train", window=8, stride=6, shuffle_buffer=0,
                             seed=0, unit="subword", batch_rows=10)
    assert _order(ds) == _order(ds)


def test_persistent_workers_reshuffle_too(tmp_path):
    path = _parquet(tmp_path)
    loader = torch.utils.data.DataLoader(_ds(path), batch_size=None, num_workers=2,
                                         persistent_workers=True)
    epochs = [[(int(x["row_id"]), int(x["start"])) for x in loader] for _ in range(2)]
    assert epochs[0] != epochs[1] and sorted(epochs[0]) == sorted(epochs[1])


def _train_module():
    spec = importlib.util.spec_from_file_location("train_anticipation",
                                                  os.path.join(ROOT, "scripts", "train_anticipation.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_curve_point_scores_a_fixed_slice_and_restores_train_mode(tmp_path):
    mod = _train_module()
    from src.models import build_model
    from src.streaming import make_loader
    path = _parquet(tmp_path)
    loader = make_loader(path, None, "train", 4, 0, window=8, stride=6, unit="subword", shuffle_buffer=0)
    m = build_model(dict(model="scratch", d_model=16, layers=1, heads=2, window=8), 60, use_lang=False)
    m.train()
    a = mod.curve_point(m, loader, torch.device("cpu"), 3, step=10)
    assert m.training
    b = mod.curve_point(m, loader, torch.device("cpu"), 3, step=20)
    assert a["n"] == b["n"] > 0 and a["val_switch_loss"] == pytest.approx(b["val_switch_loss"])
    assert {"step", "val_switch_loss", "val_f1", "tau", "val_dur_macro_f1"} <= set(a)


def test_curve_point_logs_the_val_duration_loss(tmp_path):
    """val_dur_loss is the 3-class cross-entropy at the dur_mask positions of the same slice."""
    mod = _train_module()
    from src.models import build_model
    from src.streaming import make_loader
    loader = make_loader(_parquet(tmp_path), None, "train", 4, 0, window=8, stride=6, unit="subword",
                         shuffle_buffer=0)
    m = build_model(dict(model="scratch", d_model=16, layers=1, heads=2, window=8), 60, use_lang=False)
    pt = mod.curve_point(m, loader, torch.device("cpu"), 3, step=1)
    res = mod.evaluate(m, loader, torch.device("cpu"), 3)
    assert pt["n_dur"] == len(res["dtrue"]) > 0
    want = -np.mean(np.log(res["dprob"][np.arange(len(res["dtrue"])), res["dtrue"]]))
    assert pt["val_dur_loss"] == pytest.approx(want, rel=1e-5)


def test_curve_point_leaves_the_global_rng_untouched(tmp_path):
    """A DataLoader iterator draws from torch's RNG; the snapshot must not shift dropout."""
    mod = _train_module()
    from src.models import build_model
    from src.streaming import make_loader
    loader = make_loader(_parquet(tmp_path), None, "train", 4, 0, window=8, stride=6, unit="subword",
                         shuffle_buffer=0)
    m = build_model(dict(model="scratch", d_model=16, layers=1, heads=2, window=8), 60, use_lang=False)
    torch.manual_seed(1)
    before = torch.get_rng_state()
    mod.curve_point(m, loader, torch.device("cpu"), 2, step=1)
    assert torch.equal(before, torch.get_rng_state())
