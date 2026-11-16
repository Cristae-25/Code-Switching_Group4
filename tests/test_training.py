"""Training helpers: seeding, class-weighted duration loss, saved predictions."""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.training import (count_duration_classes, duration_class_weights,  # noqa: E402
                          save_predictions, set_seed)


def test_same_seed_same_numbers():
    set_seed(3); a = (torch.randn(4), np.random.rand(2))
    set_seed(3); b = (torch.randn(4), np.random.rand(2))
    assert torch.equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_balanced_weights_are_inverse_frequency_with_mean_one():
    w = duration_class_weights(np.array([50, 25, 25]))
    assert w.mean() == pytest.approx(1.0)
    assert w[1] == pytest.approx(w[2]) == pytest.approx(2 * w[0])


def test_empty_class_gets_no_infinite_weight():
    w = duration_class_weights(np.array([10, 0, 10]))
    assert np.isfinite(w).all() and w.mean() == pytest.approx(1.0)


def _parquet(tmp_path):
    rows = [dict(row_id=0, first_language="Spanish", second_language="English", quality="p",   # train
                 piece_ids=[5, 6, 7, 8, 9, 10], piece_lang=[0, 1, 0, 0, 0, 1],
                 piece_word=[0, 1, 2, 3, 4, 5], piece_utt=[0] * 6, word_end=[True] * 6),
            dict(row_id=9, first_language="Spanish", second_language="English", quality="p",   # test
                 piece_ids=[5, 6], piece_lang=[0, 1], piece_word=[0, 1], piece_utt=[0, 0],
                 word_end=[True, True])]
    path = str(tmp_path / "sw.parquet")
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_class_counts_come_from_the_train_split_only(tmp_path):
    # train row: L1 L2 L1 L1 L1 L2 -> switches start runs of 1 (small), 3 (medium), and a
    # censored run of 1 at the end (class uncertain, not counted)
    counts = count_duration_classes(_parquet(tmp_path), unit="subword")
    assert list(counts) == [1, 1, 0]


def test_weighted_duration_loss_matches_a_hand_computation():
    w = torch.tensor([0.5, 2.0, 0.5])
    ce = torch.nn.CrossEntropyLoss(weight=w, reduction="none")
    logits = torch.zeros(2, 3); y = torch.tensor([0, 1])
    loss = ce(logits, y)                      # uniform logits: each sample's loss is w[y] * log 3
    assert torch.allclose(loss, torch.tensor([0.5, 2.0]) * np.log(3))


def test_predictions_are_saved_for_offline_tuning(tmp_path):
    res = dict(logit=np.array([0.1, -0.2]), y=np.array([1, 0]), pair=np.array([0, 0]),
               lang=np.array([0, 1]), shared=np.array([True, False]),
               dtrue=np.array([2]), dpred=np.array([1]), dprob=np.array([[0.2, 0.5, 0.3]]))
    path = str(tmp_path / "preds.npz")
    save_predictions(path, val=res, test=res)
    z = np.load(path)
    assert {"val_logit", "test_logit", "test_dprob", "val_pair", "test_shared"} <= set(z.files)
    assert z["test_dprob"].shape == (1, 3)
