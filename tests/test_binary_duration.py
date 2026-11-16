"""--dur-classes 2: Short (1-6) vs Large (7+) duration, trained and scored on two classes."""
import importlib.util
import os

import numpy as np
import pytest
import torch

from src.anticipation import BINARY_LABELS, DUR_LABELS, collapse_duration, dur_labels
from src.evaluation import format_report, full_report
from src.xlmr_model import CausalPretrained

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _train_script():
    spec = importlib.util.spec_from_file_location(
        "train_anticipation", os.path.join(ROOT, "scripts", "training", "train_anticipation.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_collapse_merges_small_and_medium_and_keeps_unlabelled():
    y = np.array([-1, 0, 1, 2, -1, 2], dtype=np.int64)
    assert collapse_duration(y).tolist() == [-1, 0, 0, 1, -1, 1]
    assert collapse_duration(y).dtype == y.dtype
    t = torch.tensor(y)
    out = collapse_duration(t)
    assert isinstance(out, torch.Tensor) and out.dtype == t.dtype
    assert out.tolist() == [-1, 0, 0, 1, -1, 1]


def test_labels_follow_the_class_count():
    assert dur_labels(3) == DUR_LABELS
    assert dur_labels(2) == BINARY_LABELS == ["short", "large"]
    with pytest.raises(KeyError):
        dur_labels(4)


def _report(n_dur):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    pairs, langs = rng.integers(0, 3, 200), rng.integers(0, 2, 200)
    dt, dp = rng.integers(0, n_dur, 80), rng.integers(0, n_dur, 80)
    return full_report(y, y, pairs, langs, dt, dp, dur_pairs=rng.integers(0, 3, 80),
                       dur_langs=rng.integers(0, 2, 80), dur_fired=rng.integers(0, 2, 80), n_dur=n_dur)


def test_two_class_report_has_two_by_two_blocks():
    r = _report(2)
    assert np.asarray(r["duration"]["confusion"]).shape == (2, 2)
    assert len(r["duration_end_to_end"]["per_class_f1"]) == 2
    for block in r["duration_per_pair"].values():
        assert np.asarray(block["confusion"]).shape == (2, 2)
    text = format_report(r, "t")
    assert "2-class accuracy" in text and "F1_short" in text and "F1_large" in text
    assert "medium" not in text


def test_default_report_is_still_three_classes():
    r = _report(3)
    assert np.asarray(r["duration"]["confusion"]).shape == (3, 3)
    assert "F1_medium" in format_report(r, "t")


def test_score_reads_the_class_count_from_the_probabilities():
    mod = _train_script()
    rng = np.random.default_rng(1)
    n, nd = 300, 60
    res = dict(logit=rng.normal(size=n), y=rng.integers(0, 2, n), pair=rng.integers(0, 3, n),
               lang=rng.integers(0, 2, n), dtrue=rng.integers(0, 2, nd), dpred=rng.integers(0, 2, nd),
               dprob=rng.dirichlet([1, 1], nd))
    assert np.asarray(mod.score(res, 0.5)["duration"]["confusion"]).shape == (2, 2)


def _tiny_roberta():
    from transformers import RobertaConfig
    return RobertaConfig(vocab_size=300, hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                         intermediate_size=64, max_position_embeddings=80)


def test_a_two_class_head_collapses_the_batch_labels():
    mod = _train_script()
    m = CausalPretrained(config=_tiny_roberta(), n_dur=2).eval()
    ids = torch.randint(5, 300, (2, 12))
    _, dur = m(ids, torch.zeros_like(ids))
    assert dur.shape[-1] == 2 and mod.head_classes(m, dur) == 2
    b = {"y_dur": torch.tensor([[-1, 0, 1, 2]])}
    assert mod.batch_dur_labels(b, 2).tolist() == [[-1, 0, 0, 1]]
    assert mod.batch_dur_labels(b, 3) is b["y_dur"]


def test_an_ordinal_three_class_head_is_not_mistaken_for_two_classes():
    mod = _train_script()
    m = CausalPretrained(config=_tiny_roberta(), dur_head="ordinal").eval()
    ids = torch.randint(5, 300, (1, 8))
    _, dur = m(ids, torch.zeros_like(ids))
    assert dur.shape[-1] == 2 and mod.head_classes(m, dur) == 3


def test_two_classes_reject_the_ordinal_head():
    import argparse
    mod = _train_script()
    a = argparse.Namespace(unit="subword", path=None, window=None, stride=None, vocab_size=None, out=None,
                           ckpt=None, names="script_words", no_lang=False, model="xlmr", lr=None,
                           turn_sep=False, prev_lang=False, predict_at="piece", context=None,
                           dur_classes=2, dur_head="ordinal")
    with pytest.raises(SystemExit):
        mod.resolve_defaults(a)
