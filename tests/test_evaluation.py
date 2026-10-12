"""The one evaluation report every model is scored with."""
import numpy as np
import pytest

from src.evaluation import PAIR_KEYS, format_report, full_report, pair_key


def test_pair_keys_keep_both_german_pairs():
    assert len(PAIR_KEYS) == 15 and len(set(PAIR_KEYS)) == 15
    assert pair_key("German", "French") == "German->French" in PAIR_KEYS
    assert pair_key("German", "English") == "German->English" in PAIR_KEYS


def _toy():
    es, de = PAIR_KEYS.index("Spanish->English"), PAIR_KEYS.index("German->French")
    truth = np.array([1, 0, 1, 0, 1, 0, 0, 1])
    pred = np.array([1, 0, 0, 0, 1, 1, 0, 1])
    pairs = np.array([es, es, es, es, de, de, de, de])
    langs = np.array([0, 0, 1, 1, 0, 1, 0, 1])          # current token: 0 = L1, 1 = L2
    return truth, pred, pairs, langs


def test_report_breaks_down_by_pair_and_direction():
    truth, pred, pairs, langs = _toy()
    r = full_report(truth, pred, pairs, langs)
    assert set(r["per_pair"]) == {"Spanish->English", "German->French"}
    es, de = r["per_pair"]["Spanish->English"], r["per_pair"]["German->French"]
    assert es["f1"] == pytest.approx(2 / 3) and de["f1"] == pytest.approx(0.8)
    assert r["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)
    assert r["overall"]["tp"] == 3 and r["overall"]["fn"] == 1 and r["overall"]["fp"] == 1
    # direction = where the switch goes: from L1 means "into L2"
    into_l2, into_l1 = r["per_direction"]["into L2"], r["per_direction"]["into L1"]
    assert into_l2["n"] == 4 and into_l1["n"] == 4
    assert into_l2["f1"] == pytest.approx(1.0)          # both L1-position switches caught
    assert into_l1["tp"] == 1 and into_l1["fn"] == 1 and into_l1["fp"] == 1


def test_report_has_confidence_intervals():
    truth, pred, pairs, langs = _toy()
    lo, hi = full_report(truth, pred, pairs, langs)["overall"]["recall_ci"]
    assert 0 <= lo <= 0.75 <= hi <= 1


def test_duration_block_only_when_given():
    truth, pred, pairs, langs = _toy()
    assert "duration" not in full_report(truth, pred, pairs, langs)
    r = full_report(truth, pred, pairs, langs, dur_true=[0, 1, 2, 2], dur_pred=[0, 1, 2, 0])
    assert r["duration"]["accuracy"] == pytest.approx(0.75)
    assert r["duration"]["confusion"][2] == [1, 0, 1]


def test_format_report_names_every_pair():
    truth, pred, pairs, langs = _toy()
    text = format_report(full_report(truth, pred, pairs, langs), "toy")
    assert "Spanish->English" in text and "German->French" in text and "into L2" in text


def test_duration_per_pair_adds_up_to_the_pooled_confusion():
    es, de = PAIR_KEYS.index("Spanish->English"), PAIR_KEYS.index("German->French")
    truth, pred, pairs, langs = _toy()
    dt, dp = np.array([0, 1, 2, 2, 1]), np.array([0, 2, 2, 0, 1])
    r = full_report(truth, pred, pairs, langs, dt, dp,
                    dur_pairs=np.array([es, es, de, de, de]), dur_langs=np.array([0, 1, 0, 1, 1]))
    per = r["duration_per_pair"]
    assert set(per) == {"Spanish->English", "German->French"}
    total = sum(np.array(v["confusion"]) for v in per.values())
    assert total.tolist() == r["duration"]["confusion"]
    assert per["Spanish->English"]["accuracy"] == pytest.approx(0.5)
    assert sum(v["n"] for v in r["duration_per_direction"].values()) == 5


def test_end_to_end_duration_never_beats_the_oracle_score():
    truth, pred, pairs, langs = _toy()
    dt, dp = np.array([0, 1, 2, 2]), np.array([0, 1, 2, 0])
    dpair = np.zeros(4, int)
    every = full_report(truth, pred, pairs, langs, dt, dp, dur_pairs=dpair,
                        dur_fired=np.ones(4, bool))
    assert every["duration_end_to_end"]["accuracy"] == pytest.approx(every["duration"]["accuracy"])
    assert every["duration_end_to_end"]["macro_f1"] == pytest.approx(every["duration"]["macro_f1"])
    some = full_report(truth, pred, pairs, langs, dt, dp, dur_pairs=dpair,
                       dur_fired=np.array([True, False, True, True]))
    e = some["duration_end_to_end"]
    assert e["accuracy"] == pytest.approx(0.5) and e["switch_recall"] == pytest.approx(0.75)
    assert e["macro_f1"] <= some["duration"]["macro_f1"]
    assert e["per_class_f1"][1] == 0.0                    # the one medium switch was missed
    assert some["duration_per_pair"][PAIR_KEYS[0]]["end_to_end"]["accuracy"] == pytest.approx(0.5)
    text = format_report(some, "toy")
    assert "Duration by language pair" in text and "End to end" in text
