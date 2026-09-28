"""Named-entity edge cases, and how each names policy labels them.

``tests/fixtures/name_cases.json`` freezes the real AnE-NER masks and AnE-LID
labels for 20 hand-picked sentences (regenerate with
``scripts/compare_name_strategies.py --refresh``), so these tests exercise the
policies on genuine model output without loading a model.

Policies (``segments.resolve_names``):
    transparent   names are no language; skipped like punctuation
    language      names count as AnE-LID tagged them
    hybrid        titles keep their language; opaque names can't open a run
    script        Latin-script names -> English, native-script names -> L1
    script_words  script in non-Latin rows; in Latin rows the name's words
                  decide (English / L1 by dictionary margin), opaque -> context

Expected labels are written as L1 / L2 / NE for the name tokens only.
"""
from __future__ import annotations

import json
import os

import pytest

from src.dispatch import tag_tokens
from src.segments import NE, repair_spans, resolve_names

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "name_cases.json")
CASES = {c["id"]: c for c in json.load(open(FIXTURE))}


def _tagged(case):
    return tag_tokens(case["tokens"], case["l1"], case["l2"],
                      ane_labels=case["ane"], is_entity=case["is_entity"])


def _codes(case, policy, words):
    out = resolve_names(_tagged(case), case["is_entity"], case["l1"], case["l2"], policy)
    by_tok = {}
    for t, lang in out:
        by_tok.setdefault(t, []).append(
            "L1" if lang == case["l1"] else "L2" if lang == case["l2"] else
            NE if lang == NE else "O")
    return [by_tok[w][0] for w in words]


# --- span repair: AnE-NER fragments names; every policy needs whole names ----

def _mask(case):
    return dict(zip(case["tokens"], repair_spans(case["tokens"], case["is_entity"],
                                                 case["l1"], case["l2"])))


def test_repair_extends_a_title_over_capitalised_words_and_numbers():
    m = _mask(CASES["severance_in_spanish"])        # NER flagged only "Severance"
    assert m["Severance"] and m["Season"] and m["2"]
    assert not m["vi"] and not m["y"]


def test_repair_bridges_a_function_word_inside_a_name():
    m = _mask(CASES["got_in_cantonese"])            # NER flagged "Game", "Thrones"
    assert m["Game"] and m["of"] and m["Thrones"]


def test_repair_extends_left_over_an_english_title_head():
    m = _mask(CASES["title_in_malay"])              # NER flagged only "of Texas"
    assert all(m[w] for w in ("The", "University", "of", "Texas"))
    assert not m["di"]


def test_repair_does_not_swallow_an_ordinary_l1_word():
    m = _mask(CASES["messi_in_spanish"])            # "Ayer" is capitalised but Spanish
    assert not m["Ayer"] and not m["marcó"]


def test_repair_does_not_swallow_german_capitalised_nouns():
    toks = ["Die", "EU", "Kommission", "tagt"]
    assert repair_spans(toks, [False, True, False, False], "German", "English") == \
        [False, True, False, False]


def test_repair_leaves_non_latin_names_alone():
    c = CASES["arabic_script_name"]
    assert repair_spans(c["tokens"], c["is_entity"], c["l1"], c["l2"]) == c["is_entity"]


# --- the edge-case table ------------------------------------------------------
#   case id: (name tokens, {policy: expected codes})
TABLE = {
    "severance_in_spanish":    (["Severance", "Season"],
                                dict(transparent="NE NE", language="L2 L2", hybrid="L1 L1",
                                     script="L2 L2", script_words="L2 L2")),
    "messi_in_spanish":        (["Leo", "Messi"],
                                dict(transparent="NE NE", language="L1 L1", hybrid="L1 L1",
                                     script="L2 L2", script_words="L1 L1")),
    "messi_in_english_clause": (["Leo", "Messi"],
                                dict(transparent="NE NE", language="L2 L1", hybrid="L2 L2",
                                     script="L2 L2", script_words="L2 L2")),
    "real_madrid_in_french":   (["Real", "Madrid"],
                                dict(transparent="NE NE", language="L2 L2", hybrid="L1 L1",
                                     script="L2 L2", script_words="L2 L2")),
    "psg_at_switch":           (["PSG"],
                                dict(transparent="NE", language="L1", hybrid="L1",
                                     script="L2", script_words="L1")),
    "union_europeenne":        (["européenne"],
                                dict(transparent="NE", language="L1", hybrid="L1",
                                     script="L2", script_words="L1")),
    "arabic_script_name":      (["ترامب"],
                                dict(transparent="NE", language="L1", hybrid="L1",
                                     script="L1", script_words="L1")),
    "severance_in_arabic":     (["Severance", "Season"],
                                dict(transparent="NE NE", language="L2 L2", hybrid="L1 L1",
                                     script="L2 L2", script_words="L2 L2")),
    "messi_latin_in_arabic":   (["Messi"],
                                dict(transparent="NE", language="L1", hybrid="L1",
                                     script="L2", script_words="L2")),
    "got_in_cantonese":        (["Game", "of", "Thrones"],
                                dict(transparent="NE NE NE", language="L2 L2 L2",
                                     hybrid="L2 L2 L2", script="L2 L2 L2",
                                     script_words="L2 L2 L2")),
    "han_place_name":          (["中", "關"],
                                dict(transparent="NE NE", language="L1 L1", hybrid="L1 L1",
                                     script="L1 L1", script_words="L1 L1")),
    "han_messi":               (["梅", "西"],
                                dict(transparent="NE NE", language="L1 L1", hybrid="L1 L1",
                                     script="L1 L1", script_words="L1 L1")),
    "iphone_in_japanese":      (["ホ", "iPhone", "Pro"],
                                dict(transparent="NE NE NE", language="L1 L2 L2",
                                     hybrid="L1 L1 L1", script="L1 L2 L2",
                                     script_words="L1 L2 L2")),
    "acronym_in_korean":       (["WWE"],
                                dict(transparent="NE", language="L2", hybrid="L1",
                                     script="L2", script_words="L2")),
    "acronyms_in_german":      (["EU", "NATO"],
                                dict(transparent="NE NE", language="L1 L2", hybrid="L1 L2",
                                     script="L2 L2", script_words="L1 L2")),
    "weekday_false_positive":  (["Monza"],
                                dict(transparent="NE", language="L1", hybrid="L1",
                                     script="L2", script_words="L1")),
    "brand_in_russian":        (["Marvel"],
                                dict(transparent="NE", language="L2", hybrid="L1",
                                     script="L2", script_words="L2")),
    "bollywood_in_hinglish":   (["Shraddha", "Kapoor"],
                                dict(transparent="NE NE", language="L1 L1", hybrid="L1 L1",
                                     script="L2 L2", script_words="L1 L1")),
    "title_in_malay":          (["The", "University", "of", "Texas"],
                                dict(transparent="NE NE NE NE", language="L2 L2 L2 L2",
                                     hybrid="L2 L2 L2 L2", script="L2 L2 L2 L2",
                                     script_words="L2 L2 L2 L2")),
    "name_in_german_french":   (["Nel"],
                                dict(transparent="NE", language="L2", hybrid="L1",
                                     script="L2", script_words="L1")),
}

POLICIES = ["transparent", "language", "hybrid", "script", "script_words"]


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("case_id", list(TABLE))
def test_edge_case_labels(case_id, policy):
    words, expected = TABLE[case_id]
    assert " ".join(_codes(CASES[case_id], policy, words)) == expected[policy]


def test_every_fixture_case_is_in_the_table():
    assert set(TABLE) == set(CASES)
