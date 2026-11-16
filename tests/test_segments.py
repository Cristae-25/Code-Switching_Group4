"""Tests for the code-switch tagging pipeline.

Run with: pytest tests/ -q
"""
import pytest

from src.routing import route, lang_code, expected_script
from src.segments import absorb_other, to_segments, l2_runs
from src.tokenizer import tokenize_mixed
from src.unicode_tagger import tag_token_unicode, tag_tokens_unicode


# --- segments (the plan's specified tests) ---------------------------------

def test_segments_reconstruct_token_count():
    tagged = [("a", "en"), ("b", "en"), ("c", "hi"), ("d", "hi"), ("e", "hi"), ("f", "en")]
    segs = to_segments(tagged)
    assert sum(length for _, length in segs) == len(tagged)
    assert segs == [["en", 2], ["hi", 3], ["en", 1]]


def test_no_adjacent_duplicate_languages():
    tagged = [("a", "en"), ("b", "hi"), ("c", "en")]
    segs = to_segments(tagged)
    langs = [lang for lang, _ in segs]
    assert all(langs[i] != langs[i + 1] for i in range(len(langs) - 1))


def test_segments_empty():
    assert to_segments([]) == []


# --- "other" absorption ----------------------------------------------------

def test_other_absorbed_into_preceding_run():
    tagged = [("hola", "es"), (",", "other"), ("mundo", "es")]
    assert to_segments(absorb_other(tagged)) == [["es", 3]]


def test_other_at_start_absorbed_into_following_run():
    tagged = [("¿", "other"), ("qué", "es"), ("tal", "es")]
    assert to_segments(absorb_other(tagged)) == [["es", 3]]


def test_other_does_not_bridge_a_language_switch():
    """Punctuation between two languages must not merge them into one run."""
    tagged = [("hola", "es"), (",", "other"), ("hello", "en")]
    segs = to_segments(absorb_other(tagged))
    assert [lang for lang, _ in segs] == ["es", "en"]
    assert sum(n for _, n in segs) == 3


def test_absorb_preserves_token_count():
    tagged = [("a", "other"), ("b", "other"), ("c", "en"), ("d", "other")]
    assert len(absorb_other(tagged)) == len(tagged)


def test_all_other_tokens_stay_other():
    tagged = [("...", "other"), ("123", "other")]
    assert to_segments(absorb_other(tagged)) == [["other", 2]]


# --- L2 run extraction (the original question) -----------------------------

def test_l2_runs_counts_consecutive_l2_tokens():
    tagged = [("Como", "es"), ("se", "es"), ("dice", "es"),
              ("Neuroplasticity", "en"), ("en", "es"), ("Espanol", "es")]
    assert l2_runs(to_segments(tagged), "en") == [1]


def test_l2_runs_multiple_switches():
    tagged = [("a", "es"), ("x", "en"), ("y", "en"), ("b", "es"), ("z", "en")]
    assert l2_runs(to_segments(tagged), "en") == [2, 1]


def test_l2_run_at_end_of_text_is_recorded():
    tagged = [("a", "es"), ("x", "en"), ("y", "en")]
    assert l2_runs(to_segments(tagged), "en") == [2]


def test_no_l2_gives_no_runs():
    assert l2_runs(to_segments([("a", "es"), ("b", "es")]), "en") == []


# --- tokenizer -------------------------------------------------------------

def test_cjk_tokenized_per_character_latin_per_word():
    assert tokenize_mixed("我們 love 咖啡") == ["我", "們", "love", "咖", "啡"]


def test_accented_latin_stays_one_token():
    """The plan's [A-Za-z]+ pattern would split 'dificil' at the accent."""
    assert tokenize_mixed("difícil") == ["difícil"]
    assert tokenize_mixed("été naïve straße") == ["été", "naïve", "straße"]


def test_vietnamese_diacritics_stay_one_token():
    assert tokenize_mixed("tiếng Việt") == ["tiếng", "Việt"]


def test_apostrophes_kept_inside_english_words():
    assert tokenize_mixed("it's") == ["it's"]
    assert tokenize_mixed("don’t") == ["don’t"]


def test_space_delimited_non_latin_scripts_tokenize_per_word():
    """Cyrillic/Arabic/Devanagari use spaces, so they are word-level, not char-level."""
    assert tokenize_mixed("привет мир") == ["привет", "мир"]
    assert tokenize_mixed("महिलाओं की") == ["महिलाओं", "की"]


def test_punctuation_and_digits_are_separate_tokens():
    assert tokenize_mixed("hi, 42!") == ["hi", ",", "42", "!"]


def test_tokenizer_drops_no_characters():
    text = "我們 love 咖啡, it's 100% 好!"
    assert "".join(tokenize_mixed(text)) == "".join(text.split())


# --- unicode tagger --------------------------------------------------------

def test_tag_token_unicode_identifies_scripts():
    assert tag_token_unicode("我") == "Han"
    assert tag_token_unicode("hello") == "Latin"
    assert tag_token_unicode("привет") == "Cyrillic"
    assert tag_token_unicode("!") == "other"


def test_unicode_tagging_maps_scripts_to_row_languages():
    tokens = ["我", "們", "love", "咖", "啡"]
    tagged = tag_tokens_unicode(tokens, "Chinese", "English")
    assert [lang for _, lang in tagged] == ["Chinese", "Chinese", "English", "Chinese", "Chinese"]


def test_unicode_tagging_handles_japanese_kana_and_han():
    tagged = tag_tokens_unicode(["私", "は", "コ", "cool"], "Japanese", "English")
    assert [lang for _, lang in tagged] == ["Japanese", "Japanese", "Japanese", "English"]


def test_unicode_punctuation_is_other():
    tagged = tag_tokens_unicode(["我", ",", "ok"], "Chinese", "English")
    assert [lang for _, lang in tagged] == ["Chinese", "other", "English"]


# --- routing ---------------------------------------------------------------

@pytest.mark.parametrize("lang", ["Cantonese", "Chinese", "Japanese", "Korean", "Russian", "Arabic"])
def test_script_disjoint_languages_route_to_unicode(lang):
    assert route(lang) == "unicode"


@pytest.mark.parametrize("lang", ["German", "Spanish", "French", "Italian",
                                  "Malay", "Philippines", "Vietnamese", "Hindi"])
def test_latin_script_languages_never_route_to_unicode(lang):
    assert route(lang) in {"ane", "dict"}


def test_hindi_routes_to_a_model_because_rows_are_romanized():
    """Resolves the plan's 'unicode_pending_check' gate: most Hindi rows are
    romanized Hinglish, so script alone cannot separate them. Devanagari rows
    are re-routed per row by dispatch.row_strategy."""
    assert route("Hindi") == "ane"


@pytest.mark.parametrize("lang", ["Spanish", "Italian", "Malay", "Philippines",
                                  "German", "French", "Vietnamese"])
def test_latin_english_pairs_route_to_ane(lang):
    """AnE-LID beat dict-freq by ~5 points on human-annotated LinCE/Denglisch."""
    assert route(lang) == "ane"


def test_french_second_language_uses_dictionary():
    """dict-freq handles a non-English L2; AnE and the LinCE models cannot."""
    assert route("German", "French") == "dict"


def test_dictionary_tags_german_french_correctly():
    from src.dispatch import tag_tokens
    toks = "Ich werde morgen Un nouveau buteur".split()
    tagged = tag_tokens(toks, "German", "French")
    assert [l for _, l in tagged] == ["German"] * 3 + ["French"] * 3


def test_every_dataset_language_has_a_fasttext_code():
    for lang in ["German", "Philippines", "Vietnamese", "Malay", "Spanish",
                 "French", "Italian", "Hindi", "English"]:
        assert lang_code(lang), f"missing fastText code for {lang}"


def test_expected_script_known_for_unicode_routed_languages():
    assert expected_script("Chinese") == {"Han"}
    assert expected_script("Japanese") == {"Han", "Hiragana", "Katakana"}
    assert expected_script("Korean") == {"Hangul"}
    assert expected_script("Russian") == {"Cyrillic"}


# --- measured quality tiers ------------------------------------------------

def test_script_disjoint_languages_are_exact():
    from src.routing import tagger_quality
    for lang in ["Cantonese", "Chinese", "Japanese", "Korean", "Russian", "Arabic"]:
        assert tagger_quality(lang) == "exact"


def test_all_latin_pairs_now_provisional():
    """dict-freq lifted every Latin pair to 100% on the gold set - but that set
    is small and self-labelled, so they are `provisional`, not `exact`."""
    from src.routing import tagger_quality
    for lang in ["Vietnamese", "Spanish", "German", "French", "Italian",
                 "Philippines", "Hindi", "Malay"]:
        assert tagger_quality(lang) == "provisional"


# --- persistence round-trip ------------------------------------------------

def test_parquet_roundtrip_preserves_segments_and_tokens(tmp_path):
    import pandas as pd
    from src.persist import load_tagged, save_tagged, tag_dataframe

    df = pd.DataFrame([
        {"data_generation_result": "['我們 love 咖啡']",
         "first_language": "Chinese", "second_language": "English"},
        {"data_generation_result": "['Como se dice Neuroplasticity en Espanol']",
         "first_language": "Spanish", "second_language": "English"},
    ])
    tagged = tag_dataframe(df, progress=False)
    path = save_tagged(tagged, str(tmp_path / "t.parquet"))
    back = load_tagged(path)

    assert len(back) == 2
    for orig, restored in zip(tagged.itertuples(), back.itertuples()):
        assert len(restored.tokens) == len(restored.token_lang)
        assert sum(restored.segment_len) == len(restored.tokens)
        assert list(orig.l2_runs) == list(restored.l2_runs)
    assert back.tokens_tagged.iloc[0][2] == ("love", "English")


# --- per-row Hindi script routing ------------------------------------------

def test_devanagari_hindi_row_uses_exact_unicode_tagger():
    """Devanagari rows are script-disjoint from English, so the LinCE model
    (trained on romanized Hinglish) is the wrong tool for them."""
    from src.dispatch import tag_tokens
    tokens = ["महिलाओं", "की", "सेहत", "right", "?"]
    tagged = tag_tokens(tokens, "Hindi", "English")
    assert [l for _, l in tagged] == ["Hindi", "Hindi", "Hindi", "English", "other"]


def test_romanized_hindi_row_still_routes_to_bert():
    from src.dispatch import _hindi_is_devanagari
    assert not _hindi_is_devanagari(["Woh", "kal", "science", "news", "padhenge"])
    assert _hindi_is_devanagari(["महिलाओं", "की", "ok"])


def test_devanagari_words_in_a_romanized_row_are_hindi_by_script():
    """A romanized-majority row goes to the model, but any Devanagari word in it
    is Hindi by construction - the model must not be allowed to call it English."""
    from src.dispatch import tag_tokens
    tokens = ["Woh", "kal", "market", "गया", "tha", "and", "उसने", "bread", "liya"]
    tagged = dict(tag_tokens(tokens, "Hindi", "English"))
    assert tagged["गया"] == "Hindi" and tagged["उसने"] == "Hindi"


# --- model cache survives autoreload ---------------------------------------

def test_model_registry_survives_module_reload():
    """%autoreload re-executes modules, wiping module-level globals. The model
    registry must outlive that or every row reloads a 679MB model from disk."""
    import importlib
    import sys as _sys
    from src import bert_tagger
    bert_tagger._cache_dict()["sentinel"] = "kept"
    importlib.reload(bert_tagger)
    assert bert_tagger._cache_dict().get("sentinel") == "kept"
    del getattr(_sys, bert_tagger._REGISTRY)["sentinel"]


# --- punctuation must not inflate run lengths ------------------------------

def test_punctuation_does_not_count_toward_l2_run_length():
    """absorb_other gives boundary punctuation to the following language; a
    sentence-final "." from the L1 sentence must not become an L2 token."""
    from src.segments import absorb_other, l2_word_runs
    tagged = [("من", "Arabic"), (".", "other"),
              ("The", "English"), ("drama", "English")]
    absorbed = absorb_other(tagged)
    assert absorbed[1][1] == "English"          # absorbed forward, as designed
    assert l2_word_runs(absorbed, "English") == [2]   # but counted as 2, not 3


def test_word_runs_ignore_internal_punctuation():
    from src.segments import l2_word_runs
    tagged = [("a", "es"), ("x", "en"), (",", "en"), ("y", "en"), ("b", "es")]
    assert l2_word_runs(tagged, "en") == [2]


def test_word_runs_match_segment_runs_when_no_punctuation():
    from src.segments import l2_runs, l2_word_runs, to_segments
    tagged = [("a", "es"), ("x", "en"), ("y", "en"), ("b", "es"), ("z", "en")]
    assert l2_word_runs(tagged, "en") == l2_runs(to_segments(tagged), "en") == [2, 1]


def test_digits_are_transparent_not_l2():
    from src.segments import l2_word_runs
    tagged = [("a", "es"), ("x", "en"), ("150", "en"), ("y", "en")]
    assert l2_word_runs(tagged, "en") == [2]


def test_recompute_runs_fixes_inflated_counts_without_retagging(tmp_path):
    """The parquet keeps full token sequences, so a change to the counting rule
    is a cheap rewrite rather than another pass over the models."""
    import pandas as pd
    from src.persist import recompute_runs, save_tagged, tag_dataframe

    df = pd.DataFrame([{
        "data_generation_result": "['من . The drama between Joan']",
        "first_language": "Arabic", "second_language": "English"}])
    tagged = tag_dataframe(df, progress=False)
    path = save_tagged(tagged, str(tmp_path / "t.parquet"))

    # simulate the pre-fix artifact: punctuation counted toward the run
    stale = pd.read_parquet(path)
    stale["l2_runs"] = [[99]]
    stale.to_parquet(path, index=False)

    fixed = recompute_runs(path)
    assert list(fixed.l2_runs.iloc[0]) == [4]      # The drama between Joan
    assert int(fixed.n_switches.iloc[0]) == 1


def test_recompute_matches_tag_row_exactly(tmp_path):
    """The parquet recompute path and the live tagging path must agree - they
    differ only in that one reads stored labels, so any drift is a bug."""
    import pandas as pd
    from src.persist import recompute_runs, save_tagged, tag_dataframe
    from src.segments import tag_row

    rows = [
        ("['मुझे lagta hai Hyderabad metro project redefine hogi. It is great!']",
         "Hindi", "English"),
        ("['El hueso fue descubierto . It is fascinating how these creatures roamed .']",
         "Spanish", "English"),
        ("['我們 love 咖啡 , it is so good !']", "Cantonese", "English"),
    ]
    df = pd.DataFrame([{"data_generation_result": c, "first_language": a,
                        "second_language": b} for c, a, b in rows])
    live = [tag_row(c, a, b)["l2_runs"] for c, a, b in rows]
    path = save_tagged(tag_dataframe(df, progress=False), str(tmp_path / "t.parquet"))
    recomputed = recompute_runs(path)
    assert [list(r) for r in recomputed.l2_runs] == live


# --- named entities are transparent ----------------------------------------

def test_entity_words_are_relabelled_but_punctuation_is_not():
    from src.segments import NE, apply_entities
    tagged = [("Leo", "English"), ("Messi", "English"), (".", "other")]
    out = apply_entities(tagged, [True, True, True])
    assert [l for _, l in out] == [NE, NE, "other"]


def test_a_name_in_a_cjk_row_is_not_a_switch():
    """The script tagger calls a Latin-script name English; as NE it must not
    create a run."""
    from src.segments import NE, absorb_other, l2_word_runs
    tagged = [("我", "Chinese"), ("見", "Chinese"), ("Messi", NE), ("好", "Chinese")]
    assert l2_word_runs(absorb_other(tagged), "English") == []


def test_a_name_neither_breaks_nor_extends_an_l2_run():
    from src.segments import NE, absorb_other, l2_word_runs
    tagged = [("vi", "Spanish"), ("the", "English"), ("Messi", NE),
              ("match", "English"), ("ayer", "Spanish")]
    assert l2_word_runs(absorb_other(tagged), "English") == [2]


def test_absorption_looks_through_entities_at_a_boundary():
    """"hola , Messi hello": the comma sits at a genuine es->en boundary even
    though a name separates it from the English word."""
    from src.segments import NE, absorb_other
    tagged = [("hola", "es"), (",", "other"), ("Messi", NE), ("hello", "en")]
    out = absorb_other(tagged)
    assert out[1][1] == "en" and out[2][1] == NE


def test_entities_are_dropped_from_the_anticipation_stream():
    from src.anticipation import word_stream
    words, _ = word_stream(["vi", "a", "Messi", "ayer"],
                           ["Spanish", "Spanish", "NE", "Spanish"], "Spanish", "English")
    assert words == ["vi", "a", "ayer"]


def test_devanagari_hindi_row_is_routed_to_unicode_per_row():
    from src.dispatch import row_strategy
    assert row_strategy(["महिलाओं", "की", "ok"], "Hindi", "English") == "unicode"
    assert row_strategy(["Woh", "kal", "science"], "Hindi", "English") == "ane"


def test_precomputed_ane_labels_skip_the_model():
    """The corpus run batches AnE-LID and hands labels in; dispatch must use
    them verbatim rather than re-running the model."""
    from src.dispatch import tag_tokens
    toks = ["hola", "friend", ","]
    tagged = tag_tokens(toks, "Spanish", "English",
                        ane_labels=["notEnglish", "English", "Other"])
    assert [l for _, l in tagged] == ["Spanish", "English", "other"]


# --- review spot-check ------------------------------------------------------

def _review_module():
    import importlib.util
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "scripts", "labelling", "review_server.py")
    spec = importlib.util.spec_from_file_location("review_server", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_review_sample_covers_every_pair_and_boosts_weak_ones():
    import os
    rs = _review_module()
    if not os.path.exists(rs.PARQUET):
        pytest.skip("data/tagged.parquet not built")
    s = rs.load_sample(3, 0)
    per = s.groupby(["first_language", "second_language"]).size()
    assert len(per) == 15
    assert per[("Hindi", "English")] == 6 and per[("Spanish", "English")] == 3
    assert s.row_id.duplicated().sum() == 0


def test_review_disagreements_only_flag_language_vs_language():
    rs = _review_module()
    # "a" is Spanish by frequency; a stored English label there is a disagreement,
    # but an NE label never is.
    flags = rs._disagreements(["voy", "a", "Messi"], ["Spanish", "English", "NE"],
                              "Spanish", "English")
    assert flags == [1]


# --- names policy -----------------------------------------------------------

def test_names_keep_their_language_by_default_and_drop_when_transparent():
    from src.segments import tag_row
    cell = str(["hola amigo Messi friend"])
    mask = [False, False, True, False]
    lang = tag_row(cell, "Spanish", "English", is_entity=mask, return_tagged=True,
                   ane_labels=["notEnglish", "notEnglish", "notEnglish", "English"])
    assert "NE" not in [l for _, l in lang["tagged"]]
    assert lang["l2_runs"] == [1]
    trans = tag_row(cell, "Spanish", "English", is_entity=mask, names="transparent",
                    ane_labels=["notEnglish", "notEnglish", "notEnglish", "English"])
    assert trans["l2_runs"] == [1]


def test_a_title_made_of_english_words_counts_as_an_english_run():
    """"Anoche vi Game of Thrones con ..." - the title is English words."""
    from src.segments import tag_row
    cell = str(["Anoche vi Game of Thrones con amigos"])
    mask = [False, False, True, True, True, False, False]
    labs = ["notEnglish"] * 2 + ["English"] * 3 + ["notEnglish"] * 2
    assert tag_row(cell, "Spanish", "English", is_entity=mask, ane_labels=labs)["l2_runs"] == [3]
    assert tag_row(cell, "Spanish", "English", is_entity=mask, ane_labels=labs,
                   names="transparent")["l2_runs"] == []


def test_latin_name_in_a_cjk_row_takes_its_contextual_label():
    from src.dispatch import tag_tokens
    toks = ["我", "見", "Messi", "好", "Netflix"]
    mask = [False, False, True, False, False]
    labs = ["", "", "notEnglish", "", ""]
    tagged = dict(tag_tokens(toks, "Chinese", "English", ane_labels=labs, is_entity=mask))
    assert tagged["Messi"] == "Chinese"            # context says Chinese
    assert tagged["Netflix"] == "English"          # not an entity here: script stands
    plain = dict(tag_tokens(toks, "Chinese", "English"))
    assert plain["Messi"] == "English"             # without AnE labels: script


def test_han_script_names_stay_l1_even_with_an_english_label():
    from src.dispatch import tag_tokens
    tagged = tag_tokens(["中", "關", "村"], "Chinese", "English",
                        ane_labels=["English"] * 3, is_entity=[True] * 3)
    assert [l for _, l in tagged] == ["Chinese"] * 3


def test_name_windows_merge_overlaps_and_skip_non_latin_names():
    from src.persist import _name_windows
    toks = list("abcdefghij")
    mask = [False, True, False, True] + [False] * 5 + [True]
    assert _name_windows(toks, mask, 1) == [(0, 5), (8, 10)]
    assert _name_windows(["中", "文"], [True, True], 3) == []


def test_ane_pass_scores_whole_latin_rows_but_only_windows_in_script_rows(tmp_path, monkeypatch):
    import pandas as pd
    from src import ane_tagger, persist
    calls = []

    def fake(sentences, device=None):
        calls.extend(len(s) for s in sentences)
        return [["notEnglish"] * len(s) for s in sentences]

    monkeypatch.setattr(ane_tagger, "labels_batch", fake)
    df = pd.DataFrame({
        "data_generation_result": [str(["uno dos tres cuatro"]),
                                   str(["我" * 40 + " Messi " + "好" * 40])],
        "first_language": ["Spanish", "Chinese"],
        "second_language": ["English", "English"]})
    toks1 = persist.row_tokens(df.data_generation_result[1])
    ents = {0: [False] * 4, 1: [t == "Messi" for t in toks1]}
    out = persist.ane_labels(df, cache=str(tmp_path / "c.parquet"), entities=ents,
                             window=3, progress=False)
    assert out[0] == ["notEnglish"] * 4
    k = toks1.index("Messi")
    assert out[1][k] == "notEnglish" and out[1][0] == "" and out[1][-1] == ""
    assert sorted(calls) == [4, 7]                 # whole Latin row + one 7-token window


def test_transparent_policy_drops_entities_from_the_anticipation_stream():
    from src.anticipation import word_stream
    toks, langs = ["vi", "a", "Messi", "ayer"], ["Spanish"] * 4
    ents = [False, False, True, False]
    assert word_stream(toks, langs, "Spanish", "English", ents)[0] == toks
    assert word_stream(toks, langs, "Spanish", "English", ents,
                       names="transparent")[0] == ["vi", "a", "ayer"]


# --- hybrid names policy ----------------------------------------------------

def _hyb(toks, langs, ents, l1="French", l2="English", names="hybrid"):
    from src.segments import resolve_names
    return [l for _, l in resolve_names(list(zip(toks, langs)), ents, l1, l2, names)]


def test_opaque_names_tagged_english_inside_french_become_french():
    """The real row that motivated the rule: AnE called LOSC and Real Madrid English."""
    toks = "la victoire du LOSC contre le Real Madrid".split()
    langs = ["French"] * 3 + ["English"] + ["French"] * 2 + ["English"] * 2
    ents = [False] * 3 + [True] + [False] * 2 + [True] * 2
    assert _hyb(toks, langs, ents) == ["French"] * 8
    assert _hyb(toks, langs, ents, names="language") == langs


def test_titles_keep_their_language():
    toks = "Anoche vi Game of Thrones con amigos".split()
    langs = ["Spanish"] * 2 + ["English"] * 3 + ["Spanish"] * 2
    ents = [False] * 2 + [True] * 3 + [False] * 2
    assert _hyb(toks, langs, ents, "Spanish") == langs
    toks = "I watched La casa de papel yesterday".split()
    langs = ["English"] * 2 + ["Spanish"] * 4 + ["English"]
    ents = [False] * 2 + [True] * 4 + [False]
    assert _hyb(toks, langs, ents, "Spanish") == langs


def test_multiword_names_without_a_function_word_are_opaque():
    toks = "Ayer Taylor Swift cantó".split()
    langs = ["Spanish", "English", "English", "Spanish"]
    ents = [False, True, True, False]
    assert _hyb(toks, langs, ents, "Spanish") == ["Spanish"] * 4


def test_a_name_at_a_real_switch_keeps_the_side_its_tagger_chose():
    """"I think PSG va gagner": French is a neighbour, so PSG stays French."""
    toks = "I think PSG va gagner".split()
    langs = ["English", "English", "French", "French", "French"]
    ents = [False, False, True, False, False]
    assert _hyb(toks, langs, ents) == langs


def test_a_name_at_the_row_edge_takes_its_only_neighbour():
    toks = "Messi marcó dos goles".split()
    assert _hyb(toks, ["English", "Spanish", "Spanish", "Spanish"],
                [True, False, False, False], "Spanish") == ["Spanish"] * 4


def test_derived_policies_match_live_tagging(tmp_path):
    """derive_name_policies (for already-tagged parquets) must agree with tag_row."""
    import pandas as pd
    from src.persist import _tag_with_tokens, derive_name_policies
    cell = str(["Après la victoire du LOSC contre le Real Madrid , I'm eager to see"])
    mask = [False] * 4 + [True] + [False] * 2 + [True, True] + [False] * 5
    labs = ["notEnglish"] * 4 + ["English"] + ["notEnglish"] * 2 + ["English"] * 2 + \
           ["Other"] + ["English"] * 4
    live = _tag_with_tokens(cell, "French", "English", mask, labs)
    frame = pd.DataFrame([{**live, "first_language": "French", "second_language": "English"}])
    derived = derive_name_policies(frame.drop(columns=["l2_runs"])).iloc[0]
    # default (script_words): LOSC is French by its words, Real Madrid English by "real"
    assert list(derived.l2_runs) == live["l2_runs"] == [6]
    assert list(derived.l2_runs_script) == live["l2_runs_script"] == [1, 6]
    assert list(derived.l2_runs_transparent) == live["l2_runs_transparent"] == [4]


def test_a_name_at_a_switch_joins_the_side_not_cut_off_by_punctuation():
    toks = "contre le Real Madrid , I'm eager".split()
    langs = ["French", "French", "English", "English", "other", "English", "English"]
    ents = [False, False, True, True, False, False, False]
    assert _hyb(toks, langs, ents)[:4] == ["French"] * 4


# --- default policy and v1 conversion -----------------------------------------

def test_default_names_policy_is_script_words():
    import src.anticipation as ant
    from src.segments import DEFAULT_NAMES, tag_row
    assert DEFAULT_NAMES == "script_words" and ant.NAMES == DEFAULT_NAMES
    cell = str(["Anoche vi Severance Season 2 y me encantó"])
    mask = [False, False, True] + [False] * 5
    labs = ["notEnglish"] * 2 + ["English", "English", "Other"] + ["notEnglish"] * 3
    # hybrid would make the opaque "Severance Season" Spanish; script_words keeps English
    assert tag_row(cell, "Spanish", "English", is_entity=mask, ane_labels=labs)["l2_runs"] == [2]


def test_export_names_the_two_stored_alternatives():
    import pandas as pd
    from src.persist import _tag_with_tokens, export_rows_csv
    r = _tag_with_tokens(str(["Ayer Messi marcó , it was great"]), "Spanish", "English",
                         [False, True] + [False] * 5, None)
    assert {"l2_runs_transparent", "l2_runs_script"} <= set(r) and "l2_runs_as_tagged" not in r


def test_v1_labels_convert_to_policy_free_labels_plus_entity_flags():
    import pandas as pd
    from src.persist import from_v1
    v1 = pd.DataFrame([
        dict(row_id=0, first_language="Spanish", second_language="English", method="ane",
             tokens=["Ayer", "Messi", "marcó", ",", "it", "was", "great"],
             token_lang=["Spanish", "NE", "Spanish", "other", "English", "English", "English"]),
        dict(row_id=1, first_language="Cantonese", second_language="English", method="unicode",
             tokens=["我", "睇", "Severance", "Season", "好"],
             token_lang=["Cantonese", "Cantonese", "NE", "English", "Cantonese"]),
    ])
    out = from_v1(v1)
    assert not any("NE" in list(t) for t in out.token_lang)
    assert list(out.is_entity[0]) == [False, True] + [False] * 5
    assert list(out.l2_runs[0]) == [3]              # Messi stays Spanish (no evidence)
    assert list(out.l2_runs[1]) == [2]              # Latin script in Cantonese: English
    assert list(out.l2_runs_transparent[1]) == []   # repair makes "Season" part of the name
