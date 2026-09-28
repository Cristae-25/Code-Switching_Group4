"""Full-dataset tagging run and the parquet handoff for Goal 1 / Goal 2.

Nested Python objects do not round-trip cleanly through parquet, so the two
ragged structures are stored as parallel primitive lists rather than lists of
tuples or mixed-type pairs:

    tokens        list[str]   every token, in order
    token_lang    list[str]   its tagged language, same length as tokens
    segment_lang  list[str]   run-length encoding: the language of each run
    segment_len   list[int]   ... and its length, same length as segment_lang

Both pairs are parquet-native (``list<string>`` / ``list<int32>``), lossless,
and zip straight back together - see ``load_tagged``.
"""
from __future__ import annotations

import os

import pandas as pd

from .routing import route, tagger_quality
from .segments import parse_utterances, tag_row
from .tokenizer import tokenize_mixed

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DEFAULT_PATH = os.path.join(DATA_DIR, "tagged.parquet")
ENTITY_PATH = os.path.join(DATA_DIR, "entities.parquet")
ANE_LID_PATH = os.path.join(DATA_DIR, "ane_lid.parquet")

# Kept out of the parquet: the source text is already in the CSVs, and copying
# it doubles the file for no gain. Join back on `row_id` if you need it.
_KEEP = ["first_language", "second_language", "topic", "cs_ratio", "cs_type",
         "cs_function", "conversation_type", "score"]


def _tag_with_tokens(cell, first_language: str, second_language: str,
                     is_entity: list[bool] | None = None,
                     ane_labels: list[str] | None = None) -> dict:
    """``tag_row`` plus the flat token/language sequences Goal 1 needs.

    ``l2_runs`` & co. follow the default names policy (``script_words``);
    ``l2_runs_transparent`` and ``l2_runs_script`` are the stored ablations.
    ``token_lang`` itself is policy-free: the tagger's language for every
    word, names included.
    """
    result = tag_row(cell, first_language, second_language, is_entity=is_entity,
                     return_tagged=True, ane_labels=ane_labels)
    tagged = result.pop("tagged", [])
    result["tokens"] = [t for t, _ in tagged]
    result["token_lang"] = [l for _, l in tagged]
    mask = list(is_entity) if is_entity is not None else [False] * len(tagged)
    result["is_entity"] = mask
    result.update(_policy_runs(tagged, mask, first_language, second_language))
    result["segment_lang"] = [s[0] for s in result["segments"]]
    result["segment_len"] = [int(s[1]) for s in result["segments"]]
    del result["segments"]
    return result


def _policy_runs(tagged, mask, l1: str, l2: str) -> dict:
    from .segments import absorb_other, l2_word_runs, resolve_names
    return {f"l2_runs_{col}": l2_word_runs(absorb_other(resolve_names(tagged, mask, l1, l2, pol)), l2)
            for col, pol in (("transparent", "transparent"), ("script", "script"))}


def derive_name_policies(frame: pd.DataFrame) -> pd.DataFrame:
    """Recompute every names-policy column from ``tokens``/``token_lang``/``is_entity``.

    No model is involved, so this takes about a minute for the corpus. The
    default-policy columns - l2_runs, n_switches, max_run, total_l2_tokens,
    segment_lang/len - are overwritten; token_lang is left untouched.
    """
    from .segments import DEFAULT_NAMES, absorb_other, l2_word_runs, resolve_names, to_segments

    out = frame.copy()
    cols: dict[str, list] = {k: [] for k in ("l2_runs", "l2_runs_transparent", "l2_runs_script",
                                            "segment_lang", "segment_len")}
    for t, l, e, l1, l2 in zip(out.tokens, out.token_lang, out.is_entity,
                               out.first_language, out.second_language):
        tagged, mask = list(zip(t, l)), list(e)
        hyb = absorb_other(resolve_names(tagged, mask, l1, l2, DEFAULT_NAMES))
        cols["l2_runs"].append(l2_word_runs(hyb, l2))
        segs = to_segments(hyb)
        cols["segment_lang"].append([s[0] for s in segs])
        cols["segment_len"].append([int(s[1]) for s in segs])
        for k, v in _policy_runs(tagged, mask, l1, l2).items():
            cols[k].append(v)
    for k, v in cols.items():
        out[k] = v
    out["n_switches"] = [len(r) for r in out.l2_runs]
    out["max_run"] = [max(r) if len(r) else 0 for r in out.l2_runs]
    out["total_l2_tokens"] = [int(sum(r)) for r in out.l2_runs]
    return out


def from_v1(frame: pd.DataFrame) -> pd.DataFrame:
    """Turn v1 labels (names as ``NE``) into policy-free labels + ``is_entity``.

    v1 kept AnE-NER's name positions but not a language for the names, so one
    is reconstructed: the script tagger's label in script-tagged rows, dict-freq
    in the rest. The default policy barely consults it - in non-Latin rows a
    name's language comes from its script, in Latin rows from its words or its
    neighbours - it only decides a name sitting on an unpunctuated switch.
    Every non-name token is unchanged. Then all policy columns are derived.
    """
    from .dict_tagger import tag_frequency
    from .segments import NE
    from .unicode_tagger import tag_tokens_unicode

    out = frame.copy()
    langs, ents = [], []
    for toks, labs, l1, l2, method in zip(out.tokens, out.token_lang, out.first_language,
                                          out.second_language, out.method):
        toks, labs = list(toks), list(labs)
        ent = [x == NE for x in labs]
        if any(ent):
            fill = (tag_tokens_unicode(toks, l1, l2) if method == "unicode"
                    else tag_frequency(toks, l1, l2))
            labs = [f if e else x for x, (_, f), e in zip(labs, fill, ent)]
        langs.append(labs)
        ents.append(ent)
    out["token_lang"], out["is_entity"] = langs, ents
    return derive_name_policies(out)


def row_tokens(cell) -> list[str]:
    """The token sequence ``tag_row`` will see for one row."""
    tokens: list[str] = []
    for utt in parse_utterances(cell):
        tokens.extend(tokenize_mixed(utt))
    return tokens


def _cached_pass(df: pd.DataFrame, cache: str, column: str, fn, desc: str,
                 batch: int = 32, progress: bool = True,
                 select=None) -> dict[int, list]:
    """Run a batched per-word model ``fn`` over rows, cached by ``row_id``.

    ``fn`` takes ``[(row_id, tokens), ...]`` and returns one per-token list per
    row. ``row_id`` is ``df``'s index. Rows already cached are skipped, so an
    interrupted run resumes; a cached entry whose length no longer matches the
    row's tokens (tokenizer change) is recomputed. ``select(row, tokens)``
    restricts which rows need the pass at all.
    """
    os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
    done: dict[int, list] = {}
    if os.path.exists(cache):
        c = pd.read_parquet(cache)
        done = {int(r): list(m) for r, m in zip(c.row_id, c[column])}

    todo = []
    for row in df.itertuples():
        toks = row_tokens(row.data_generation_result)
        if select is not None and not select(row, toks):
            continue
        if len(done.get(int(row.Index), ())) != len(toks):
            todo.append((int(row.Index), toks))

    bar = None
    if progress and todo:
        try:
            from tqdm.auto import tqdm
            bar = tqdm(total=len(todo), desc=desc)
        except ImportError:
            pass
    # Sort by length so each batch pads to similar sizes; flush every ~2k rows.
    todo.sort(key=lambda x: len(x[1]))
    since_flush = 0
    for b in range(0, len(todo), batch):
        chunk = todo[b:b + batch]
        for (rid, _), out in zip(chunk, fn(chunk)):
            done[rid] = out
        since_flush += len(chunk)
        if bar:
            bar.update(len(chunk))
        if since_flush >= 2048 or b + batch >= len(todo):
            pd.DataFrame({"row_id": list(done), column: list(done.values())}
                         ).to_parquet(cache, index=False)
            since_flush = 0
    if bar:
        bar.close()
    return done


def entity_masks(df: pd.DataFrame, cache: str = ENTITY_PATH, device: str | None = None,
                 **kw) -> dict[int, list[bool]]:
    """AnE-NER mask per row, cached in ``data/entities.parquet``.

    This is the one expensive model pass *every* row needs (~8 h on MPS), so it
    is stored separately: a tagger or routing change re-tags without redoing it.
    """
    from .ane_tagger import entity_mask_batch
    return _cached_pass(df, cache, "is_entity",
                        lambda items: entity_mask_batch([t for _, t in items], device=device),
                        "entities", **kw)


def _name_windows(tokens: list[str], mask: list[bool], window: int) -> list[tuple[int, int]]:
    """Merged ``[start, end)`` spans of +-``window`` tokens around Latin-script names."""
    import regex as re
    latin = re.compile(r"\p{Script=Latin}")
    spans: list[list[int]] = []
    for k, (t, ent) in enumerate(zip(tokens, mask)):
        if ent and latin.search(t):
            a, b = max(0, k - window), min(len(tokens), k + window + 1)
            if spans and a <= spans[-1][1]:
                spans[-1][1] = max(spans[-1][1], b)
            else:
                spans.append([a, b])
    return [(a, b) for a, b in spans]


def ane_labels(df: pd.DataFrame, cache: str = ANE_LID_PATH, device: str | None = None,
               entities: dict[int, list[bool]] | None = None, window: int = 16,
               **kw) -> dict[int, list[str]]:
    """Raw AnE-LID word labels, cached on disk.

    * ``ane`` rows: every token (these labels *are* the row's tags).
    * script-tagged rows with a Latin-script entity (needs ``entities``): only
      +-``window`` tokens around each such name, so its label comes from real
      context without paying for the whole row; other positions are ``""``.
      This is what lets "Messi" in a Cantonese sentence stop being English.
    """
    from .ane_tagger import labels_batch
    from .dispatch import row_strategy

    ents = entities or {}
    windows: dict[int, list[tuple[int, int]]] = {}   # script rows -> name windows

    def needs(row, toks):
        s = row_strategy(toks, row.first_language, row.second_language)
        if s == "ane":
            return True
        m = ents.get(int(row.Index))
        if s != "unicode" or m is None:
            return False
        spans = _name_windows(toks, m, window)
        if spans:
            windows[int(row.Index)] = spans
        return bool(spans)

    def fn(items):
        jobs, where = [], []           # one model input per full row or window
        out = []
        for i, (rid, toks) in enumerate(items):
            for a, b in windows.get(rid, [(0, len(toks))]):   # ane rows: whole row
                jobs.append(toks[a:b])
                where.append((i, a))
            out.append([""] * len(toks))
        for (i, a), labs in zip(where, labels_batch(jobs, device=device)):
            out[i][a:a + len(labs)] = labs
        return out

    return _cached_pass(df, cache, "ane_label", fn, "ane-lid", select=needs, **kw)


def tag_dataframe(df: pd.DataFrame, progress: bool = True,
                  entities: dict[int, list[bool]] | None = None,
                  ane: dict[int, list[str]] | None = None) -> pd.DataFrame:
    """Tag every row. Returns a frame aligned to ``df`` by position.

    ``entities`` (from :func:`entity_masks`) relabels named-entity words ``NE``;
    without it, names get whatever language the tagger assigned. ``ane`` (from
    :func:`ane_labels`) supplies precomputed AnE-LID labels; rows missing from
    it run the model inline, one row at a time.
    """
    rows = df.itertuples()
    if progress:
        try:
            from tqdm.auto import tqdm
            rows = tqdm(rows, total=len(df), desc="tagging")
        except ImportError:
            pass

    ents, lid = entities or {}, ane or {}
    out = pd.DataFrame([
        _tag_with_tokens(r.data_generation_result, r.first_language, r.second_language,
                         ents.get(int(r.Index)), lid.get(int(r.Index)))
        for r in rows
    ])
    out.insert(0, "row_id", df.index.to_numpy())
    for col in _KEEP:
        if col in df.columns:
            out[col] = df[col].to_numpy()
    out["quality"] = [tagger_quality(a, b) for a, b in
                      zip(out.first_language, out.second_language)]
    return out


# One code per token in the CSV export. Language names vary per row, so the
# codes are relative to the row's own pair - the same binarisation the
# anticipation task uses. Names carry a language code like any other word,
# resolved with the default hybrid policy; entity-ness is the ``entities`` column.
LABEL_CODES = "L1 = first_language, L2 = second_language, O = punctuation/digits/other"


def _code(lang: str, l1: str, l2: str) -> str:
    if lang == l1:
        return "L1"
    if lang == l2:
        return "L2"
    return "O"


def export_rows_csv(frame: pd.DataFrame, path: str) -> str:
    """One CSV line per corpus row, tokens and labels as aligned space-joined strings.

    Tokens never contain whitespace (see ``tokenizer``), so
    ``row.tokens.split(" ")`` and ``row.labels.split(" ")`` zip back together.
    ``entities`` is an aligned 0/1 string (1 = part of a named entity).
    ``labels`` and ``l2_runs`` follow the default names policy (script_words);
    ``l2_runs`` are the second-language run lengths in words - the "how long
    does a switch last" targets - with punctuation transparent.
    ``l2_runs_names_transparent`` / ``l2_runs_names_script`` are the stored
    alternatives (see ``segments.resolve_names``).
    ``split`` is the row-level train/val/test assignment used everywhere else
    (``row_id % 10``: 0-7 train, 8 val, 9 test).
    """
    from .segments import resolve_names

    split = {**{i: "train" for i in range(8)}, 8: "val", 9: "test"}
    ents = (list(frame.is_entity) if "is_entity" in frame.columns
            else [[False] * len(t) for t in frame.tokens])
    out = pd.DataFrame({
        "row_id": frame.row_id,
        "first_language": frame.first_language,
        "second_language": frame.second_language,
        "split": [split[int(r) % 10] for r in frame.row_id],
        "quality": frame.quality,
        "method": frame.method,
        "n_tokens": [len(t) for t in frame.tokens],
        "tokens": [" ".join(t) for t in frame.tokens],
        "labels": [" ".join(_code(l, a, b) for _, l in resolve_names(list(zip(t, ls)), e, a, b))
                   for t, ls, e, a, b in zip(frame.tokens, frame.token_lang, ents,
                                             frame.first_language, frame.second_language)],
        "entities": [" ".join("1" if e else "0" for e in m) for m in ents],
        "l2_runs": [" ".join(str(int(x)) for x in r) for r in frame.l2_runs],
        "l2_runs_names_script": [
            " ".join(str(int(x)) for x in r) for r in
            (frame.l2_runs_script if "l2_runs_script" in frame.columns
             else frame.l2_runs)],
        "l2_runs_names_transparent": [
            " ".join(str(int(x)) for x in r) for r in
            (frame.l2_runs_transparent if "l2_runs_transparent" in frame.columns
             else frame.l2_runs)],
        "n_switches": frame.n_switches,
        "max_run": frame.max_run,
        "total_l2_tokens": frame.total_l2_tokens,
        "l2_token_ratio": frame.l2_token_ratio.round(4),
    })
    for c in ("topic", "cs_type", "cs_function", "conversation_type"):
        if c in frame.columns:
            out[c] = frame[c].to_numpy()
    out.to_csv(path, index=False)
    return path


def save_tagged(frame: pd.DataFrame, path: str = DEFAULT_PATH) -> str:
    """Write the tagged frame to parquet, creating ``data/`` if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frame.to_parquet(path, index=False, compression="snappy")
    return path


def load_tagged(path: str = DEFAULT_PATH) -> pd.DataFrame:
    """Read back a tagged parquet, restoring ``segments`` and ``tokens_tagged``."""
    frame = pd.read_parquet(path)
    frame["segments"] = [
        [[l, int(n)] for l, n in zip(ls, ns)]
        for ls, ns in zip(frame.segment_lang, frame.segment_len)
    ]
    frame["tokens_tagged"] = [
        list(zip(ts, ls)) for ts, ls in zip(frame.tokens, frame.token_lang)
    ]
    return frame


def recompute_runs(path: str = DEFAULT_PATH) -> pd.DataFrame:
    """Recompute ``l2_runs`` from stored tokens, without re-tagging.

    Because the parquet keeps the full ``tokens``/``token_lang`` sequences, a
    change to the run-length rule does not require another pass over the
    models - the expensive part is the tagging, not the counting.
    """
    from .segments import absorb_other, l2_word_runs

    # `token_lang` is stored pre-absorption, but tag_row counts post-absorption.
    # Absorb here too, or the two paths disagree wherever a *word* carries the
    # `other` label (BERT emits it for named entities), which would split runs
    # that tag_row keeps whole.
    frame = pd.read_parquet(path)
    runs = [
        l2_word_runs(absorb_other(list(zip(ts, ls))), l2)
        for ts, ls, l2 in zip(frame.tokens, frame.token_lang, frame.second_language)
    ]
    frame["l2_runs"] = runs
    frame["n_switches"] = [len(r) for r in runs]
    frame["max_run"] = [max(r) if len(r) else 0 for r in runs]
    frame["total_l2_tokens"] = [int(sum(r)) for r in runs]
    frame.to_parquet(path, index=False, compression="snappy")
    return frame
