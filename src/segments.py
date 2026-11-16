"""Run-length encoding of tagged tokens, plus the dataset builders.

The plan leaves one decision open (line 174): whether an ``"other"`` tag
(punctuation, digits) breaks a run or is absorbed into the surrounding
language. Absorbing is cleaner for the downstream goals, so ``absorb_other``
does that - with one guard the naive version gets wrong: an ``other`` token
sitting *between two different languages* must not bridge them into a single
run. "hola, hello" is two runs, not one.

Named entities: every token keeps a *language* label, and entity-ness travels
separately as an ``is_entity`` flag. How names count is a policy, not a label
(``resolve_names``; every edge case is pinned in tests/test_name_strategies.py).
AnE-NER fragments names, so ``repair_spans`` mends them first for every policy.

* ``names="script_words"`` (default) - a name is a switch only with visible
  evidence. A Latin-script name in a non-Latin row is English ("Messi",
  "Severance Season 2" inside Arabic), since the writer chose the Latin script;
  a name in the row's own script stays L1. In Latin-script rows the name's
  words decide when they are clearly one language's ("Severance Season" ->
  English, "Union europeenne" -> French); otherwise it takes its surroundings
  ("Ayer Leo Messi marco" -> Spanish), and may not open a run on its own.
* ``names="transparent"`` - names are no language and are skipped (LinCE's
  convention). Stored as the main ablation.
* ``names="script"`` - every Latin-script name is English. Stored too; adds a
  one-word switch for each Messi / Monza / EU inside a Latin-script sentence.
* ``names="hybrid"`` - titles keep their language, other names take context.
* ``names="language"`` - names count as AnE-LID tagged them.
"""
from __future__ import annotations

import ast

import regex as re

from .routing import route
from .tokenizer import tokenize_mixed

NE = "NE"
NAME_POLICIES = ("hybrid", "language", "transparent", "script", "script_words")
# A name counts as a switch only with visible evidence: a change of script, or
# words that are clearly one language's. See tests/test_name_strategies.py.
DEFAULT_NAMES = "script_words"

# A function word: short and extremely frequent in its language (wordfreq Zipf).
# Measured: of/the/and/de/la/der/dan/ng/cua all score 6.5-7.9; the most common
# content words in candidate names (game, real, united, league) stay <= 5.8.
_FUNCTION_ZIPF = 6.4
_FUNCTION_MAXLEN = 4


def _is_function_word(word: str, language: str) -> bool:
    from .dict_tagger import LANG_TO_CODE, zipf
    code = LANG_TO_CODE.get(language)
    return (code is not None and len(word) <= _FUNCTION_MAXLEN
            and zipf(word.lower(), code) >= _FUNCTION_ZIPF)


_LATIN = re.compile(r"\p{Script=Latin}")
# Languages whose wordfreq list is written in Latin script, so a Latin-script
# name can be looked up in it. Hindi's list is Devanagari: romanized names
# score ~0 there, which would make every Bollywood name "English".
_LATIN_DICT = {"English", "Spanish", "German", "French", "Italian", "Malay",
               "Philippines", "Vietnamese"}
_MARGIN = 1.0      # Zipf gap for a word to count as clearly one language's


def _latin(tok: str) -> bool:
    return bool(_LATIN.search(tok))


def _zipf(word: str, language: str):
    if language not in _LATIN_DICT:
        return None
    from .dict_tagger import LANG_TO_CODE, zipf
    return zipf(word.lower(), LANG_TO_CODE[language])


def _word_side(word: str, first_language: str, second_language: str):
    """Which of the pair a Latin word clearly belongs to, by dictionary margin; None if unclear.

    Names sit in both frequency lists at similar levels (messi 3.8 en / 4.1 es,
    madrid 4.3 / 5.4), so only a >= 1.0 Zipf gap decides: severance, season ->
    English; europeenne, monza -> L1; messi, psg, eu -> unclear.
    """
    z1, z2 = _zipf(word, first_language), _zipf(word, second_language)
    if z1 is None or z2 is None:
        return None
    if z2 - z1 >= _MARGIN:
        return second_language
    if z1 - z2 >= _MARGIN:
        return first_language
    return None


def repair_spans(tokens, is_entity, first_language: str, second_language: str) -> list[bool]:
    """Mend AnE-NER's fragmented Latin-script names before any policy uses them.

    Measured on real output: "Severance Season 2" came back as just
    "Severance", "Game of Thrones" as "Game" + "Thrones", "The University of
    Texas" as "of Texas". Two repairs, Latin script only:

    * bridge a single function word between two name words ("Game [of] Thrones");
    * grow a name over adjacent capitalised words - and numbers to its right -
      unless the word is clearly an ordinary L1 word ("Ayer Leo Messi" keeps
      "Ayer"; German nouns like "Kommission" are never swallowed).
    Non-Latin names are left exactly as tagged.
    """
    mask = list(is_entity)
    n = len(tokens)
    for k in range(1, n - 1):
        if (not mask[k] and mask[k - 1] and mask[k + 1]
                and _latin(tokens[k - 1]) and _latin(tokens[k + 1])
                and any(_is_function_word(tokens[k], lang)
                        for lang in {first_language, second_language, "English"})):
            mask[k] = True

    def joins(tok: str, to_the_right: bool) -> bool:
        if to_the_right and tok.isdigit():
            return True
        return (_latin(tok) and tok[:1].isupper()
                and _word_side(tok, first_language, second_language) != first_language)

    changed = True
    while changed:
        changed = False
        for k in range(n):
            if mask[k]:
                continue
            after_name = k > 0 and mask[k - 1] and (_latin(tokens[k - 1]) or tokens[k - 1].isdigit())
            before_name = k + 1 < n and mask[k + 1] and _latin(tokens[k + 1])
            if (after_name and joins(tokens[k], True)) or (before_name and joins(tokens[k], False)):
                mask[k] = changed = True
    return mask


def _context_language(out, ent, i, j, own, langs):
    """An opaque name (tokens i..j-1) may not open a run of its own.

    Inside one language it takes that language. At a real switch it joins the
    side not cut off by punctuation; if neither or both are, it keeps ``own``
    when that is one of the two sides.
    """
    n = len(out)

    def neighbour(k, step):
        k += step
        cut = False
        while 0 <= k < n:
            t, lang = out[k]
            if not _is_word(t):
                cut = True
            elif not ent[k] and lang in langs:
                return lang, cut
            k += step
        return None, cut

    (prev, prev_cut), (nxt, nxt_cut) = neighbour(i, -1), neighbour(j - 1, 1)
    if prev and nxt and prev != nxt:
        if prev_cut != nxt_cut:
            return nxt if prev_cut else prev
        return own if own in (prev, nxt) else prev
    return prev or nxt or own


def _spans(ent):
    i, n = 0, len(ent)
    while i < n:
        if not ent[i]:
            i += 1
            continue
        j = i
        while j < n and ent[j]:
            j += 1
        yield i, j
        i = j


def resolve_names(tagged: list[tuple[str, str]], is_entity, first_language: str,
                  second_language: str, names: str = DEFAULT_NAMES) -> list[tuple[str, str]]:
    """Apply a names policy to language-tagged tokens (see the module docstring)."""
    if is_entity is None or names == "language":
        return list(tagged)
    if names not in NAME_POLICIES:
        raise ValueError(f"unknown names policy {names!r}; use one of {NAME_POLICIES}")
    toks = [t for t, _ in tagged]
    ent = repair_spans(toks, is_entity, first_language, second_language)
    if names == "transparent":
        return apply_entities(tagged, ent)

    from .dispatch import _hindi_is_devanagari
    from .routing import SCRIPT_DISJOINT_CONFIRMED
    script_row = (first_language in SCRIPT_DISJOINT_CONFIRMED
                  or (first_language == "Hindi" and _hindi_is_devanagari(toks)))
    langs = (first_language, second_language)
    out = list(tagged)
    for i, j in _spans(ent):
        words = [k for k in range(i, j) if _is_word(out[k][0])]
        tagged_langs = [out[k][1] for k in words if out[k][1] in langs]
        if not tagged_langs:
            continue
        own = max(langs, key=tagged_langs.count)
        latin = any(_latin(out[k][0]) for k in words)

        if names == "hybrid":
            title = len(words) >= 2 and any(_is_function_word(out[k][0], own) for k in words)
            if not title:
                own = _context_language(out, ent, i, j, own, langs)
        elif names == "script":
            if latin and second_language == "English":
                own = second_language
        elif names == "script_words" and latin:
            if script_row:
                if second_language == "English":
                    own = second_language
            else:
                votes = [_word_side(out[k][0], first_language, second_language) for k in words]
                votes = [v for v in votes if v]
                if votes and votes.count(langs[0]) != votes.count(langs[1]):
                    own = max(langs, key=votes.count)
                else:
                    own = _context_language(out, ent, i, j, own, langs)
        for k in words:
            out[k] = (out[k][0], own)
    return out

def apply_entities(tagged: list[tuple[str, str]],
                   is_entity: list[bool]) -> list[tuple[str, str]]:
    """The ``transparent`` policy: relabel entity *words* ``NE``.

    Punctuation inside a name stays ``other``.
    """
    return [(t, NE if ent and _is_word(t) else lang)
            for (t, lang), ent in zip(tagged, is_entity)]


def absorb_other(tagged: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Reassign ``other`` tokens to the surrounding language.

    An ``other`` run takes the language of the token before it, unless that
    would merge two different languages - at a genuine switch boundary the
    punctuation joins the *following* language instead, so the boundary stays
    where it is. Leading ``other`` tokens join whatever follows. Token count is
    preserved.
    """
    out = list(tagged)
    n = len(out)
    i = 0
    while i < n:
        if out[i][1] != "other":
            i += 1
            continue
        j = i
        while j < n and out[j][1] == "other":
            j += 1
        # Look through NE tokens: they belong to neither neighbour, so the
        # boundary test must be made against the nearest real language.
        p = i - 1
        while p >= 0 and out[p][1] == NE:
            p -= 1
        q = j
        while q < n and out[q][1] in ("other", NE):
            q += 1
        prev_lang = out[p][1] if p >= 0 else None
        next_lang = out[q][1] if q < n else None
        if prev_lang is None:
            fill = next_lang
        elif next_lang is None or prev_lang == next_lang:
            fill = prev_lang
        else:
            # Genuine switch boundary: do not let punctuation bridge it.
            fill = next_lang
        if fill is not None:
            for k in range(i, j):
                out[k] = (out[k][0], fill)
        i = j
    return out


def to_segments(tagged_tokens: list[tuple[str, str]]) -> list[list]:
    """Collapse consecutive same-language tokens into ``[language, length]`` runs."""
    segments: list[list] = []
    for _, lang in tagged_tokens:
        if segments and segments[-1][0] == lang:
            segments[-1][1] += 1
        else:
            segments.append([lang, 1])
    return segments


def l2_runs(segments: list[list], second_language: str) -> list[int]:
    """Run lengths straight off the segments, punctuation included.

    Prefer :func:`l2_word_runs` for the switch-length question - segments carry
    absorbed punctuation, which inflates every run.
    """
    return [n for lang, n in segments if lang == second_language]


def _is_word(token: str) -> bool:
    return any(ch.isalpha() for ch in token)


def l2_word_runs(tagged: list[tuple[str, str]], second_language: str) -> list[int]:
    """Lengths of each consecutive second-language run, counting WORDS only.

    This is the original question: how many second-language tokens are used
    consecutively before switching back or ending.

    Punctuation is *transparent* - it neither breaks a run nor counts toward
    it. That matters because ``absorb_other`` hands boundary punctuation to the
    following language, so a sentence-final "." belonging to the first-language
    sentence would otherwise be counted as a second-language token and inflate
    every run by one.
    """
    runs, current = [], 0
    for token, lang in tagged:
        if not _is_word(token) or lang == NE:
            continue                      # transparent: no break, no count
        if lang == second_language:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


def parse_utterances(cell: str) -> list[str]:
    """``data_generation_result`` is a stringified Python list of utterances."""
    try:
        value = ast.literal_eval(str(cell))
    except (ValueError, SyntaxError):
        return [str(cell)]
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def tag_row(cell: str, first_language: str, second_language: str,
            cross_utterance: bool = True, is_entity: list[bool] | None = None,
            return_tagged: bool = False, ane_labels: list[str] | None = None,
            names: str = DEFAULT_NAMES) -> dict:
    """Tokenise, tag and run-length-encode one dataframe row.

    ``cross_utterance=True`` streams every utterance together, so a run may span
    a turn boundary; the utterance each run starts in is still recorded.
    """
    from .dispatch import row_strategy, tag_tokens

    utterances = parse_utterances(cell)

    tokens: list[str] = []
    origin: list[int] = []
    for idx, utt in enumerate(utterances):
        toks = tokenize_mixed(utt)
        tokens.extend(toks)
        origin.extend([idx] * len(toks))

    if not tokens:
        return dict(segments=[], l2_runs=[], n_switches=0, max_run=0,
                    total_l2_tokens=0, total_tokens=0, l2_token_ratio=0.0,
                    method=route(first_language, second_language),
                    unknown_rate=0.0, run_utterance_idx=[])

    strategy = row_strategy(tokens, first_language, second_language)
    tagged = tag_tokens(tokens, first_language, second_language,
                        ane_labels=ane_labels, is_entity=is_entity)
    lang_tagged = tagged                  # stored labels: always a language, policy-free
    tagged = resolve_names(tagged, is_entity, first_language, second_language, names)
    if strategy == "fasttext":
        from .model_tagger import unknown_rate
        unk = unknown_rate(tokens, first_language, second_language)
    else:
        unk = 0.0

    if not cross_utterance:
        # Break runs at turn boundaries by inserting a sentinel language.
        tagged = [
            t if i == 0 or origin[i] == origin[i - 1] else (t[0], t[1])
            for i, t in enumerate(tagged)
        ]

    tagged = absorb_other(tagged)
    segments = to_segments(tagged)
    runs = l2_word_runs(tagged, second_language)

    # Utterance index where each second-language run begins.
    run_start_idx: list[int] = []
    pos = 0
    for lang, n in segments:
        if lang == second_language:
            run_start_idx.append(origin[pos] if pos < len(origin) else -1)
        pos += n

    total_l2 = sum(runs)
    counted = sum(n for lang, n in segments if lang not in ("other", NE))
    result = dict(
        segments=segments,
        l2_runs=runs,
        n_switches=len(runs),
        max_run=max(runs) if runs else 0,
        total_l2_tokens=total_l2,
        total_tokens=len(tokens),
        l2_token_ratio=(total_l2 / counted) if counted else 0.0,
        method=strategy,
        unknown_rate=unk,
        run_utterance_idx=run_start_idx,
    )
    if return_tagged:
        result["tagged"] = lang_tagged   # pre-absorption, never NE
    return result
