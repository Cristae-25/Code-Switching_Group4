"""Visual inspection of tagged rows - the plan's "eyeball 20-30 rows" checklist item.

Renders each token tinted by the language it was tagged as, so a mis-tagged
span is obvious at a glance instead of buried in a list of tuples. Works both
in Jupyter (HTML) and a terminal (ANSI).
"""
from __future__ import annotations

import html

from .routing import route, tagger_quality
from .segments import absorb_other, parse_utterances
from .tokenizer import tokenize_mixed

# Categorical slots 1 and 2; identity is carried by the legend + the language
# name printed on each row, never by colour alone.
_L1_INK, _L1_BG = "#12467f", "#dce9f9"
_L2_INK, _L2_BG = "#8a3512", "#fce3d8"
_OTHER_INK = "#8a8a85"


def tag_tokens(text: str, first_language: str, second_language: str):
    """Tokenise + tag one string, routing by language pair."""
    from .dispatch import tag_tokens as _dispatch
    return _dispatch(tokenize_mixed(text), first_language, second_language)


def render_html(text: str, first_language: str, second_language: str,
                absorb: bool = False) -> str:
    """Return an HTML fragment with every token tinted by its tagged language.

    ``absorb=False`` shows the raw tags, so you can see which tokens were
    ``other`` before punctuation was folded into the neighbouring run.
    """
    tagged = tag_tokens(text, first_language, second_language)
    if absorb:
        tagged = absorb_other(tagged)

    spans = []
    for tok, lang in tagged:
        esc = html.escape(tok)
        if lang == first_language:
            spans.append(f'<span style="background:{_L1_BG};color:{_L1_INK};'
                         f'border-radius:3px;padding:1px 3px;margin:1px">{esc}</span>')
        elif lang == second_language:
            spans.append(f'<span style="background:{_L2_BG};color:{_L2_INK};'
                         f'border-radius:3px;padding:1px 3px;margin:1px;'
                         f'font-weight:600">{esc}</span>')
        else:
            spans.append(f'<span style="color:{_OTHER_INK};margin:1px">{esc}</span>')
    return " ".join(spans)


def show_row(row, max_utterances: int = 3, absorb: bool = False):
    """Display one dataframe row's utterances, colour-coded, in Jupyter.

    ``row`` is any object with ``data_generation_result``, ``first_language``
    and ``second_language`` attributes (e.g. from ``df.itertuples()``).
    """
    from IPython.display import HTML, display

    l1, l2 = row.first_language, row.second_language
    quality = tagger_quality(l1, l2)
    badge = {"exact": "#1baf7a", "provisional": "#eda100", "low": "#e34948"}[quality]
    header = (
        f'<div style="font:600 13px system-ui;margin:14px 0 4px">'
        f'{html.escape(l1)} &rarr; {html.escape(l2)} '
        f'<span style="background:{badge};color:#fff;border-radius:3px;'
        f'padding:1px 6px;font-size:11px;margin-left:6px">{quality}</span> '
        f'<span style="color:#8a8a85;font-weight:400">via {route(l1, l2)}</span></div>'
    )
    legend = (
        f'<div style="font:12px system-ui;color:#52514e;margin-bottom:6px">'
        f'<span style="background:{_L1_BG};color:{_L1_INK};padding:1px 5px;'
        f'border-radius:3px">{html.escape(l1)}</span> '
        f'<span style="background:{_L2_BG};color:{_L2_INK};padding:1px 5px;'
        f'border-radius:3px;font-weight:600">{html.escape(l2)}</span> '
        f'<span style="color:{_OTHER_INK}">other</span></div>'
    )
    body = "".join(
        f'<div style="font:13px/2.0 system-ui;margin:3px 0">'
        f'<span style="color:#b5b4ae;font-size:11px">[{i}]</span> '
        f'{render_html(utt, l1, l2, absorb=absorb)}</div>'
        for i, utt in enumerate(parse_utterances(row.data_generation_result)[:max_utterances])
    )
    display(HTML(header + legend + body))


def show_sample(frame, n_per_language: int = 1, random_state: int = 0, **kwargs):
    """Display one or more tagged rows per ``first_language`` for eyeballing."""
    # groupby().sample() keeps the grouping column; .apply() drops it in pandas 3.
    picks = frame.groupby("first_language", group_keys=False).sample(
        n=n_per_language, random_state=random_state)
    for row in picks.itertuples():
        show_row(row, **kwargs)


def render_ansi(text: str, first_language: str, second_language: str) -> str:
    """Terminal equivalent of ``render_html`` - L1 blue, L2 bold orange."""
    out = []
    for tok, lang in tag_tokens(text, first_language, second_language):
        if lang == first_language:
            out.append(f"\033[34m{tok}\033[0m")
        elif lang == second_language:
            out.append(f"\033[1;33m{tok}\033[0m")
        else:
            out.append(f"\033[90m{tok}\033[0m")
    return " ".join(out)
