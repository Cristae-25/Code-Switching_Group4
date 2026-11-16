"""Validation helpers: compare tagger output against the dataset's own ratios.

Step 0 of the plan asked whether per-span labels survive upstream. They do not -
but ``cs_ratio_result`` carries a per-row aggregate the generator computed
(``{'computed_ratio': '60% Spanish : 40% English', ...}``). That is not
token-level truth, so it cannot score the tagger directly; it is a strong
*correlation* check that catches a systematically broken language pair.
"""
from __future__ import annotations

import ast

import regex as re

_PCT = re.compile(r"(\d+(?:\.\d+)?)\s*%\s*([A-Za-z]+)")


def parse_computed_ratio(cell, second_language: str) -> float | None:
    """Extract the second language's share (0-1) from ``cs_ratio_result``."""
    try:
        value = ast.literal_eval(str(cell))
        text = value.get("computed_ratio", "") if isinstance(value, dict) else str(value)
    except (ValueError, SyntaxError):
        text = str(cell)
    pairs = _PCT.findall(text)
    if not pairs:
        return None
    target = second_language.lower()
    for pct, name in pairs:
        if name.lower().startswith(target[:4]):
            return float(pct) / 100.0
    # Fall back to "everything that is not the first language" when the second
    # language is named differently (e.g. "Mandarin" vs "Chinese").
    if len(pairs) == 2:
        return float(pairs[1][0]) / 100.0
    return None
