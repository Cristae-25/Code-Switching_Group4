"""Script-based tagging, for language pairs whose scripts are disjoint.

Exact by construction: if the first language is written in Han and the second
in Latin, a token's script settles which language it belongs to. No model, no
threshold, nothing to validate beyond the script table itself.
"""
import regex as re

from .routing import expected_script

_SCRIPTS = ["Han", "Hiragana", "Katakana", "Hangul", "Devanagari",
            "Cyrillic", "Arabic", "Latin"]
_SCRIPT_RE = {name: re.compile(fr"\p{{Script={name}}}") for name in _SCRIPTS}


def char_script(ch: str) -> str:
    for name in _SCRIPTS:
        if _SCRIPT_RE[name].match(ch):
            return name
    return "other"


def tag_token_unicode(token: str) -> str:
    """Return the token's single script name, or ``"other"`` if mixed/unscripted.

    Combining marks are ignored when deciding, so an accented Latin token stays
    ``Latin`` rather than falling through to ``other``.
    """
    scripts = {char_script(c) for c in token if not c.isspace()}
    scripts.discard("other")
    if len(scripts) == 1:
        return scripts.pop()
    return "other"


def tag_tokens_unicode(tokens: list[str], first_language: str,
                       second_language: str) -> list[tuple[str, str]]:
    """Tag tokens as the row's first or second language by script.

    Returns ``(token, language)`` pairs, where language is the row's own
    ``first_language`` / ``second_language`` value, or ``"other"`` for
    punctuation and digits.
    """
    l1_scripts = expected_script(first_language)
    l2_scripts = expected_script(second_language)
    out = []
    for tok in tokens:
        script = tag_token_unicode(tok)
        if script in l1_scripts:
            out.append((tok, first_language))
        elif script in l2_scripts:
            out.append((tok, second_language))
        else:
            out.append((tok, "other"))
    return out
