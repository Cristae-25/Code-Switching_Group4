"""Mixed-granularity tokenizer.

CJK does not tokenize on whitespace - a run of Han/Kana/Hangul between two
English words may hold several words with no space - so those scripts are cut
per character. Everything else is cut per word.

Two fixes over the pattern in the plan:

1. ``[A-Za-z]+`` splits accented Latin: "dificil" became ``dif|i|cil``, which
   shatters every Spanish/French/German/Vietnamese word carrying a diacritic
   and would badly inflate run counts. ``\\p{Script=Latin}`` covers the accented
   range, since U+00E9 and friends *are* Latin script.
2. Cyrillic, Arabic and Devanagari are space-delimited, so word-level is right
   for them; the plan's fallback branch would have cut them per character.
"""
import regex as re

# Order matters: character-level CJK alternatives are tried before the
# word-level ones, and the single-character fallback is last.
_PATTERN = re.compile(
    r"\p{Script=Han}"
    r"|\p{Script=Hiragana}"
    r"|\p{Script=Katakana}"
    r"|\p{Script=Hangul}"
    r"|[\p{Script=Latin}\p{M}]+(?:['’][\p{Script=Latin}\p{M}]+)*"
    r"|[\p{Script=Cyrillic}\p{M}]+"
    r"|[\p{Script=Arabic}\p{M}]+"
    r"|[\p{Script=Devanagari}\p{M}]+"
    r"|\d+"
    r"|[^\s]"
)


def tokenize_mixed(text: str) -> list[str]:
    """Split text into CJK-character tokens and word tokens for other scripts."""
    return _PATTERN.findall(text)
