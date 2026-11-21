"""Text helpers shared by the agent: tokenization and chunking."""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"\w+", re.UNICODE)

STOPWORDS = frozenset("""
a an and are as at be but by can could do does did for from had has have he her
hers him his how i if in into is it its just like may me might more most my no
nor not of on or our out over said same she should so some such than that the
their them then there these they this those through to too under up very was we
were what when where which while who whom why will with would you your
""".split())


def words(text: str) -> list[str]:
    """Lowercase word tokens without stopwords or single characters."""
    return [w for w in _WORD_RE.findall(text.lower()) if len(w) > 1 and w not in STOPWORDS]
