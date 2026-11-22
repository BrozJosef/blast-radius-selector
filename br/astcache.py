"""One parse per source text: the analyses of a commit all look at the same two versions of a file,
and parsing the big modules costs a tenth of a second each time. Trees are shared read-only."""
import ast

_CACHE = {}
_ORDER = []
KEEP = 6


def parse(text):
    t = _CACHE.get(text)
    if t is None:
        t = ast.parse(text)
        _CACHE[text] = t
        _ORDER.append(text)
        if len(_ORDER) > KEEP:
            _CACHE.pop(_ORDER.pop(0), None)
    return t
