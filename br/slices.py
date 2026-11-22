"""Fixture slicing: let a fixture-driven test run only selected statement pairs.

Tests load `tests/fixtures/*.sql` through tests.helpers.load_sql_fixture_pairs / load_sql_fixtures and
loop over the entries independently, so a test restricted to the entries that can observe a change
gives the same verdict as the whole test at a fraction of the cost.
"""
import sys

ALLOW = {}      # loader filename -> set of entry indices (absent = everything)
OVERRIDE = {}   # loader filename -> alternative fixtures directory holding a patched copy
SEEN = {}       # loader filename -> number of entries yielded by the unsliced loader (for discovery)
_orig = {}


def _wrap(name, fn):
    def wrapped(filename, *a, **k):
        allow = ALLOW.get(filename)
        n = 0
        if filename in OVERRIDE and not a and not k:
            src = entries(filename, name, OVERRIDE[filename])
        else:
            src = fn(filename, *a, **k)
        for i, item in enumerate(src):
            n = i + 1
            if allow is None or i in allow:
                yield item
        SEEN[filename] = n
    wrapped.__name__ = name
    wrapped.__br_wrapped__ = fn
    return wrapped


def install():
    """Wrap the loaders in tests.helpers and rebind them in every imported tests.* module."""
    import tests.helpers as h
    for name in ("load_sql_fixture_pairs", "load_sql_fixtures"):
        fn = getattr(h, name, None)
        if fn is None or getattr(fn, "__br_wrapped__", None) is not None:
            continue
        _orig[name] = fn
        w = _wrap(name, fn)
        setattr(h, name, w)
        for mn, m in list(sys.modules.items()):
            if mn.startswith("tests.") and m is not None and m is not h and getattr(m, name, None) is fn:
                setattr(m, name, w)


def entries(filename, loader="load_sql_fixture_pairs", fixtures_dir=None):
    """Entries of a fixture file as the unsliced loader yields them (optionally from another tree)."""
    import tests.helpers as h
    fn = _orig.get(loader) or getattr(getattr(h, loader), "__br_wrapped__", getattr(h, loader))
    if fixtures_dir is None:
        return list(fn(filename))
    old = h.FIXTURES_DIR
    h.FIXTURES_DIR = fixtures_dir
    try:
        return list(fn(filename))
    finally:
        h.FIXTURES_DIR = old


# ---- text-level entry splitting (mirrors tests/helpers.py at the pinned commit) -----------------
def _filter_comments(s):
    return "\n".join([line for line in s.splitlines() if line and not line.startswith("--")])


def split_singles(text):
    out = []
    lines = []
    for line in _filter_comments(text).splitlines():
        lines.append(line)
        if not line.startswith("#"):
            out.append("\n".join(lines))
            lines.clear()
    return out


def split_pairs(text):
    st = _filter_comments(text).split(";")
    out = []
    for i in range(0, len(st), 2):
        if i + 1 < len(st):
            out.append((st[i].strip(), st[i + 1].strip()))
    return out


def changed_entries(old_text, new_text, singles):
    """Indices of entries that differ, or None when the entry lists do not line up."""
    f = split_singles if singles else split_pairs
    try:
        a, b = f(old_text), f(new_text)
    except Exception:
        return None
    if len(a) != len(b):
        return None
    return {i for i, (x, y) in enumerate(zip(a, b)) if x != y}
