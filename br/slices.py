"""Fixture slicing: let a fixture-driven test run only selected statement pairs.

Tests load `tests/fixtures/*.sql` through tests.helpers.load_sql_fixture_pairs / load_sql_fixtures and
loop over the entries independently, so a test restricted to the entries that can observe a change
gives the same verdict as the whole test at a fraction of the cost.
"""
import sys

ALLOW = {}      # loader filename -> set of entry indices (absent = everything)
SEEN = {}       # loader filename -> number of entries yielded by the unsliced loader (for discovery)
_orig = {}


def _wrap(name, fn):
    def wrapped(filename, *a, **k):
        allow = ALLOW.get(filename)
        n = 0
        for i, item in enumerate(fn(filename, *a, **k)):
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
