# Blast Radius selector

Entry for the RWX Blast Radius track: given patches against sqlglot at
`eb11aec4be8cb2122afe713a4816662c3f0fb2fa`, pick the tests each patch can break.

How it works:

1. `br/patchlib.py` parses and applies each unified diff in memory.
2. `br/pydiff.py` compares the old and new syntax trees and classifies each
   difference as a change to deferred code (a function or lambda body), to
   something evaluated at import time (class and module bodies, def headers),
   or as a plain operator swap at a known comparison / boolean site.
3. `br/select.py` turns that into candidate tests using maps built from the
   pinned commit only (`maps/`): per-test line coverage, files each test opens,
   classes each test uses, static attribute read sites, lines that run during
   imports, and value profiles (for every comparison and boolean expression,
   the operand patterns each test produced there). A test in which the new
   operator gives the same result on every recorded evaluation cannot be
   affected by the swap and is dropped without being run.
4. `br/main.py` decides, by estimated cost, which candidates to run against
   the patched code and which to include unrun. Function-body changes are
   patched into a warm process (`br/hot.py`); anything import-level runs
   against a symlink overlay of the patched tree (`br/overlay.py`) after an
   import probe. `br/runner.py` gives pytest-equivalent verdicts for the
   unittest-style suite, including process-pool behaviour (`br/inlinepool.py`).
5. Fixture-driven tests are restricted to the fixture entries that can observe
   the change (`br/slices.py`), which is equivalent to running the whole test.
6. `br/memo.py` caches, at image build time and from the pinned commit only,
   pure intermediate results of the TPC executor tests: DuckDB's reference
   answers keyed by SQL text, the parsed CSV rows, and the Python executor's
   result keyed by a structural key of the query or plan. A cache level is
   used only when no changed line can run inside the stage it replaces.

There is no table from patches or kinds of edit to failing tests anywhere:
every decision is derived from the patch itself, from maps of the unpatched
code, or from actually running tests on the patched code. Anything the
analysis does not understand falls back to running every test.

`make sample` builds the image and scores it on the 30 sample commits.
