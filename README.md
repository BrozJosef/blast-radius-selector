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
   Entries are chosen by per-entry line coverage, per-entry operand profiles,
   and for class-attribute changes by which classes an entry instantiates.
   A test that runs well past its expected time is treated as failing (the
   track counts timeouts as failures) rather than waited for.
6. `br/memo.py` caches, at image build time and from the pinned commit only,
   pure intermediate results of the TPC executor tests: DuckDB's reference
   answers keyed by SQL text, the parsed CSV rows, and the Python executor's
   result keyed by a structural key of the query or plan. A cache level is
   used only when no changed line can run inside the stage it replaces.

7. `br/fingerprint.py` digests the state importing sqlglot leaves behind (class
   attributes, module globals, closure contents, default arguments). The same
   digest of the pinned tree is taken at image build time. When a patch changes
   code that runs during import but the digests are equal, only tests that
   execute the changed lines at run time can be affected; when only the assigned
   attribute differs, only tests that read it at run time can.
   The comparison of interpreter-level state is taken as a before/after
   difference so that the surrounding process does not count. `maps/ck.bin`
   records, per test run alone on the pinned commit, which class-level
   constants it fetched and, for dict tables, which entries it looked up
   (`lab`-side tracer replaces each constant by a recording descriptor). When
   the fingerprint shows that only the assigned constant differs, the readers
   of that constant (or of the changed entries) are the candidates. This
   narrowing is behind `BR_CK=1` for now.
8. `br/maxnodes_mini.py` is a scaled-down run of `test_max_nodes` (same
   statements, 30 columns instead of 10,000). The 29-second original is included
   only if the patch touches the node-limit lines, the scaled run fails, or it
   does noticeably more work than on the pinned tree. This is the one heuristic
   in the entry; it agreed with the real test on all 196 mutants we tried.

There is no table from patches or kinds of edit to failing tests anywhere:
every decision is derived from the patch itself, from maps of the unpatched
code, or from actually running tests on the patched code. Anything the
analysis does not understand falls back to running every test.

`make sample` builds the image and scores it on the 30 sample commits.
