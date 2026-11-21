# Blast Radius selector

Entry for the RWX Blast Radius track: given patches against sqlglot at
`eb11aec4be8cb2122afe713a4816662c3f0fb2fa`, pick the tests each patch can break.

How it works:

1. `br/patchlib.py` parses and applies each unified diff in memory.
2. `br/pydiff.py` compares the old and new syntax trees and classifies each
   difference as a change to deferred code (a function or lambda body) or to
   something evaluated at import time (class and module bodies, def headers).
3. `br/select.py` turns that into candidate tests using maps built from the
   pinned commit only (`maps/maps.bin`): per-test line coverage, files each
   test opens, classes each test uses, and static attribute read sites.
4. `br/main.py` runs the cheap candidates against the patched tree in a forked
   child (`br/runner.py`, `br/overlay.py`) and keeps only the ones that fail,
   then packs the selection across workers (`br/pack.py`).

Anything the analysis does not understand falls back to running every test.

`make sample` builds the image and scores it on the 30 sample commits.
