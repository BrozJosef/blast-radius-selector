"""Miniature replica of tests/test_parser.py::TestParser::test_max_nodes (28.8 s recorded).

The real test parses 10,000 copies of "x=x=...=x" (2.02 M tokens) against node limits of 1,000,000 and
50,000. This replica runs the same statements at width 30 with limits 3,000 and 500, in about 0.1 s.

Use (inside the process that already has the PATCHED sqlglot importable, e.g. the engine's forked child):

    from maxnodes_mini import decide
    include, why = decide(changed={"sqlglot/parser.py": {2202}}, base_work=BASE_WORK)

Rule (the one measured in scout/REPORT.md, Round 2 / S2):
    include the real test  iff  the patch touches a GUARD line
                            or  the replica fails (assertion, any exception, or its own time limit)
                            or  the replica's work count exceeds SLOW_FACTOR x base_work (optional guard)
    otherwise leave test_max_nodes out.

`base_work` is `replica(count=True)[1]` on the unpatched tree in the same interpreter build (compute it once
at image build time). Pass base_work=None to skip the slowdown guard (then no monitoring cost at all).
The rule is a heuristic, not a proof: see the report for the measured disagreement count.
The process must have `unittest` (or pytest) imported BEFORE sqlglot, as the engine's children do: sqlglot
turns on extra argument validation only then (`expressions/core.py: UNITTEST`), and the real test runs with it.
"""
import signal
import sys
import unittest  # noqa: F401  sqlglot enables its UNITTEST argument checks only if this is loaded first

TEST_ID = "tests/test_parser.py::TestParser::test_max_nodes"
WIDTH, DEPTH, BIG, SMALL = 30, 100, 3_000, 500        # real test: 10000, 100, 1_000_000, 50_000
SLOW_FACTOR = 1.5
TIME_LIMIT = 10                                        # seconds of wall for the replica itself
# Lines of the pinned commit that implement the node limit and that almost no other test executes
# (Parser.__init__/reset bookkeeping and the check in Parser.expression). A patch touching them goes
# straight to the real test.
GUARD = {"sqlglot/parser.py": set(range(2040, 2077)) | set(range(2196, 2205))}


def _body():
    from sqlglot import exp, parse_one
    from sqlglot.errors import ParseError

    assert isinstance(parse_one("SELECT 1", max_nodes=BIG), exp.Select)
    cols = [f"col{i}" for i in range(50)]
    assert isinstance(parse_one(f"SELECT {','.join(cols)} FROM t", max_nodes=BIG), exp.Select)
    sql = "SELECT " + ",".join(["x" + "=x" * DEPTH] * WIDTH)
    for limit, needle in ((BIG, "Maximum number of AST nodes"), (SMALL, "(%d)" % SMALL)):
        try:
            parse_one(sql, max_nodes=limit)
        except ParseError as e:
            assert needle in str(e), str(e)[:80]
        else:
            raise AssertionError("no ParseError")
    try:
        parse_one("SELECT " + ",".join(f"x={i}" for i in range(100)), max_nodes=5)
    except ParseError:
        pass
    else:
        raise AssertionError("no ParseError for the small limit")
    assert isinstance(parse_one("SELECT 1, 2, 3"), exp.Select)


class _Timeout(BaseException):
    pass


def replica(count=False):
    """Run the miniature test. Returns (ok, work, detail); work is a line-event count when count=True."""
    work = [0]
    mon = getattr(sys, "monitoring", None) if count else None
    tool = None
    if mon is not None:
        for tool in (4, 3, 5):
            try:
                mon.use_tool_id(tool, "maxnodes_mini")
                break
            except ValueError:
                tool = None
        if tool is not None:
            def on_line(code, line):
                work[0] += 1
            mon.register_callback(tool, mon.events.LINE, on_line)

    def on_alarm(signum, frame):
        raise _Timeout()

    old = signal.signal(signal.SIGALRM, on_alarm)
    old_timer = signal.setitimer(signal.ITIMER_REAL, TIME_LIMIT)
    try:
        import sqlglot  # noqa: F401  (import cost is not part of the work count)
        import sqlglot.expressions.core as _core
        if not getattr(_core, "UNITTEST", True):
            # sqlglot was imported before unittest/pytest: the replica would skip checks the real test runs
            return False, None, "sqlglot imported without unittest loaded"
        if tool is not None:
            mon.set_events(tool, mon.events.LINE)
        _body()
        return True, (work[0] if tool is not None else None), "pass"
    except BaseException as e:  # noqa: BLE001 - any failure, including our own time limit, means "include"
        if isinstance(e, KeyboardInterrupt):
            raise
        return False, None, "%s: %s" % (type(e).__name__, str(e)[:80])
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if tool is not None:
            mon.set_events(tool, 0)
            mon.register_callback(tool, mon.events.LINE, None)
            mon.free_tool_id(tool)
        signal.signal(signal.SIGALRM, old)
        if old_timer and old_timer[0]:
            signal.setitimer(signal.ITIMER_REAL, old_timer[0])


def decide(changed=None, base_work=None):
    """changed: {relative path: set of old line numbers the patch touches}. Returns (include, reason)."""
    for path, lines in (changed or {}).items():
        g = GUARD.get(path)
        if g and (g & set(lines)):
            return True, "patch touches the node-limit lines"
    ok, work, detail = replica(count=base_work is not None)
    if not ok:
        return True, "replica failed (%s)" % detail
    if base_work is not None and work is not None and work > SLOW_FACTOR * base_work:
        return True, "replica did %.2fx the base work" % (work / base_work)
    return False, "replica passed"


if __name__ == "__main__":      # python maxnodes_mini.py <tree>  -> prints the verdict for that tree
    import os
    import time
    tree = sys.argv[1]
    os.chdir(tree)
    sys.path.insert(0, tree)
    t = time.process_time()
    ok, work, detail = replica(count="--count" in sys.argv)
    print("PASS" if ok else "FAIL", work, detail, "cpu %.2fs" % (time.process_time() - t))
