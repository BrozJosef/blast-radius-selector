"""Run unittest-style test ids in this process and report which fail (pytest-equivalent verdicts)."""
import importlib
import signal
import sys
import unittest


FAIL_FAST = True
MINI = None      # {"changed": {path: [lines]}, "base": work count} when the miniature max_nodes test may stand in
MAX_TIMEOUTS = 2


class _Abort(Exception):
    pass


MEMO_OK = True   # cleared when the patch touches test code


class Timeout(BaseException):
    pass


class _Result(unittest.TestResult):
    def __init__(self):
        super().__init__()
        self.bad = set()       # test ids (unittest id form) that failed
        self.bad_classes = set()
        self.pool_broke_at = None
        self.limits = {}
        self.default_limit = 30.0
        self.setup_limit = 30.0
        self.timeouts = 0

    def addError(self, test, err):
        if err and err[0] is Timeout:
            self.timeouts += 1
            if self.pool_broke_at is None and isinstance(test, unittest.TestCase) and hasattr(type(test), "pool"):
                self.pool_broke_at = test.id()   # its hung tasks keep the class-wide pool's workers busy
            if self.timeouts >= MAX_TIMEOUTS:
                self.shouldStop = True   # a hanging mutant: do not burn the budget on every remaining test
        super().addError(test, err)
        self._mark(test)

    def _unused(self):
        pass

    def _mark(self, test):
        tid = getattr(test, "test_case", test)  # subtests point at their parent
        if isinstance(tid, unittest.TestCase):
            self.bad.add(tid.id())
        else:
            # _ErrorHolder: "setUpClass (tests.x.Class)" / "setUpModule (tests.x)" / tearDownClass ...
            d = getattr(tid, "description", str(tid))
            self.bad_classes.add(d)

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self._mark(test)

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err is not None:
            self.bad.add(test.id())
            if err[0] is Timeout:
                self.timeouts += 1
                if self.pool_broke_at is None and hasattr(type(test), "pool"):
                    self.pool_broke_at = test.id()
                if self.timeouts >= MAX_TIMEOUTS:
                    self.shouldStop = True
            if FAIL_FAST:
                raise _Abort()  # verdict is known; skip the rest of this test

    def startTest(self, test):
        super().startTest(test)
        m = sys.modules.get("br.memo")
        if m is not None:
            m.SPENT[0] = 0.0     # the memo-miss budget is per test
        lim = self.limits.get(test.id(), self.default_limit)
        signal.setitimer(signal.ITIMER_REAL, lim)

    def stopTest(self, test):
        signal.setitimer(signal.ITIMER_REAL, self.setup_limit)   # covers class/module fixtures in between
        super().stopTest(test)
        # a class-wide pool that broke during this test takes every later pool user down with it
        if self.pool_broke_at is None:
            pool = getattr(type(test), "pool", None)
            if getattr(pool, "_broken", False):
                self.pool_broke_at = test.id()

    def addUnexpectedSuccess(self, test):
        super().addUnexpectedSuccess(test)
        self._mark(test)


class _PytestLikeHandler:
    pass


def install_logging():
    """pytest attaches capture handlers to the root logger that format every record and re-raise
    formatting errors, so a log call whose arguments cannot be rendered fails the test."""
    import logging

    class H(logging.Handler):
        def emit(self, record):
            self.format(record)

        def handleError(self, record):
            raise

    root = logging.getLogger()
    if not any(type(h).__name__ == "H" and getattr(h, "_br", False) for h in root.handlers):
        h = H()
        h._br = True
        h.setFormatter(logging.Formatter("%(levelname)-8s %(name)s:%(filename)s:%(lineno)d %(message)s"))
        root.addHandler(h)


def _alarm(signum, frame):
    raise Timeout()


def run_ids(ids, limits=None, default_limit=30.0, setup_limit=30.0):
    """ids: 'tests/x.py::Class::method'. Returns (failing set, info dict)."""
    failing = set()
    info = {}
    by_class = {}
    install_logging()
    if MINI is not None:
        from . import maxnodes_mini
        if maxnodes_mini.TEST_ID in ids:
            # A scaled-down run of the same statements decides whether the 29 s original is needed.
            ids = [t for t in ids if t != maxnodes_mini.TEST_ID]
            try:
                inc, why = maxnodes_mini.decide({k: set(v) for k, v in (MINI.get("changed") or {}).items()}, MINI.get("base"))
            except BaseException as e:  # noqa
                if isinstance(e, KeyboardInterrupt):
                    raise
                inc, why = True, "replica error %r" % (e,)
            if inc:
                failing.add(maxnodes_mini.TEST_ID)
            info["mini_max_nodes"] = why
    for tid in ids:
        f, c, m = tid.split("::")
        by_class.setdefault((f, c), []).append((m, tid))
    signal.signal(signal.SIGALRM, _alarm)
    hung = False
    for (f, c), items in by_class.items():
        if hung:
            for m, tid in items:
                failing.add(tid)
            continue
        modname = f[:-3].replace("/", ".")
        try:
            mod = importlib.import_module(modname)
            cls = getattr(mod, c)
            if MEMO_OK and modname == "tests.test_executor":
                from . import memo
                memo.install(mod)
        except Timeout:
            raise
        except BaseException as e:  # import error -> every test in the file fails
            if isinstance(e, (KeyboardInterrupt,)):
                raise
            for m, tid in items:
                failing.add(tid)
            info[f] = "import: %r" % (e,)
            continue
        cases = []
        idmap = {}
        for m, tid in items:
            try:
                case = cls(m)
            except Exception:
                failing.add(tid)  # test no longer exists
                continue
            cases.append(case)
            idmap[case.id()] = tid
        if not cases:
            continue
        res = _Result()
        suite = unittest.TestSuite(cases)
        lim = limits or {}
        res.limits = {case.id(): lim.get(idmap[case.id()], default_limit) for case in cases}
        res.default_limit = default_limit
        res.setup_limit = setup_limit
        signal.setitimer(signal.ITIMER_REAL, setup_limit)
        try:
            suite.run(res)
        except Timeout:
            # raised outside any test (class setup or teardown hung)
            for m, tid in items:
                failing.add(tid)
            info[f + "::" + c] = "timeout"
            info["timeouts"] = info.get("timeouts", 0) + 1
            continue
        except BaseException as e:
            if isinstance(e, KeyboardInterrupt):
                raise
            for m, tid in items:
                failing.add(tid)
            info[f + "::" + c] = "crash %r" % (e,)
            continue
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        if res.shouldStop:
            # stopped after repeated timeouts: everything not run is assumed to hang too
            done = {t.id() for t in cases[:res.testsRun]}
            for case in cases:
                if case.id() not in done:
                    failing.add(idmap[case.id()])
            info[f + "::" + c] = "stopped after %d timeouts" % res.timeouts
            hung = True
        if res.timeouts:
            info["timeouts"] = info.get("timeouts", 0) + res.timeouts
        for uid in res.bad:
            if uid in idmap:
                failing.add(idmap[uid])
        if res.bad_classes:
            for m, tid in items:
                failing.add(tid)
            info[f + "::" + c] = "class-level: %s" % sorted(res.bad_classes)
        if res.pool_broke_at is not None:
            info.setdefault("pool_broken", {})[f + "::" + c] = res.pool_broke_at.rsplit(".", 1)[-1]
    return failing, info


if __name__ == "__main__":
    import json
    import os
    sys.path.insert(0, os.getcwd())
    from br import inlinepool  # noqa
    inlinepool.install()
    ids = json.loads(sys.stdin.read())
    bad, info = run_ids(ids)
    sys.stdout.write(json.dumps({"failing": sorted(bad), "info": info}))
