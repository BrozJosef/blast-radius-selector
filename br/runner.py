"""Run unittest-style test ids in this process and report which fail (pytest-equivalent verdicts)."""
import importlib
import signal
import sys
import unittest


class Timeout(BaseException):
    pass


class _Result(unittest.TestResult):
    def __init__(self):
        super().__init__()
        self.bad = set()       # test ids (unittest id form) that failed
        self.bad_classes = set()

    def _mark(self, test):
        tid = getattr(test, "test_case", test)  # subtests point at their parent
        if isinstance(tid, unittest.TestCase):
            self.bad.add(tid.id())
        else:
            # _ErrorHolder: "setUpClass (tests.x.Class)" / "setUpModule (tests.x)" / tearDownClass ...
            d = getattr(tid, "description", str(tid))
            self.bad_classes.add(d)

    def addError(self, test, err):
        super().addError(test, err)
        self._mark(test)

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self._mark(test)

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err is not None:
            self.bad.add(test.id())

    def addUnexpectedSuccess(self, test):
        super().addUnexpectedSuccess(test)
        self._mark(test)


def _alarm(signum, frame):
    raise Timeout()


def run_ids(ids, limits=None, default_limit=60.0):
    """ids: 'tests/x.py::Class::method'. Returns (failing set, info dict)."""
    failing = set()
    info = {}
    by_class = {}
    for tid in ids:
        f, c, m = tid.split("::")
        by_class.setdefault((f, c), []).append((m, tid))
    signal.signal(signal.SIGALRM, _alarm)
    for (f, c), items in by_class.items():
        modname = f[:-3].replace("/", ".")
        try:
            mod = importlib.import_module(modname)
            cls = getattr(mod, c)
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
        budget = sum((limits or {}).get(tid, default_limit) for _, tid in items) if limits else default_limit * len(items)
        signal.setitimer(signal.ITIMER_REAL, max(2.0, budget), 5.0)
        try:
            suite.run(res)
        except Timeout:
            for m, tid in items:
                failing.add(tid)
            info[f + "::" + c] = "timeout"
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
        for uid in res.bad:
            if uid in idmap:
                failing.add(idmap[uid])
        if res.bad_classes:
            for m, tid in items:
                failing.add(tid)
            info[f + "::" + c] = "class-level: %s" % sorted(res.bad_classes)
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
