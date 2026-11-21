"""In-process stand-in for ProcessPoolExecutor (tests fan work out to spawn pools)."""
import pickle
import concurrent.futures as cf
import concurrent.futures.process as cfp
from concurrent.futures import Future


class InlineExecutor:
    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(), **kw):
        self._init = (initializer, initargs)
        self._inited = False

    def submit(self, fn, /, *args, **kwargs):
        f = Future()
        # A real pool pickles the call into the worker and the result back; sqlglot
        # expressions define their own reduce protocol, so the round trip is behaviour.
        try:
            if not self._inited:
                self._inited = True
                if self._init[0]:
                    ifn, iargs = pickle.loads(pickle.dumps(self._init))
                    ifn(*iargs)
            fn, args, kwargs = pickle.loads(pickle.dumps((fn, args, kwargs)))
            res = fn(*args, **kwargs)
            f.set_result(pickle.loads(pickle.dumps(res)))
        except BaseException as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit)) or type(e).__name__ == "Timeout":
                raise
            f.set_exception(e)
        return f

    def map(self, fn, *iterables, **kw):
        return iter([fn(*a) for a in zip(*iterables)])

    def shutdown(self, wait=True, **kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def install():
    cf.ProcessPoolExecutor = InlineExecutor
    cfp.ProcessPoolExecutor = InlineExecutor
