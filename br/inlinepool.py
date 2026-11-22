"""In-process stand-in for ProcessPoolExecutor (tests fan work out to spawn pools)."""
import pickle
import concurrent.futures as cf
import concurrent.futures.process as cfp
from concurrent.futures import Future


BROKEN = []   # one entry per executor whose (emulated) worker died
IN_PICKLE = [0]  # >0 while a task or reply is being pickled/unpickled (used by offline tracing)


def _dumps(o):
    IN_PICKLE[0] += 1
    try:
        return pickle.dumps(o)
    finally:
        IN_PICKLE[0] -= 1


def _loads(b):
    IN_PICKLE[0] += 1
    try:
        return pickle.loads(b)
    finally:
        IN_PICKLE[0] -= 1


class InlineExecutor:
    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(), **kw):
        self._init = (initializer, initargs)
        self._inited = False
        self._broken = False

    def _broken_future(self):
        f = Future()
        f.set_exception(cfp.BrokenProcessPool(
            "A process in the process pool was terminated abruptly while the future was running or pending."))
        return f

    def submit(self, fn, /, *args, **kwargs):
        if self._broken:
            raise cfp.BrokenProcessPool("A child process terminated abruptly, the process pool is not usable anymore")
        f = Future()
        # A real pool pickles the call into the worker and the result back; sqlglot
        # expressions define their own reduce protocol, so the round trip is behaviour.
        try:
            if not self._inited:
                self._inited = True
                if self._init[0]:
                    # a spawned worker receives the initializer and its arguments through pickle
                    blob = _dumps(self._init)
                    try:
                        ifn, iargs = _loads(blob)
                    except BaseException as e:
                        if type(e).__name__ == "Timeout":
                            raise
                        self._broken = True
                        BROKEN.append(self)
                        return self._broken_future()
                    ifn(*iargs)
            data = _dumps((fn, args, kwargs))
        except BaseException as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit)) or type(e).__name__ == "Timeout":
                raise
            f.set_exception(e)   # the parent could not send the task: only this future fails
            return f
        try:
            fn, args, kwargs = _loads(data)
        except BaseException as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit)) or type(e).__name__ == "Timeout":
                raise
            # A worker that cannot unpickle its task dies, and the whole pool breaks with it.
            self._broken = True
            BROKEN.append(self)
            return self._broken_future()
        try:
            res = fn(*args, **kwargs)
        except BaseException as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit)) or type(e).__name__ == "Timeout":
                raise
            return self._back(f, e, True)
        return self._back(f, res, False)

    def _back(self, f, obj, is_exc):
        """Send a result (or exception) back the way a worker would."""
        try:
            blob = _dumps(obj)
        except BaseException as e2:
            if type(e2).__name__ == "Timeout":
                raise
            f.set_exception(e2)       # the worker reports the pickling error instead
            return f
        try:
            back = _loads(blob)
        except BaseException as e3:
            if type(e3).__name__ == "Timeout":
                raise
            # the parent cannot read the worker's reply: the pool is declared broken
            self._broken = True
            BROKEN.append(self)
            return self._broken_future()
        if is_exc:
            f.set_exception(obj)
        else:
            f.set_result(back)
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
