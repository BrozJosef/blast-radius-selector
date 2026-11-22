"""HOT mode: run candidate tests against a patch by swapping function code in a warm, forked process.

A long-lived zygote keeps sqlglot and the test modules imported (unpatched).  For every job it forks
a grandchild that replaces ``__code__`` on the live function objects touched by the patch and runs the
tests.  Anything that is not a pure, exactly-matched function-body change is refused ("cannot") so the
caller can fall back to a real patched tree.  Never guesses.

Parent side:
    z = hot.Zygote(src="/app/src")
    failing, why = z.run(cid, {relpath: new_source}, ids, limits, timeout)
        failing is a set of test ids, or None when HOT cannot / did not decide (why says so)
    z.close()
"""
import gc
import importlib
import json
import os
import select as _select
import signal
import struct
import sys
import time
import types

CO_OPTIMIZED = 0x1
CO_GEN_FLAGS = 0x20 | 0x80 | 0x100 | 0x200  # generator, coroutine, iterable coroutine, async generator
CO_VARFLAGS = 0x04 | 0x08                    # *args, **kwargs


# --------------------------------------------------------------------------- code trees
def _const_sig(c):
    if isinstance(c, types.CodeType):
        return ("<code>", c.co_name)
    if isinstance(c, tuple):
        return ("tuple", tuple(_const_sig(x) for x in c))
    if isinstance(c, frozenset):
        return ("frozenset", tuple(sorted((_const_sig(x) for x in c), key=repr)))
    return (type(c).__name__, repr(c))


def shallow_sig(co):
    """Everything that defines this code object's behaviour except line numbers and nested bodies."""
    return (
        co.co_code,
        tuple(_const_sig(c) for c in co.co_consts),
        co.co_names,
        co.co_varnames,
        co.co_freevars,
        co.co_cellvars,
        co.co_flags,
        co.co_argcount,
        co.co_posonlyargcount,
        co.co_kwonlyargcount,
        co.co_name,
        co.co_qualname,
        co.co_exceptiontable,
    )


def code_tree(root):
    """{path: code}; path is a tuple of (co_name, ordinal among same-named siblings)."""
    out = {}

    def rec(co, path):
        out[path] = co
        seen = {}
        for c in co.co_consts:
            if isinstance(c, types.CodeType):
                k = seen.get(c.co_name, 0)
                seen[c.co_name] = k + 1
                rec(c, path + ((c.co_name, k),))

    rec(root, ())
    return out


def is_func(co):
    return bool(co.co_flags & CO_OPTIMIZED)


class Cannot(Exception):
    pass


def _doc(co):
    return co.co_consts[0] if (co.co_flags & 0x4000000 and co.co_consts) else None


def plan_patch(old_root, new_root):
    """Return [(old_code, new_code)] for every function-like code object that must be swapped.

    Raises Cannot when the difference is not confined to function bodies.
    """
    ot, nt = code_tree(old_root), code_tree(new_root)
    if set(ot) != set(nt):
        raise Cannot("code structure differs")
    changed = [p for p in ot if shallow_sig(ot[p]) != shallow_sig(nt[p])]
    if not changed:
        return []
    roots = set()
    for p in changed:
        co = ot[p]
        if not is_func(co):
            raise Cannot("import-level code differs at %s" % (p,))
        # outermost function-like ancestor (first function on the way down from the module)
        q = None
        for i in range(1, len(p) + 1):
            if is_func(ot[p[:i]]):
                q = p[:i]
                break
        # every code object above q is a module/class body; every one below must be a function
        head = ot[q]
        if head.co_name.startswith("<") and head.co_name != "<lambda>":
            raise Cannot("import-time comprehension/genexpr %s" % head.co_name)
        roots.add(q)
    # two indistinguishable code objects (same line, name and body) cannot be told apart at run time
    twins = {}
    for p, co in ot.items():
        if is_func(co):
            k = (co.co_firstlineno, co.co_name, shallow_sig(co))
            twins[k] = twins.get(k, 0) + 1
    pairs = []
    for q in sorted(roots):
        for p in sorted(ot):
            if p[:len(q)] != q:
                continue
            o, n = ot[p], nt[p]
            if twins.get((o.co_firstlineno, o.co_name, shallow_sig(o)), 0) > 1:
                raise Cannot("ambiguous twin code object %s" % o.co_qualname)
            if _doc(o) != _doc(n):
                raise Cannot("docstring differs in %s" % o.co_qualname)
            if o.co_freevars != n.co_freevars:
                raise Cannot("closure variables differ in %s" % o.co_qualname)
            if (o.co_flags & (CO_GEN_FLAGS | CO_VARFLAGS)) != (n.co_flags & (CO_GEN_FLAGS | CO_VARFLAGS)):
                raise Cannot("function kind differs in %s" % o.co_qualname)
            if (o.co_argcount, o.co_posonlyargcount, o.co_kwonlyargcount) != (
                    n.co_argcount, n.co_posonlyargcount, n.co_kwonlyargcount):
                raise Cannot("signature differs in %s" % o.co_qualname)
            na = o.co_argcount + o.co_kwonlyargcount + bool(o.co_flags & 0x04) + bool(o.co_flags & 0x08)
            if o.co_varnames[:na] != n.co_varnames[:na]:
                raise Cannot("parameter names differ in %s" % o.co_qualname)
            pairs.append((o, n, p == q))
    return pairs


# --------------------------------------------------------------------------- live function index
class Index:
    def __init__(self, src):
        self.src = src.rstrip("/") + "/"
        self.map = {}
        self.dirty = True

    def rebuild(self):
        m = {}
        src = self.src
        FT = types.FunctionType
        for o in gc.get_objects():
            if type(o) is FT:
                co = o.__code__
                if co.co_filename.startswith(src):
                    m.setdefault((co.co_filename, co.co_firstlineno, co.co_name), []).append(o)
        self.map = m
        self.dirty = False

    def live(self, old):
        sig = None
        out = []
        for f in self.map.get((old.co_filename, old.co_firstlineno, old.co_name), ()):
            co = f.__code__
            if co is old:
                out.append(f)
                continue
            if sig is None:
                sig = shallow_sig(old)
            if shallow_sig(co) == sig:
                out.append(f)
        return out


def modname(rel):
    m = rel[:-3].replace("/", ".")
    return m[:-9] if m.endswith(".__init__") else m


# --------------------------------------------------------------------------- framing
def _send(fd, obj):
    data = json.dumps(obj).encode()
    data = struct.pack("<I", len(data)) + data
    while data:
        n = os.write(fd, data)
        data = data[n:]


def _recv(fd, timeout=None):
    """Read one framed message. Returns None on EOF/timeout."""
    deadline = None if timeout is None else time.time() + timeout
    buf = b""
    need = 4
    size = None
    while True:
        if deadline is not None:
            left = deadline - time.time()
            if left <= 0:
                return None
            r, _, _ = _select.select([fd], [], [], left)
            if not r:
                return None
        try:
            chunk = os.read(fd, 1 << 16)
        except InterruptedError:
            continue
        if not chunk:
            return None
        buf += chunk
        if size is None and len(buf) >= 4:
            size = struct.unpack("<I", buf[:4])[0]
            buf = buf[4:]
            need = size
        if size is not None and len(buf) >= need:
            return json.loads(buf[:need].decode())


# --------------------------------------------------------------------------- zygote (server side)
class _Server:
    def __init__(self, src, rfd, wfd, debug):
        self.src = src.rstrip("/")
        self.rfd, self.wfd = rfd, wfd
        self.debug = debug
        self.index = Index(self.src)
        self.old_code = {}  # relpath -> compiled original module code (or Cannot)

    def setup(self):
        devnull = os.open(os.devnull, os.O_RDWR)
        os.dup2(devnull, 0)
        os.dup2(devnull, 1)
        if not self.debug:
            os.dup2(devnull, 2)
        os.environ.setdefault("SKIP_INTEGRATION", "0")
        os.chdir(self.src)
        if self.src not in sys.path:
            sys.path.insert(0, self.src)
        from . import inlinepool
        inlinepool.install()
        import sqlglot  # noqa: F401
        import unittest  # noqa: F401

    def original(self, rel):
        c = self.old_code.get(rel)
        if c is None:
            path = os.path.join(self.src, rel)
            try:
                with open(path, encoding="utf-8", newline="") as f:
                    text = f.read()
                c = compile(text, path, "exec", dont_inherit=True)
            except Exception as e:  # noqa: BLE001
                c = Cannot("original does not compile: %r" % (e,))
            self.old_code[rel] = c
        if isinstance(c, Cannot):
            raise c
        return c

    def ensure(self, mod):
        if mod in sys.modules:
            return
        importlib.import_module(mod)
        self.index.dirty = True

    def serve(self):
        self.setup()
        import resource
        while True:
            job = _recv(self.rfd)
            if job is None or job.get("op") == "quit":
                os._exit(0)
            try:
                res = self.job(job)
            except Cannot as e:
                res = {"cannot": str(e)}
            except BaseException as e:  # noqa: BLE001
                res = {"cannot": "zygote error %r" % (e,)}
            a = resource.getrusage(resource.RUSAGE_SELF)
            b = resource.getrusage(resource.RUSAGE_CHILDREN)
            res["cpu"] = a.ru_utime + a.ru_stime + b.ru_utime + b.ru_stime
            _send(self.wfd, res)

    def job(self, job):
        files = job["files"]
        ids = job["ids"]
        # 1. everything the grandchild needs must already be imported here, unpatched
        for rel in files:
            if not rel.endswith(".py"):
                raise Cannot("not python: %s" % rel)
            try:
                self.ensure(modname(rel))
            except BaseException as e:  # noqa: BLE001
                raise Cannot("cannot import %s: %r" % (rel, e))
        for f in sorted({t.split("::")[0] for t in ids}):
            try:
                self.ensure(modname(f))
            except BaseException as e:  # noqa: BLE001
                raise Cannot("cannot import test module %s: %r" % (f, e))
        olds = {rel: self.original(rel) for rel in files}
        if self.index.dirty:
            self.index.rebuild()
        before = set(sys.modules)
        r, w = os.pipe()
        pid = os.fork()
        if pid == 0:
            code = 1
            try:
                os.close(r)
                res = self.grandchild(files, olds, ids, job.get("limits"), before, job.get("allow"), job.get("override"), job.get("opts"))
                _send(w, res)
                code = 0
            except Cannot as e:
                try:
                    _send(w, {"cannot": str(e)})
                except Exception:  # noqa: BLE001
                    pass
            except BaseException as e:  # noqa: BLE001
                try:
                    _send(w, {"cannot": "child error %r" % (e,)})
                except Exception:  # noqa: BLE001
                    pass
            finally:
                os._exit(code)
        os.close(w)
        res = _recv(r, job.get("timeout", 60.0))
        os.close(r)
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            os.waitpid(pid, 0)
        except OSError:
            pass
        if res is None:
            return {"cannot": "timeout or crash"}
        # warm up: import here what the grandchild had to import lazily
        for m in res.pop("newmods", ()):
            if m not in sys.modules:
                try:
                    importlib.import_module(m)
                    self.index.dirty = True
                except BaseException:  # noqa: BLE001
                    pass
        return res

    def grandchild(self, files, olds, ids, limits, before, allow=None, override=None, opts=None):
        npatched = 0
        if opts and opts.get("exec_memo"):
            from . import memo
            memo.EXEC_OK = opts["exec_memo"]
        if allow or override:
            from . import slices
            slices.ALLOW.clear()
            slices.ALLOW.update({k: set(v) for k, v in (allow or {}).items()})
            if override:
                d = os.path.join(os.environ.get("BR_TMP", "/tmp"), "fx_%d" % os.getpid())
                for name, text in override.items():
                    path = os.path.join(d, name)
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, "w", encoding="utf-8", newline="") as f:
                        f.write(text)
                    slices.OVERRIDE[name] = d
            slices.install()
        for rel, text in files.items():
            path = os.path.join(self.src, rel)
            try:
                new = compile(text, path, "exec", dont_inherit=True)
            except BaseException as e:  # noqa: BLE001
                raise Cannot("patched source does not compile: %r" % (e,))
            pairs = plan_patch(olds[rel], new)
            if not pairs:
                raise Cannot("no code difference in %s" % rel)
            todo = []
            for o, n, is_root in pairs:
                live = self.index.live(o)
                if is_root and not live:
                    raise Cannot("no live function for %s" % o.co_qualname)
                for f in live:
                    if len(f.__code__.co_freevars) != len(n.co_freevars):
                        raise Cannot("free variable mismatch on %s" % o.co_qualname)
                    todo.append((f, n))
            for f, n in todo:
                f.__code__ = n
                npatched += 1
        from . import runner
        runner.MEMO_OK = not any(p.startswith("tests/") for p in files)
        bad, info = runner.run_ids(ids, limits)
        newmods = [m for m in sys.modules if m not in before and m.startswith(("sqlglot", "tests"))]
        return {"failing": sorted(bad), "info": info, "newmods": newmods, "patched": npatched}


# --------------------------------------------------------------------------- parent side
class Zygote:
    def __init__(self, src, debug=False):
        self.src = src
        self.debug = debug
        self.pid = None
        self.cpu = 0.0       # cumulative CPU seconds reported by the current zygote
        self.cpu_dead = 0.0  # CPU of zygotes that were killed/restarted
        self.jobs = 0

    def start(self):
        to_r, to_w = os.pipe()      # parent -> zygote
        from_r, from_w = os.pipe()  # zygote -> parent
        pid = os.fork()
        if pid == 0:
            try:
                os.close(to_w)
                os.close(from_r)
                try:
                    os.setpgid(0, 0)
                except OSError:
                    pass
                _Server(self.src, to_r, from_w, self.debug).serve()
            finally:
                os._exit(1)
        os.close(to_r)
        os.close(from_w)
        self.pid, self.w, self.r = pid, to_w, from_r

    def stop(self, kill=True):
        if self.pid is None:
            return
        try:
            if kill:
                try:
                    os.killpg(self.pid, signal.SIGKILL)
                except OSError:
                    os.kill(self.pid, signal.SIGKILL)
            else:
                _send(self.w, {"op": "quit"})
        except OSError:
            pass
        for fd in (self.w, self.r):
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            os.waitpid(self.pid, 0)
        except OSError:
            pass
        self.pid = None
        self.cpu_dead += self.cpu
        self.cpu = 0.0

    close = stop

    def total_cpu(self):
        return self.cpu_dead + self.cpu

    def run(self, cid, files, ids, limits=None, timeout=60.0, allow=None, override=None, opts=None):
        """files: {relpath: full new source}. Returns (failing set | None, reason)."""
        if self.pid is None:
            try:
                self.start()
            except OSError as e:
                return None, "cannot start zygote: %r" % (e,)
        try:
            _send(self.w, {"cid": cid, "files": files, "ids": list(ids), "limits": limits, "timeout": timeout,
                           "allow": {k: sorted(v) for k, v in (allow or {}).items()}, "override": override or {}, "opts": opts or {}})
        except OSError as e:
            self.stop()
            return None, "zygote gone: %r" % (e,)
        res = _recv(self.r, timeout + 20.0)
        self.jobs += 1
        if res is None:
            self.stop()
            return None, "zygote unresponsive"
        self.cpu = res.get("cpu", self.cpu)
        if "cannot" in res:
            return None, res["cannot"]
        self.last_info = res.get("info")
        return set(res["failing"]), "hot (%d functions)" % res.get("patched", 0)


def eligible(plan):
    """A plan HOT may try: only in-place patchable function changes, nothing import-level."""
    if plan.all or plan.overlay or plan.no_exec or plan.renames or not plan.hot:
        return False
    hot_files = {p for p, _ in plan.hot}
    for p, c in plan.files.items():
        if c is None or p not in hot_files:
            return False  # some other file changed too (data, deleted, AST-identical python...)
    return True
