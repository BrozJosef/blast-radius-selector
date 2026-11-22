"""Per-rule memo of the optimizer pipeline for the TPC executor tests.

`sqlglot.executor.execute` runs a query through `optimize` (a fixed sequence of rule functions),
plans it and executes the plan. When a patch can only reach code that runs inside some of those
rules, everything before a touched rule is unchanged, so its input is the one recorded on the
pinned commit. The touched rule is run for real on that recorded input; if its output is the
recorded output, the rest of the pipeline, the plan and the result are the recorded ones too, and
the recorded result is returned. As soon as one output differs, the remaining rules, the planner
and the executor run for real on it.

Recorded at image build time, from the pinned commit only: for every query of the executor tests
and every rule, the rule's input tree (serialised field by field, not through sqlglot's own
pickling) and a structural digest of its output; plus which source lines run inside each rule and
which run in the query stage outside any rule. A recorded input is kept only if it reloads to a
structurally identical tree and the rule gives the recorded output on the reloaded tree.
"""
import hashlib
import io
import os
import pickle
import sys
import zlib

from . import memo

PATH = os.path.join(os.path.dirname(memo.PATH), "rule_memo.bin")
_IDX = []
_Match = type("_Match", (BaseException,), {})


def _expr_base():
    from sqlglot.expressions.core import Expr
    return Expr


def _rebuild(cls, args, comments, typ, meta):
    o = cls.__new__(cls)
    o.args = args
    o.parent = None
    o.arg_key = None
    o.index = None
    o.comments = comments
    o._type = typ
    o._meta = meta
    o._hash = None
    base = _expr_base()
    for k, v in args.items():
        if isinstance(v, base):
            v.parent = o
            v.arg_key = k
            v.index = None
        elif type(v) is list:
            for i, x in enumerate(v):
                if isinstance(x, base):
                    x.parent = o
                    x.arg_key = k
                    x.index = i
    return o


def dumps(e):
    base = _expr_base()

    class P(pickle.Pickler):
        def reducer_override(self, o):
            if isinstance(o, base):
                return (_rebuild, (type(o), o.args, o.comments, o._type, o._meta))
            return NotImplemented

    b = io.BytesIO()
    P(b, protocol=5).dump(e)
    return b.getvalue()


def digest(e):
    return hashlib.sha1(repr(memo._key(e, ({}, set()))).encode("utf-8", "surrogatepass")).hexdigest()


def _tree_ok(root):
    """Every node hangs where its parent pointer says, has only the standard fields, and types are
    free-standing: the properties the reload reproduces by construction."""
    base = _expr_base()
    seen = set()
    types = set()
    tstack = []
    stack = [(root, None, None, None)]
    while stack or tstack:
        if not stack:
            stack.append((tstack.pop(), None, None, None))
        n, par, key, idx = stack.pop()
        if id(n) in seen:
            return False            # shared node: one parent pointer cannot describe it
        seen.add(id(n))
        if par is not None and (n.parent is not par or n.arg_key != key or n.index != idx):
            return False
        if getattr(n, "__dict__", None):
            return False
        t = n._type
        if t is not None and isinstance(t, base):
            if t.parent is not None:
                return False
            if id(t) not in types:      # type objects may be shared between nodes; the reload keeps that
                types.add(id(t))
                tstack.append(t)
        for k, v in n.args.items():
            if isinstance(v, base):
                stack.append((v, n, k, None))
            elif type(v) is list:
                for i, x in enumerate(v):
                    if isinstance(x, base):
                        stack.append((x, n, k, i))
            elif isinstance(v, (tuple, dict, set, frozenset)):
                return False        # containers the parent bookkeeping does not cover
    return True


def _rules():
    import sqlglot.optimizer.optimizer as so
    return so, so.RULES


def _wrap(j, rule, call):
    import functools
    import inspect

    @functools.wraps(rule)
    def w(expression, *a, **k):
        return call(j, rule, expression, a, k)

    w.__signature__ = inspect.signature(rule)   # optimize() picks keyword arguments by signature
    del w.__wrapped__
    return w


def _load():
    if not _IDX:
        try:
            with open(PATH, "rb") as f:
                n = int.from_bytes(f.read(8), "big")
                idx = pickle.loads(f.read(n))
                idx["base"] = 8 + n
            _IDX.append(idx)
        except Exception:
            _IDX.append(None)
    return _IDX[0]


def _blob(idx, off, ln):
    with open(PATH, "rb") as f:
        f.seek(idx["base"] + off)
        return zlib.decompress(f.read(ln))


def eligible(lines_by_path, stage):
    """Rule indexes a patch (changed lines per file) reaches, or None when it can also reach the
    query stage outside the rules (then nothing here applies)."""
    if not stage or "R" not in stage or "G" not in stage:
        return None
    G, R = stage["G"], stage["R"]
    hit = set()
    for path, ls in lines_by_path.items():
        g = G.get(path)
        if g and (g & ls):
            return None
        for j, by_file in R.items():
            r = by_file.get(path)
            if r and (r & ls):
                hit.add(j)
    return hit


def install(affected):
    """Run time: wrap sqlglot.executor.execute so recorded queries go through the touched rules only."""
    import sqlglot.executor as se
    if getattr(se, "_br_rules", False):
        return
    idx = _load()
    if idx is None:
        return
    qmemo = memo._load().get("__query__", {})
    so, rules = _rules()
    if [r.__name__ for r in rules] != idx["rules"]:
        return
    affected = set(affected)
    ctx = {"q": None, "div": False}
    real_opt = se.optimize

    def call(j, rule, expression, a, k):
        ent = ctx["q"]
        if ent is None or ctx["div"]:
            return rule(expression, *a, **k)
        if j not in affected:
            return expression             # unchanged code: its recorded output is the next recorded input
        off, ln, want = ent[j]
        out = rule(pickle.loads(_blob(idx, off, ln)), *a, **k)
        if digest(out) != want:
            ctx["div"] = True
        return out

    wrapped = tuple(_wrap(j, r, call) for j, r in enumerate(rules))

    def optimize(expression, schema=None, **kw):
        if ctx["q"] is None or "rules" in kw:
            return real_opt(expression, schema, **kw)
        res = real_opt(expression, schema, rules=wrapped, **kw)
        if not ctx["div"]:
            raise _Match()
        return res

    orig = se.execute
    depth = [0]

    def execute(sql, schema=None, dialect=None, tables=None):
        if depth[0]:
            return orig(sql, schema, dialect, tables)
        try:
            k = memo.qkey(sql, schema, dialect, tables)
        except BaseException:
            k = None
        ent = idx["q"].get(k) if k is not None else None
        if ent is None or k not in qmemo:
            return orig(sql, schema, dialect, tables)
        depth[0] += 1
        ctx["q"] = ent
        ctx["div"] = False
        try:
            return orig(sql, schema, dialect, tables)
        except _Match:
            return pickle.loads(qmemo[k])
        finally:
            ctx["q"] = None
            depth[0] -= 1

    execute.__wrapped__ = orig
    se.optimize = optimize
    se.execute = execute
    se._br_rules = True
    for mn, m in list(sys.modules.items()):
        if m is not None and mn.startswith(("tests.", "sqlglot")) and getattr(m, "execute", None) is orig:
            m.execute = execute


def build(out=PATH):
    """Image build: record rule inputs/outputs and the lines each rule runs, on the pinned commit."""
    import unittest
    os.environ["SKIP_INTEGRATION"] = "0"
    from . import inlinepool
    inlinepool.install()
    import tests.test_executor as te
    import sqlglot.executor as se
    memo.EXEC_OK = False
    memo.install(te)
    qmemo = memo._load().get("__query__", {})
    so, rules = _rules()
    real_opt = se.optimize
    orig = se.execute
    state = {"q": None, "rule": None, "trace": False}
    rec = {}          # qk -> {j: (blob, digest)}
    bad = set()
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_limit, 20000))

    def call(j, rule, expression, a, k):
        qk = state["q"]
        if qk is None:
            return rule(expression, *a, **k)
        if state["trace"]:
            state["rule"] = j
            try:
                return rule(expression, *a, **k)
            finally:
                state["rule"] = None
        try:
            ok = _tree_ok(expression)
            blob = dumps(expression) if ok else None
            copy = pickle.loads(blob) if ok else None
            if ok and (digest(copy) != digest(expression) or not _tree_ok(copy)):
                ok = False
        except BaseException:
            ok = False
        res = rule(expression, *a, **k)
        if ok:
            try:
                want = digest(res)
                if digest(rule(copy, *a, **k)) != want:
                    ok = False
            except BaseException:
                ok = False
        if ok:
            rec.setdefault(qk, {})[j] = (blob, want)
        else:
            bad.add(qk)
        return res

    wrapped = tuple(_wrap(j, r, call) for j, r in enumerate(rules))

    def optimize(expression, schema=None, **kw):
        if state["q"] is None or "rules" in kw:
            return real_opt(expression, schema, **kw)
        return real_opt(expression, schema, rules=wrapped, **kw)

    depth = [0]

    def execute(sql, schema=None, dialect=None, tables=None):
        if depth[0]:
            return orig(sql, schema, dialect, tables)
        try:
            k = memo.qkey(sql, schema, dialect, tables)
        except BaseException:
            k = None
        if k is None or k not in qmemo:
            return orig(sql, schema, dialect, tables)
        depth[0] += 1
        state["q"] = k
        try:
            return orig(sql, schema, dialect, tables)
        finally:
            state["q"] = None
            depth[0] -= 1

    se.optimize = optimize
    se.execute = execute
    for mn, m in list(sys.modules.items()):
        if m is not None and mn.startswith(("tests.", "sqlglot")) and getattr(m, "execute", None) is orig:
            m.execute = execute
    names = [n for n in dir(te.TestExecutor) if n.startswith("test_")]

    def run():
        r = unittest.TestResult()
        unittest.TestSuite([te.TestExecutor(n) for n in names]).run(r)
        return r

    r = run()
    if r.failures or r.errors:
        print("rule memo: baseline failures", [str(x[0]) for x in r.failures + r.errors][:5], file=sys.stderr)
    # second pass: which lines run inside each rule, and which in the query stage outside the rules
    root = os.getcwd() + "/"
    R = {j: set() for j in range(len(rules))}
    G = set()
    mon = sys.monitoring
    tid = mon.COVERAGE_ID

    def on_line(code, line):
        fn = code.co_filename
        if not fn.startswith(root):
            return mon.DISABLE
        if state["q"] is not None:
            j = state["rule"]
            (G if j is None else R[j]).add((fn[len(root):], line))
        return None

    def on_start(code, off):
        return on_line(code, code.co_firstlineno)

    state["trace"] = True
    mon.use_tool_id(tid, "br-rules")
    mon.register_callback(tid, mon.events.LINE, on_line)
    mon.register_callback(tid, mon.events.PY_START, on_start)
    mon.set_events(tid, mon.events.LINE | mon.events.PY_START)
    try:
        run()
    finally:
        mon.set_events(tid, 0)
        mon.free_tool_id(tid)
        sys.setrecursionlimit(old_limit)

    def by_file(pairs):
        d = {}
        for f_, l_ in pairs:
            d.setdefault(f_, set()).add(l_)
        return d

    idx = {"rules": [r_.__name__ for r_ in rules], "q": {}}
    blobs = []
    off = 0
    for qk, per in rec.items():
        if qk in bad or len(per) != len(rules):
            continue
        ent = {}
        for j, (blob, want) in per.items():
            z = zlib.compress(blob, 1)
            ent[j] = (off, len(z), want)
            blobs.append(z)
            off += len(z)
        idx["q"][qk] = ent
    head = pickle.dumps(idx, protocol=5)
    with open(out, "wb") as f:
        f.write(len(head).to_bytes(8, "big"))
        f.write(head)
        for z in blobs:
            f.write(z)
    try:
        with open(memo.STAGE_PATH, "rb") as f:
            st = pickle.load(f)
    except Exception:
        st = {}
    st["R"] = {j: by_file(v) for j, v in R.items()}
    st["G"] = by_file(G)
    with open(memo.STAGE_PATH, "wb") as f:
        pickle.dump(st, f, protocol=5)
    print("rule memo: %d queries kept, %d dropped, %d bytes; lines outside rules %d, inside %s" % (
        len(idx["q"]), len(set(rec) | bad) - len(idx["q"]), off, len(G), [len(R[j]) for j in sorted(R)]), file=sys.stderr)


if __name__ == "__main__":
    from br import rulememo as _rm   # so that recorded trees refer to br.rulememo, not __main__
    _rm.build()
