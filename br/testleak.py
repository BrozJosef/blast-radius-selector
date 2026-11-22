"""Test-side edits that can leak state to other tests.

One pytest session imports every test module before running any test and runs all tests in one
process. A test-side edit that touches shared state (a removed cleanup, a module-level monkeypatch,
`patcher.start()` without `stop()`, an assignment to an attribute of a library object) can therefore
fail tests in other files, which no per-test coverage shows. `may_leak` compares the statements of the
old and new test file and reports whether any added, removed or changed statement could do that.
It errs on the side of "yes": then every test is selected.
"""
import ast
import collections

_MUTATORS = {"update", "pop", "popitem", "clear", "append", "extend", "insert", "remove", "add", "discard",
             "setdefault", "sort", "reverse", "start", "stop", "register", "unregister", "cache_clear", "reset",
             "set", "addCleanup", "enterContext", "enterClassContext", "seed", "chdir", "putenv",
             "setrecursionlimit", "setLevel", "disable", "addHandler", "removeHandler", "simplefilter",
             "filterwarnings"}
_FIXTURES = {"setUp", "tearDown", "setUpClass", "tearDownClass", "setUpModule", "tearDownModule",
             "asyncSetUp", "asyncTearDown"}
_STATE_NAMES = {"environ", "mock", "patch", "monkeypatch", "sys", "logging", "warnings", "importlib"}
_STATE_ATTRS = {"environ", "modules", "path", "patch", "pretty", "logger"}
_COMPOUND = (ast.If, ast.While, ast.For, ast.AsyncFor, ast.With, ast.AsyncWith, ast.Match)


def _root_name(node):
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Call, ast.Starred)):
        node = node.func if isinstance(node, ast.Call) else node.value
    return node.id if isinstance(node, ast.Name) else None


def _units(tree):
    """(scope, enclosing function name, statement) for every simple statement and compound header."""
    out = []

    def walk(body, scope, fn):
        for st in body:
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
                walk(st.body, "func", st.name if fn is None else fn)
                continue
            if isinstance(st, ast.ClassDef):
                walk(st.body, scope if scope == "func" else "class", fn)
                continue
            out.append((scope, fn, st))
            for name in ("body", "orelse", "finalbody"):
                sub = getattr(st, name, None)
                if isinstance(sub, list) and sub and isinstance(sub[0], ast.stmt):
                    walk(sub, scope, fn)
            for h in getattr(st, "handlers", ()) or ():
                walk(h.body, scope, fn)
            for c in getattr(st, "cases", ()) or ():
                walk(c.body, scope, fn)

    walk(tree.body, "module", None)
    return out


def _key(st):
    if isinstance(st, (ast.If, ast.While)):
        return "H:" + type(st).__name__ + ast.dump(st.test)
    if isinstance(st, (ast.For, ast.AsyncFor)):
        return "H:For" + ast.dump(st.target) + ast.dump(st.iter)
    if isinstance(st, (ast.With, ast.AsyncWith)):
        return "H:With" + "".join(ast.dump(i) for i in st.items)
    if isinstance(st, ast.Try) or type(st).__name__ == "TryStar":
        return "H:Try%d,%d,%d" % (len(st.handlers), len(st.orelse), len(st.finalbody)) + "".join(
            ast.dump(h.type) if h.type is not None else "-" for h in st.handlers)
    if isinstance(st, ast.Match):
        return "H:Match" + ast.dump(st.subject)
    return ast.dump(st)


def _own_nodes(st):
    """The expression nodes that belong to this unit itself (a compound statement's header only)."""
    if isinstance(st, (ast.If, ast.While)):
        parts = [st.test]
    elif isinstance(st, (ast.For, ast.AsyncFor)):
        parts = [st.target, st.iter]
    elif isinstance(st, (ast.With, ast.AsyncWith)):
        parts = [x for i in st.items for x in (i.context_expr, i.optional_vars) if x is not None]
    elif isinstance(st, ast.Match):
        parts = [st.subject]
    else:
        parts = [st]
    for p in parts:
        for n in ast.walk(p):
            yield n


def _stmt_may_leak(scope, fn, st):
    if scope != "func":
        # module / class level of test code runs at collection, before every test of the session
        if isinstance(st, (ast.Import, ast.ImportFrom, ast.Pass)):
            return False
        if isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant):
            return False
        if isinstance(st, (ast.Assign, ast.AnnAssign)):
            tg = st.targets if isinstance(st, ast.Assign) else [st.target]
            if all(isinstance(t, ast.Name) for t in tg) and not any(
                    isinstance(n, (ast.Call, ast.Await, ast.Yield, ast.YieldFrom, ast.NamedExpr)) for n in ast.walk(st)):
                return False        # a name of the test module bound to a literal structure
        return True
    if fn in _FIXTURES:
        return True
    if isinstance(st, (ast.Global, ast.Nonlocal, ast.Try)) or type(st).__name__ == "TryStar":
        return True
    for n in _own_nodes(st):
        if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)):
            tg = n.targets if isinstance(n, (ast.Assign, ast.Delete)) else [n.target]
            for t in tg:
                for leaf in (list(t.elts) if isinstance(t, (ast.Tuple, ast.List)) else [t]):
                    if isinstance(leaf, (ast.Attribute, ast.Subscript)) and _root_name(leaf) != "self":
                        return True
        elif isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name) and f.id in ("setattr", "delattr", "exec", "eval", "globals", "__import__"):
                return True
            if isinstance(f, ast.Attribute) and f.attr in _MUTATORS:
                return True
        elif isinstance(n, ast.Name) and n.id in _STATE_NAMES:
            return True
        elif isinstance(n, ast.Attribute) and n.attr in _STATE_ATTRS:
            return True
    return False


def may_leak(old, new):
    try:
        tb = ast.parse(new)
    except SyntaxError:
        return False        # the file cannot be imported at all: none of its code runs
    except Exception:
        return True
    try:
        ua, ub = _units(ast.parse(old)), _units(tb)
    except Exception:
        return True
    ca = collections.Counter((sc, fn, _key(st)) for sc, fn, st in ua)
    cb = collections.Counter((sc, fn, _key(st)) for sc, fn, st in ub)
    for units, mine, other in ((ua, ca, cb), (ub, cb, ca)):
        for sc, fn, st in units:
            k = (sc, fn, _key(st))
            if mine[k] > other.get(k, 0) and _stmt_may_leak(sc, fn, st):
                return True
    return False


# ---- library code that writes process-wide state ---------------------------------------------------
# A patch inside a function that fills a registry or cache shared by the whole process (a class-level
# or module-level table, an lru_cache) can poison that state for tests that never run the changed
# line themselves; whether they fail then depends on the order of the pytest session.
_CACHE_DECOS = {"lru_cache", "cache"}
_WRITE_CALLS = {"update", "pop", "popitem", "clear", "append", "extend", "insert", "remove", "add", "discard",
                "setdefault", "sort", "reverse", "register"}


def _locals_of(fn):
    names = {a.arg for a in fn.args.args + fn.args.kwonlyargs + fn.args.posonlyargs}
    if fn.args.vararg:
        names.add(fn.args.vararg.arg)
    if fn.args.kwarg:
        names.add(fn.args.kwarg.arg)
    for n in ast.walk(fn):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            names.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n is not fn:
            names.add(n.name)
        elif isinstance(n, ast.alias):
            names.add((n.asname or n.name).split(".")[0])
    return names


def writes_shared_state(fn):
    """fn: ast.FunctionDef. True if its body can write a table that outlives the call and is not
    reached through `self` or a local."""
    for d in fn.decorator_list:
        r = d.func if isinstance(d, ast.Call) else d
        nm = r.attr if isinstance(r, ast.Attribute) else getattr(r, "id", None)
        if nm in _CACHE_DECOS:
            return True
    loc = _locals_of(fn) - {"cls"}
    loc.discard("self")

    def shared(node):
        root = _root_name(node)
        return root is not None and root != "self" and root not in loc

    for n in ast.walk(fn):
        if isinstance(n, ast.Global):
            return True
        if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)):
            tg = n.targets if isinstance(n, (ast.Assign, ast.Delete)) else [n.target]
            for t in tg:
                for leaf in (list(t.elts) if isinstance(t, (ast.Tuple, ast.List)) else [t]):
                    if isinstance(leaf, (ast.Attribute, ast.Subscript)) and shared(leaf):
                        return True
        elif isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name) and f.id in ("setattr", "delattr") and n.args and shared(n.args[0]):
                return True
            if isinstance(f, ast.Attribute) and f.attr in _WRITE_CALLS and shared(f.value):
                return True
    return False


def changed_state_writers(old, new, ranges_old):
    """Names of functions that overlap the changed old lines and write shared state (in either version)."""
    out = []
    try:
        ta = ast.parse(old)
    except Exception:
        return ["<unparsable>"]
    try:
        tb = ast.parse(new)
    except Exception:
        tb = None
    new_by_name = {}
    if tb is not None:
        for n in ast.walk(tb):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                new_by_name.setdefault(n.name, []).append(n)
    for n in ast.walk(ta):
        if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        lo = min([n.lineno] + [d.lineno for d in n.decorator_list])
        if not any(a <= n.end_lineno and b >= lo for a, b in ranges_old):
            continue
        if writes_shared_state(n) or any(writes_shared_state(m) for m in new_by_name.get(n.name, ())):
            out.append(n.name)
    return out
