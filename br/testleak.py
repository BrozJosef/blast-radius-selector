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
