"""Cache of DuckDB reference results for the TPC executor tests.

TestExecutor compares sqlglot's Python executor against DuckDB run on the same fixed CSV data.
DuckDB's answer is a pure function of the SQL text it is handed, so it is computed once when the
image is built (from the pinned commit) and reused; the text itself is still produced by the
patched transpiler at run time, and any text not seen before is executed for real.
"""
import os
import pickle

PATH = os.environ.get("BR_MEMO", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "maps", "duck_memo.pkl"))
MEMO = None
RECORD = None


def _load():
    global MEMO
    if MEMO is None:
        try:
            with open(PATH, "rb") as f:
                MEMO = pickle.load(f)
        except Exception:
            MEMO = {}
    return MEMO


def install(mod):
    """Replace TestExecutor.cached_execute in tests.test_executor with a memoising equivalent."""
    cls = getattr(mod, "TestExecutor", None)
    if cls is None or getattr(cls, "_br_memo", False):
        return
    memo = _load()

    def cached_execute(self, sql, tpch=True):
        conn = self.tpch_conn if tpch else self.tpcds_conn
        if sql not in self.cache:
            q = mod.transpile(sql, write="duckdb")[0]
            key = (bool(tpch), q)
            df = memo.get(key)
            if df is None:
                df = conn.execute(q).fetchdf()
                if RECORD is not None:
                    RECORD[key] = df.copy()
            else:
                df = df.copy()
            self.cache[sql] = df
        return self.cache[sql]

    cls.cached_execute = cached_execute
    cls._br_memo = True
    rows = memo.get("__rows__")
    if rows:
        _fast_setup(mod, cls, rows)
        if EXEC_OK and RECORD is None:
            try:
                if EXEC_OK == "A":
                    install_query()
                else:
                    install_exec()
            except Exception:
                pass


class _LazyConn:
    """DuckDB connection over the TPC CSV views, opened only if a query is not in the cache."""

    def __init__(self, mod, directory, schema):
        self._args = (mod, directory, schema)
        self._conn = None

    def _real(self):
        if self._conn is None:
            mod, directory, schema = self._args
            conn = mod.duckdb.connect()
            for table, columns in schema.items():
                file_name = f"{directory}{table}.csv.gz"
                conn.execute(
                    f"""
                CREATE VIEW {table} AS
                SELECT *
                FROM READ_CSV('{file_name}', delim='|', header=True, columns={columns})
                """
                )
            self._conn = conn
        return self._conn

    def execute(self, *a, **k):
        return self._real().execute(*a, **k)

    def close(self):
        if self._conn is not None:
            self._conn.close()

    def __getattr__(self, name):
        return getattr(self._real(), name)


def _fast_setup(mod, cls, rows):
    """setUpClass with the CSV parsing replaced by its cached result (same objects otherwise)."""

    def setUpClass(c):
        c.tpch_conn = _LazyConn(mod, mod.DIR_TPCH, mod.TPCH_SCHEMA)
        c.tpcds_conn = _LazyConn(mod, mod.DIR_TPCDS, mod.TPCDS_SCHEMA)
        c.tpch_tables = {}
        c.tpcds_tables = {}
        for table, columns in mod.TPCH_SCHEMA.items():
            c.tpch_tables[table] = t = mod.Table(columns=columns, rows=list(rows[("tpch", table)]))
            _tag(t, "tpch." + table)
        for table, columns in mod.TPCDS_SCHEMA.items():
            c.tpcds_tables[table] = t = mod.Table(columns=columns, rows=list(rows[("tpcds", table)]))
            _tag(t, "tpcds." + table)
        c.cache = {}
        c.tpch_sqls = list(mod.load_sql_fixture_pairs("optimizer/tpc-h/tpc-h.sql"))
        c.tpcds_sqls = list(mod.load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql"))

    cls.setUpClass = classmethod(setUpClass)


def _tag(t, name):
    try:
        t._br_tag = name
    except Exception:
        pass


def _read_rows(mod):
    """The row-parsing half of TestExecutor.setUpClass, verbatim."""
    import ast
    import csv
    out = {}
    for tag, directory, schema in (("tpch", mod.DIR_TPCH, mod.TPCH_SCHEMA), ("tpcds", mod.DIR_TPCDS, mod.TPCDS_SCHEMA)):
        for table, columns in schema.items():
            file_name = f"{directory}{table}.csv.gz"
            reader = csv.reader(mod.open_file(file_name), delimiter="|")
            rows = []
            ctypes = []
            next(reader)
            for row in reader:
                if not ctypes:
                    for i, v in enumerate(row):
                        try:
                            ctypes.append(type(ast.literal_eval(v)))
                        except (ValueError, SyntaxError):
                            ctypes.append(str)
                rows.append(tuple(None if (t is not str and v == "") else t(v) for t, v in zip(ctypes, row)))
            out[(tag, table)] = rows
    return out


# ---- executor stage memo -------------------------------------------------------------------------
# PythonExecutor.execute(plan) is a pure function of the plan and the tables it reads. Its results on
# the pinned commit are stored under a structural key of the plan, computed by walking the objects
# directly (never through sqlglot's own equality, hashing or SQL generation, which a patch may touch).
# The memo is used only when the patch cannot reach any line the executor stage runs.
EXEC_OK = False
STAGE_PATH = os.path.join(os.path.dirname(PATH), "exec_stage.pkl")
_PRIM = (type(None), bool, int, float, str, bytes)


def _key(o, seen):
    """Structural key that does not depend on object identity or set iteration order.
    `seen` is (cache by id, ids on the current path)."""
    t = type(o)
    if t in _PRIM:
        return (t.__name__, o)
    if t in (list, tuple):
        return ("l", tuple(_key(x, seen) for x in o))
    if t is dict:
        return ("d", tuple((_key(k, seen), _key(v, seen)) for k, v in o.items()))   # insertion order kept
    if t in (set, frozenset):
        return ("s", tuple(sorted((_key(x, seen) for x in o), key=repr)))
    if isinstance(o, type):
        return ("t", o.__module__, o.__qualname__)
    import enum
    if isinstance(o, enum.Enum):
        return ("e", t.__module__, t.__qualname__, o.name)
    mod = t.__module__ or ""
    cache, path = seen
    i = id(o)
    if i in path:
        return ("cyc",)
    hit = cache.get(i)
    if hit is not None:
        return hit
    tag = getattr(o, "_br_tag", None)
    if tag is not None:
        return ("tag", tag)
    if mod.startswith("sqlglot.expressions"):
        d = getattr(o, "args", None)
        if type(d) is not dict:
            return ("?", i)
        path.add(i)
        try:
            extra = []
            for name in ("_type", "_comments", "comments", "_meta"):
                v = getattr(o, name, None)
                if v is not None:
                    extra.append((name, _key(v, seen)))
            r = ("E", t.__qualname__, tuple((k, _key(v, seen)) for k, v in d.items()), tuple(extra))
        finally:
            path.discard(i)
        cache[i] = r
        return r
    if mod == "sqlglot.planner":   # tables must be the tagged TPC tables; anything else is not memoised
        try:
            d = vars(o)
        except TypeError:
            return ("?", i)
        path.add(i)
        try:
            deps = d.get("dependents")
            dn = tuple(sorted((type(x).__qualname__, str(getattr(x, "name", None))) for x in deps)) \
                if isinstance(deps, (set, frozenset, list, tuple)) else ()
            r = ("O", mod, t.__qualname__, dn, tuple((k, _key(v, seen)) for k, v in sorted(d.items())
                                                      if k not in ("dependents", "_dag", "reader", "range_reader")))
        finally:
            path.discard(i)
        cache[i] = r
        return r
    return ("?", i)


def exec_key(executor, plan):
    import hashlib
    seen = ({}, set())
    k = (_key(plan.root, seen), _key(getattr(plan, "ctes", None), seen),
         _key(getattr(executor, "tables", None) and getattr(executor.tables, "mapping", None), seen))
    r = repr(k)
    if "'?'" in r:
        return None
    return hashlib.sha1(r.encode("utf-8", "surrogatepass")).hexdigest()


def install_exec(record=None, stage=None):
    """Wrap PythonExecutor.execute with the stage memo (or with a recorder at build time)."""
    import sqlglot.executor.python as ep
    cls = ep.PythonExecutor
    if getattr(cls, "_br_exec", False):
        return
    orig = cls.execute
    memo = _load().get("__exec__", {}) if record is None else None

    depth = [0]

    def execute(self, plan, outer_scope=None):
        # only whole top-level queries are memoised; correlated sub-plans depend on the outer row
        if depth[0] or outer_scope is not None:
            return orig(self, plan, outer_scope)
        try:
            k = exec_key(self, plan)
        except BaseException:
            k = None
        if record is None and k is not None:
            blob = memo.get(k)
            if blob is not None:
                return pickle.loads(blob)
        depth[0] += 1
        if stage is not None:
            stage[0] += 1
        try:
            res = orig(self, plan, outer_scope)
        finally:
            depth[0] -= 1
            if stage is not None:
                stage[0] -= 1
        if record is not None and k is not None:
            try:
                record[k] = pickle.dumps(res, protocol=5)
            except Exception:
                pass
        return res

    cls.execute = execute
    cls._br_exec = True


def install_query(record=None, stage=None):
    """Level A: memoise sqlglot.executor.execute(expression, schema, tables) on the parsed query."""
    import hashlib
    import sqlglot.executor as se
    if getattr(se, "_br_query", False):
        return
    orig = se.execute
    memo = _load().get("__query__", {}) if record is None else None
    depth = [0]

    def qkey(sql, schema, dialect, tables):
        if isinstance(sql, str) or dialect is not None or type(tables) is not dict:
            return None
        seen = ({}, set())
        r = repr((_key(sql, seen), _key(schema, seen), _key(tables, seen)))
        if "'?'" in r:
            return None
        return hashlib.sha1(r.encode("utf-8", "surrogatepass")).hexdigest()

    def execute(sql, schema=None, dialect=None, tables=None):
        if depth[0]:
            return orig(sql, schema, dialect, tables)
        try:
            k = qkey(sql, schema, dialect, tables)
        except BaseException:
            k = None
        if record is None and k is not None:
            blob = memo.get(k)
            if blob is not None:
                return pickle.loads(blob)
        depth[0] += 1
        if stage is not None:
            stage[0] += 1
        try:
            res = orig(sql, schema, dialect, tables)
        finally:
            depth[0] -= 1
            if stage is not None:
                stage[0] -= 1
        if record is not None and k is not None:
            try:
                record[k] = pickle.dumps(res, protocol=5)
            except Exception:
                pass
        return res

    execute.__wrapped__ = orig
    se.execute = execute
    se._br_query = True
    import sys as _sys
    for mn, m in list(_sys.modules.items()):
        if m is not None and mn.startswith(("tests.", "sqlglot")) and getattr(m, "execute", None) is orig:
            m.execute = execute


def build(out=PATH):
    """Run the executor tests once on the unpatched tree and store every DuckDB result."""
    global RECORD, MEMO
    import sys
    import unittest
    os.environ["SKIP_INTEGRATION"] = "0"
    from . import inlinepool
    inlinepool.install()
    import tests.test_executor as te
    MEMO = {"__rows__": _read_rows(te)}   # same class setup as at run time, so tables carry their tags
    RECORD = {}
    install(te)
    # executor stage: record results and the lines the stage runs
    rec_exec = {}
    inside = [0]
    inside_q = [0]
    stage_lines = set()
    stage_q = set()
    rec_query = {}
    root = os.getcwd() + "/"
    mon = sys.monitoring
    tid = mon.COVERAGE_ID

    def on_line(code, line):
        fn = code.co_filename
        if not fn.startswith(root):
            return mon.DISABLE
        if inside_q[0]:
            stage_q.add((fn[len(root):], line))
            if inside[0]:
                stage_lines.add((fn[len(root):], line))
                return mon.DISABLE
        return None

    def on_start(code, off):
        # a body on the def/lambda line raises no LINE event: record function entry as its first line
        return on_line(code, code.co_firstlineno)

    mon.use_tool_id(tid, "br-stage")
    mon.register_callback(tid, mon.events.LINE, on_line)
    mon.register_callback(tid, mon.events.PY_START, on_start)
    mon.set_events(tid, mon.events.LINE | mon.events.PY_START)
    install_exec(rec_exec, inside)
    install_query(rec_query, inside_q)
    names = [n for n in dir(te.TestExecutor) if n.startswith("test_")]
    r = unittest.TestResult()
    unittest.TestSuite([te.TestExecutor(n) for n in names]).run(r)
    if r.failures or r.errors:
        print("memo build: baseline failures", [str(x[0]) for x in r.failures + r.errors][:5], file=sys.stderr)
    mon.set_events(tid, 0)
    RECORD["__rows__"] = MEMO["__rows__"]
    RECORD["__exec__"] = rec_exec
    RECORD["__query__"] = rec_query
    by_file = {}
    for f_, l_ in stage_lines:
        by_file.setdefault(f_, set()).add(l_)
    by_file_q = {}
    for f_, l_ in stage_q:
        by_file_q.setdefault(f_, set()).add(l_)
    with open(STAGE_PATH, "wb") as f:
        pickle.dump({"B": by_file, "A": by_file_q}, f, protocol=5)
    print("query memo entries", len(rec_query), "stage A lines", len(stage_q), file=sys.stderr)
    print("exec memo entries", len(rec_exec), "stage lines", len(stage_lines), file=sys.stderr)
    with open(out, "wb") as f:
        pickle.dump(RECORD, f, protocol=5)
    print("duck memo entries", len(RECORD), file=sys.stderr)


if __name__ == "__main__":
    build()
