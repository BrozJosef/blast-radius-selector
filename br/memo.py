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


def build(out=PATH):
    """Run the executor tests once on the unpatched tree and store every DuckDB result."""
    global RECORD, MEMO
    import sys
    import unittest
    os.environ["SKIP_INTEGRATION"] = "0"
    from . import inlinepool
    inlinepool.install()
    import tests.test_executor as te
    MEMO = {}
    RECORD = {}
    install(te)
    names = [n for n in dir(te.TestExecutor) if n.startswith("test_")]
    r = unittest.TestResult()
    unittest.TestSuite([te.TestExecutor(n) for n in names]).run(r)
    if r.failures or r.errors:
        print("memo build: baseline failures", [str(x[0]) for x in r.failures + r.errors][:5], file=sys.stderr)
    with open(out, "wb") as f:
        pickle.dump(RECORD, f, protocol=5)
    print("duck memo entries", len(RECORD), file=sys.stderr)


if __name__ == "__main__":
    build()
