"""Static selection: from a patch to a set of candidate tests (a superset of the tests it can break)."""
import os
import pickle
import zlib
from array import array

from . import patchlib, pydiff

SRC = os.environ.get("BR_SRC", "/app/src")
USE_VP = not os.environ.get("BR_NOVP")
VP_INTERSECT = False
PYTEST_CFG = ("conftest.py", "pytest.ini", "tox.ini", "setup.cfg", "pyproject.toml", ".pytest.ini", "pytest.toml", ".pytest.toml")
INERT_DIRS = ("benchmarks", "pdoc", "posts", "sqlglotc", ".github", "sqlglot-integration-tests")


class _PerKey:
    """dict of zlib+pickle blobs, decoded on first access."""

    def __init__(self, blobs):
        self._b = blobs
        self._d = {}

    def get(self, k, default=None):
        v = self._d.get(k, self)
        if v is self:
            b = self._b.get(k)
            v = self._d[k] = pickle.loads(zlib.decompress(b)) if b is not None else None
        return default if v is None else v

    def __contains__(self, k):
        return k in self._b

    def __getitem__(self, k):
        v = self.get(k)
        if v is None:
            raise KeyError(k)
        return v

    def __iter__(self):
        return iter(self._b)

    def keys(self):
        return self._b.keys()


class Maps:
    def __init__(self, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        self._lazy = {k: d.pop(k) for k in d.pop("__lazy__")}
        for k in d.pop("__perkey__"):
            d[k] = _PerKey(d[k])
        self.__dict__.update(d)
        self.n = len(self.tests)
        self.idx = {t: i for i, t in enumerate(self.tests)}
        self.by_file = {}
        for i, t in enumerate(self.tests):
            self.by_file.setdefault(t.split("::")[0], set()).add(i)
        self._mod_users = {}
        self._importers = None
        self._cls_by_file = None

    def __getattr__(self, name):
        lz = self.__dict__.get("_lazy")
        if lz is not None and name in lz:
            v = pickle.loads(zlib.decompress(lz.pop(name)))
            self.__dict__[name] = v
            return v
        raise AttributeError(name)

    @staticmethod
    def un(b):
        a = array("H")
        a.frombytes(b)
        return a

    def cover(self, f, lines):
        d = self.lines.get(f)
        out = set()
        if d:
            for l in lines:
                b = d.get(l)
                if b:
                    out.update(self.un(b))
        return out

    def cover_span(self, f, a, b):
        d = self.lines.get(f)
        out = set()
        if d:
            for l, bts in d.items():
                if a <= l <= b:
                    out.update(self.un(bts))
        return out

    def readers(self, f):
        b = self.files.get(f)
        return set(self.un(b)) if b else set()

    def mod(self, f):
        m = f[:-3].replace("/", ".")
        return m[:-9] if m.endswith(".__init__") else m

    def classes_in(self, f):
        if self._cls_by_file is None:
            self._cls_by_file = {}
            for k, (cf, a, b) in self.classes.items():
                self._cls_by_file.setdefault(cf, []).append((k, a, b))
        return self._cls_by_file.get(f, [])

    def users_class(self, f, scope):
        key = self.mod(f) + ":" + scope
        out = set()
        b = self.usage.get(key)
        if b:
            out.update(self.un(b))
        c = self.classes.get(key)
        if c:
            out |= self.cover_span(f, c[1], c[2])
        # nested classes share fate with their users too
        for k, a, e in self.classes_in(f):
            if k.startswith(key + "."):
                b = self.usage.get(k)
                if b:
                    out.update(self.un(b))
        return out

    def users_module(self, f):
        """Tests that run code of, or use classes defined in, module file f."""
        r = self._mod_users.get(f)
        if r is None:
            r = set()
            d = self.lines.get(f)
            if d:
                for bts in d.values():
                    r.update(self.un(bts))
            for k, a, e in self.classes_in(f):
                b = self.usage.get(k)
                if b:
                    r.update(self.un(b))
            if f.startswith("tests/"):
                r |= self.by_file.get(f, set())
            self._mod_users[f] = r
        return r

    def importers(self, f, transitive=True):
        if self._importers is None:
            inv = {}
            for a, bs in self.imports.items():
                for b in bs:
                    inv.setdefault(b, set()).add(a)
            self._importers = inv
        seen = set()
        todo = [f]
        while todo:
            x = todo.pop()
            for y in self._importers.get(x, ()):
                if y not in seen:
                    seen.add(y)
                    if transitive:
                        todo.append(y)
        seen.discard(f)
        return seen

    def dependents(self, f, transitive=False):
        out = set(self.users_module(f))
        for g in self.importers(f, transitive):
            out |= self.users_module(g)
        return out

    def is_deferred(self, f, line, col=None):
        """Is this position inside a function or lambda body (code that runs when called)?"""
        for l1, c1, l2, c2 in self.deferred.get(f, ()):
            if col is None:
                if l1 < line < l2 or (l1 == l2 == line):
                    return True
                continue
            if (line, col) >= (l1, c1) and (line, col) < (l2, c2):
                return True
        return False

    def import_time_line(self, f, line, col=None):
        """True when the position runs (or may run) while modules are being imported."""
        if not self.is_deferred(f, line, col):
            return True
        d = self.imp.get(f)
        return bool(d and line in d)

    def name_sites(self, name):
        return list(self.reads.get(name, ())) + list(self.strs.get(name, ()))


_ORDER = {"<": 2, "<=": 3, ">": 4, ">=": 5}


def _cmp_val(op, m):
    """Value of `l op r` given a recorded operand pattern; None when unknown."""
    if op == "is" or op == "is not":
        v = m & 3
        neg = op == "is not"
    elif op == "==" or op == "!=":
        v = (m >> 2) & 3
        neg = op == "!="
    elif op in _ORDER:
        if m >> 4 == 0 and False:
            return None
        v = (m >> (2 * _ORDER[op])) & 3
        neg = False
    else:
        return None
    if v == 2:
        return None
    return (not v) if neg else bool(v)


def site_pred(M, path, site):
    """For an operator swap at `site`: (sid, kind, differs(mask) -> bool), or None when unknown."""
    line, col, kind, old, new, n = site
    info = M.vsite.get(path, {}).get((line, col))
    if info is None:
        return None
    sid, skind, sop, sn, boolctx, pure = info
    if skind != kind or (old is not None and sop != old):
        return None
    if kind == "cmp":
        if new is None or old is None:
            return None
        order_site = sop in _ORDER

        def differs(m):
            if (new in _ORDER) and not order_site:
                return True   # ordering was never recorded at an equality site
            a, b = _cmp_val(old, m), _cmp_val(new, m)
            return a is None or b is None or a != b
        return sid, "c", differs
    if old is None or new is None or n != 2 or sn != 2 or not boolctx or not pure:
        return sid, "b", (lambda m: True)   # any evaluation of the expression

    def differs_b(m):
        i, t = divmod(m, 3)
        if t == 2:
            return True
        # `a and b` -> `a or b` differs when a is falsy (b unknown) or b is falsy; mirrored for `or`
        return t == 0 if old == "and" else t == 1
    return sid, "b", differs_b


def infected(M, path, site):
    """Tests in which swapping the operator at `site` can change the value there; None = unknown."""
    sp = site_pred(M, path, site)
    if sp is None:
        return None
    sid, which, differs = sp
    out = set()
    for m, tb in (M.vc if which == "c" else M.vb).get(sid, {}).items():
        if differs(m):
            out.update(M.un(tb))
    return out


class Plan:
    def __init__(self):
        self.all = False
        self.cands = set()
        self.reasons = []
        self.overlay = False     # needs a real patched tree (import-level effects)
        self.hot = []            # (file, FuncChange) patchable in place
        self.no_exec = False     # execution cannot decide (e.g. binary patch)
        self.files = {}          # path -> new content (None = deleted)
        self.renames = []
        self.import_probe = set()  # modules whose import may now raise
        self.broad = set()         # what fails if such an import does raise
        self.broad_all = False
        self.sites = {}            # path -> operator-swap sites, when every in-function change is one
        self.sites_only = True
        self.lines = {}            # path -> old lines touched by in-function changes
        self.lines_ok = True       # False when some change is not captured by `lines`
        self.data_files = {}       # changed data file -> (old text, new text)
        self.ranges = {}           # path -> (old line ranges, new line ranges) touched by the patch
        self.attr_keys = set()     # "module:Class.attr" keys the patch assigns; None when not expressible
        self.ck = {}               # attr key -> (changed entry indexes, entry count) for value-only dict edits; None if any other kind
        self.attr_rt = set()       # tests that read those attributes at run time
        self.attr_classes = set()  # "module:Class" whose body the patch changes; None when not only that
        self.attr_change = False   # some module/class-level statement changed (import state surely differs)
        self.must = set()          # included whatever execution says (the overlay cannot be trusted for them)
        self.header_funcs = set()  # "module:qualname" of functions whose def header changed
        self.rt = set()            # candidates justified by run-time execution alone
        self.fp_ok = True          # False when some change is not plain sqlglot code or data
        self.data_cands = set()    # candidates that read a changed data file (almost surely failing)

    def why(self, s):
        self.reasons.append(s)


def read_src(path):
    try:
        with open(os.path.join(SRC, path), encoding="utf-8", newline="") as f:
            return f.read()
    except (OSError, UnicodeDecodeError):
        return None


def exists(path):
    return os.path.exists(os.path.join(SRC, path))


def is_test_module(f):
    b = os.path.basename(f)
    return f.startswith("tests/") and b.startswith("test_") and b.endswith(".py")


def analyze(patch_text, M):
    plan = Plan()
    try:
        fcs = patchlib.parse(patch_text)
    except Exception as e:
        plan.all = True
        plan.why("unparsable patch: %r" % (e,))
        return plan
    if not fcs:
        plan.why("empty patch")
        return plan
    for fc in fcs:
        try:
            _file(fc, M, plan)
        except Exception as e:  # never let an analysis bug become a miss
            plan.all = True
            plan.why("analysis error on %s: %r" % (fc.path, e))
    return plan


def _all_dependents(f, M, plan, why):
    """Module f is broken/changed at import level in an unknown way."""
    plan.overlay = True
    deps = M.dependents(f, transitive=True)
    plan.cands |= deps
    plan.why("%s: %s -> %d dependents" % (f, why, len(deps)))


def _file(fc, M, plan):
    old, new = fc.old, fc.new
    if fc.mode_only and not fc.hunks:
        plan.why("%s: mode change only" % fc.path)
        return
    # ---- renames / copies -------------------------------------------------
    if (fc.rename and old != new) or fc.binary:
        plan.fp_ok = False
    if fc.rename and old != new:
        plan.renames.append((old, new))
        plan.overlay = True
        if old is not None and exists(old):
            _removed(old, M, plan, "renamed away")
        if new is not None:
            _added(new, M, plan, None)
        if fc.hunks or fc.binary:
            plan.all = plan.all or fc.binary
        try:
            src = read_src(old)
            plan.files[old] = None
            plan.files[new] = patchlib.apply(src, fc) if fc.hunks else src
        except Exception:
            plan.no_exec = True
        return
    path = fc.path
    if fc.binary:
        plan.no_exec = True
        if exists(path) and not fc.new_file:
            r = M.readers(path)
            plan.cands |= r
            plan.why("%s: binary change -> %d readers" % (path, len(r)))
            if path.endswith(".py"):
                plan.all = True
        else:
            plan.why("%s: binary add" % path)
        return
    if fc.deleted or new is None:
        plan.files[path] = None
        plan.overlay = True
        _removed(path, M, plan, "deleted")
        return
    src = read_src(path) if not fc.new_file else None
    if fc.new_file or src is None:
        if not fc.new_file and src is None and exists(path):
            plan.all = True
            plan.why("%s: unreadable original" % path)
            return
        try:
            content = patchlib.apply(None if fc.new_file else src, fc)
        except Exception as e:
            plan.all = True
            plan.why("%s: cannot apply (%r)" % (path, e))
            return
        if exists(path) and fc.new_file:
            plan.all = True
            plan.why("%s: 'new' file already exists" % path)
            return
        plan.files[path] = content
        plan.overlay = True
        _added(path, M, plan, content)
        return
    try:
        content = patchlib.apply(src, fc)
    except Exception as e:
        plan.all = True
        plan.why("%s: cannot apply (%r)" % (path, e))
        return
    if content == src:
        plan.why("%s: no content change" % path)
        return
    plan.files[path] = content
    if path.endswith(".py"):
        plan.ranges[path] = patchlib.changed_ranges(fc)
        _python(path, src, content, M, plan, plan.ranges[path])
    else:
        plan.data_files[path] = (src, content)
        _data(path, M, plan)


def _data(path, M, plan):
    r = M.readers(path)
    plan.cands |= r
    plan.rt |= r
    plan.data_cands |= r
    if not path.startswith("tests/"):
        plan.must |= r   # tests reach such files through resolved paths that leave the overlay
    plan.why("%s: data file -> %d readers" % (path, len(r)))
    base = os.path.basename(path)
    if base in PYTEST_CFG or path in M.collect_files:
        # read while collecting / configuring: effect unknown
        plan.overlay = True
        if base in PYTEST_CFG and (path.count("/") == 0 or path.startswith("tests/")):
            old, new = plan.data_files.get(path, (None, None))
            if base in ("setup.cfg", "pyproject.toml", "tox.ini") and old is not None and new is not None \
                    and _cfg_parses(base, new) and _pytest_cfg(base, old) == _pytest_cfg(base, new):
                plan.why("%s: configuration without any pytest section" % path)
            else:
                plan.all = True
                plan.why("%s: pytest configuration" % path)


def _pytest_cfg(base, text):
    """The part of a configuration file pytest reads, in comparable form."""
    try:
        if base.endswith(".toml"):
            import tomllib
            d = tomllib.loads(text)
            tool = d.get("tool", {}) if isinstance(d.get("tool", {}), dict) else {}
            return repr(sorted((k, repr(v)) for k, v in tool.items() if "pytest" in k))
        import configparser
        cp = configparser.RawConfigParser(strict=False)
        cp.read_string(text)
        return repr(sorted((sec, sorted(cp.items(sec))) for sec in cp.sections() if "pytest" in sec.lower()))
    except Exception:
        return "unparsable:" + text


def _cfg_parses(base, text):
    """Does the configuration file still parse the way pytest will read it?"""
    try:
        if base.endswith(".toml"):
            import tomllib
            tomllib.loads(text)
            return True
        import re
        for line in text.split("\n"):
            if not line.strip() or line.lstrip().startswith(("#", ";")):
                continue
            if line[:1] in (" ", "\t"):
                continue                      # continuation of the previous value
            if re.match(r"^\[[^\]]+\]\s*$", line) or re.match(r"^[^=:\[\]]+[=:]", line):
                continue
            return False
        return True
    except Exception:
        return False


def _added(path, M, plan, content):
    plan.fp_ok = False
    base = os.path.basename(path)
    top = path.split("/")[0]
    if base in PYTEST_CFG and (path.count("/") == 0 or top == "tests"):
        plan.all = True
        plan.why("%s: new pytest configuration" % path)
        return
    is_py = path.endswith((".py", ".pth", ".so", ".pyd"))
    if is_py and top not in ("sqlglot", "tests") and top not in INERT_DIRS:
        plan.all = True  # a new root module or package can shadow any import (duckdb/, pandas/, json.py ...)
        plan.why("%s: new importable code outside the packages" % path)
        return
    if is_py and top in ("sqlglot", "tests"):
        d = os.path.dirname(path)
        stem = base[:-3] if base.endswith(".py") else base
        if exists(os.path.join(d, stem)) or exists(os.path.join(d, stem + ".py")):
            plan.all = True
            plan.why("%s: shadows existing module" % path)
            return
        if base in ("__init__.py", "conftest.py", "__main__.py") or not exists(os.path.join(d, "__init__.py")):
            plan.all = True
            plan.why("%s: new package init / namespace" % path)
            return
    # tests that read every file of this directory (package walkers, glob loaders) will see the new one
    d = os.path.dirname(path)
    sib = [f for f in M.files if os.path.dirname(f) == d and f != path]
    if sib:
        walkers = None
        for f in sib:
            r = M.readers(f)
            walkers = r if walkers is None else (walkers & r)
        if walkers and len(sib) >= 2:
            plan.cands |= walkers
            plan.overlay = True
            plan.why("%s: new file in a directory %d tests read in full" % (path, len(walkers)))
            return
    plan.why("%s: new file, nothing reads its directory" % path)


def _removed(path, M, plan, why):
    plan.fp_ok = False
    if path.endswith(".py"):
        if is_test_module(path):
            ts = M.by_file.get(path, set())
            plan.cands |= ts
            plan.why("%s: test module %s -> %d tests" % (path, why, len(ts)))
        _all_dependents(path, M, plan, why)
        if path.startswith("sqlglot/") and _eager(path, M):
            plan.all = True
            plan.why("%s: eagerly imported module %s" % (path, why))
    else:
        r = M.readers(path)
        plan.cands |= r
        plan.why("%s: data file %s -> %d readers" % (path, why, len(r)))


def _eager(path, M):
    """Is this module imported by `import sqlglot` itself (so breaking it breaks everything)?"""
    return path in M.eager


def _python(path, old, new, M, plan, ranges=None):
    md = pydiff.diff_module(old, new, ranges)
    in_tests = path.startswith("tests/")
    if in_tests or not path.startswith("sqlglot/"):
        plan.fp_ok = False
    known = path in M.lines or path in M.imp or path in M.imports or in_tests
    if md.same:
        r = M.readers(path)
        plan.cands |= r
        plan.rt |= r
        plan.why("%s: AST-identical -> %d source readers" % (path, len(r)))
        if old.split("\n", 2)[:2] != new.split("\n", 2)[:2] or old[:1] == "\ufeff" or new[:1] == "\ufeff":
            # an encoding declaration lives in the first two lines; the interpreter decodes bytes with it
            plan.overlay = True
            plan.import_probe.add(path)
            plan.broad |= M.dependents(path, transitive=True)
            if path in M.eager or in_tests:
                plan.broad_all = True
            plan.why("%s: first lines changed (encoding declaration?) -> import probe" % path)
        if r:
            plan.overlay = True
        elif path not in plan.import_probe and not in_tests:
            plan.files.pop(path, None)   # behaves exactly like the pinned file: no need to materialise it
            plan.ranges.pop(path, None)
        return
    if md.syntax_error:
        plan.fp_ok = False
        if is_test_module(path):
            ts = M.by_file.get(path, set())
            plan.cands |= ts
        _all_dependents(path, M, plan, "syntax error")
        if not in_tests and _eager(path, M):
            plan.all = True
            plan.why("%s: eagerly imported module broken" % path)
        return
    if not path.startswith(("sqlglot/", "tests/")):
        r = M.readers(path) | M.users_module(path)
        plan.fp_ok = False
        plan.cands |= r
        plan.must |= r
        plan.overlay = True
        plan.why("%s: python outside packages -> %d readers" % (path, len(r)))
        return
    r = M.readers(path)
    if r:
        plan.cands |= r
        plan.rt |= r
        plan.overlay = True
        plan.why("%s: %d tests read this source file" % (path, len(r)))
    for fc in md.funcs:
        lines = set(range(fc.start, fc.end + 1)) if (fc.whole or fc.header or fc.doc_only) else set(fc.lines)
        c = M.cover(path, lines)
        if USE_VP and fc.sites and not (fc.other or fc.whole or fc.header or fc.doc_only):
            inf = set()
            for site in fc.sites:
                r = infected(M, path, site)
                if r is None:
                    inf = None
                    break
                inf |= r
            if inf is not None:
                plan.why("%s:%s operator swap: %d of %d covering tests can see a different value" % (path, fc.qual, len(inf & c), len(c)))
                inf |= set(M.un(M.vp_bad)) & c   # tests that could not be profiled
                c = inf & c if VP_INTERSECT else inf
                plan.sites.setdefault(path, []).extend(fc.sites)
            else:
                plan.sites_only = False
        else:
            plan.sites_only = False
        plan.rt |= c
        imp_mods = set()
        d = M.imp.get(path, {})
        span = range(fc.start, fc.end + 1)
        for l in (span if (fc.header or fc.whole) else lines):
            if l in d:
                imp_mods.update(d[l])
        if fc.header:
            # def-time objects (defaults/decorators) changed
            imp_mods.add(path)
            plan.header_funcs.add("%s:%s" % (M.mod(path), fc.qual))
        if imp_mods:
            plan.overlay = True
            for m in imp_mods:
                c |= M.dependents(m, transitive=False)
                plan.import_probe.add(m)
                plan.broad |= M.dependents(m, transitive=True)
                if m in M.eager or m.startswith("tests/"):
                    plan.broad_all = True
            plan.why("%s:%s import-executed via %s" % (path, fc.qual, sorted(imp_mods)[:4]))
            plan.lines_ok = False
        else:
            plan.hot.append((path, fc))
            plan.lines.setdefault(path, set()).update(lines)
        plan.cands |= c
        plan.why("%s:%s lines %s -> %d" % (path, fc.qual, sorted(lines)[:6], len(c)))
    for ic in md.imports:
        plan.overlay = True
        plan.lines_ok = False
        plan.attr_change = True
        if plan.attr_classes is not None:
            if ic.scope and ic.names and ic.what.startswith("changed") and not in_tests:
                plan.attr_classes.add("%s:%s" % (M.mod(path), ic.scope))
            else:
                plan.attr_classes = None
        c = _import_change(path, ic, M, plan)
        plan.cands |= c


def _import_change(path, ic, M, plan):
    in_tests = path.startswith("tests/")
    if in_tests:
        # module/class level change in test code: everything defined in (or importing) the file
        c = set(M.by_file.get(path, set()))
        if ic.scope and ic.names:
            pass
        for g in M.importers(path, True):
            c |= M.by_file.get(g, set())
            c |= M.users_module(g)
        c |= M.users_module(path)
        plan.why("%s: test-side import-level change (%s) -> %d" % (path, ic.what, len(c)))
        return c
    broad = M.dependents(path, transitive=False)
    plan.import_probe.add(path)
    plan.broad |= M.dependents(path, transitive=True)
    if path in M.eager:
        plan.broad_all = True
    if ic.scope:
        fam = M.users_class(path, ic.scope)
    else:
        fam = None
    names = ic.names
    if not names or not ic.what.startswith("changed"):
        plan.fp_ok = False     # a bare statement, or something added/removed: its effect may live anywhere
    if not names:
        c = broad if fam is None else (fam | set())
        if fam is None or ic.what == "class header":
            c = broad
        plan.why("%s[%s]: %s (no names) -> %d" % (path, ic.scope, ic.what, len(c)))
        return c
    # attribute-exact view for the import-state fingerprint: who reads these names at run time
    # (class attributes only: a module-level name can be re-bound under another name by an import)
    if names and ic.scope and ic.what.startswith("changed") and not in_tests and plan.attr_keys is not None:
        art = set()
        nsite = 0
        escaped = False
        for nm in names:
            for (f, l, col) in M.name_sites(nm):
                if f == path and ic.lines and ic.lines[0] <= l <= ic.lines[1]:
                    continue
                nsite += 1
                if M.is_deferred(f, l, col):
                    art |= M.cover(f, (l,))   # function code: whoever runs it at run time may read the value
                elif f.startswith("tests/"):
                    escaped = True            # read while a test module is imported: its copy is not fingerprinted
            plan.attr_keys.add("%s:%s%s" % (M.mod(path), (ic.scope + ".") if ic.scope else "", nm))
            if plan.ck is not None:
                dv = getattr(ic, "dict_vals", None)
                if dv is not None and len(names) == 1:
                    plan.ck["%s:%s.%s" % (M.mod(path), ic.scope, nm)] = (set(dv[0]), dv[1])
                else:
                    plan.ck = None
        if nsite == 0 or escaped:
            plan.attr_keys = None       # reached only dynamically, or copied out of our sight: cannot narrow
            plan.ck = None
        else:
            if fam is not None:
                art &= fam
            plan.attr_rt |= art
    else:
        plan.attr_keys = None
        plan.ck = None
    restricted = set()
    unrestricted = False
    seen_sites = 0
    removed = ic.what.startswith(("removed", "added", "renamed"))
    if removed:
        # a name that appears or disappears can break whoever imports it by name
        for g in M.importers(path, False):
            plan.import_probe.add(g)
            plan.broad |= M.dependents(g, transitive=True)
            if g in M.eager:
                plan.broad_all = True
    for nm in names:
        sites = M.name_sites(nm)
        own = 0
        for (f, l, col) in sites:
            if f == path and ic.lines and ic.lines[0] <= l <= ic.lines[1]:
                own += 1
                continue
            seen_sites += 1
            if M.import_time_line(f, l, col):
                unrestricted = True
                break
            restricted |= M.cover(f, (l,))
        if unrestricted:
            break
    if unrestricted:
        c = fam if fam is not None else broad
        plan.why("%s[%s]: %s %s import-time readers -> %d" % (path, ic.scope, ic.what, sorted(names), len(c)))
        return c
    if not seen_sites:
        # nobody names it statically: it is reached dynamically (getattr dispatch, dunder protocols)
        c = fam if fam is not None else broad
        plan.why("%s[%s]: %s %s no static readers -> %d" % (path, ic.scope, ic.what, sorted(names), len(c)))
    else:
        c = restricted & fam if fam is not None else restricted
        plan.why("%s[%s]: %s %s runtime readers %d -> %d" % (path, ic.scope, ic.what, sorted(names), len(restricted), len(c)))
    if ic.lines:
        c = c | M.cover_span(path, ic.lines[0], ic.lines[1])   # whoever ran the old definition
    return c
