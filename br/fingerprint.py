"""Fingerprint of import-time state.

A change to code that runs while modules are imported (class bodies, module-level calls, helpers
that build tables) matters to a test only through the objects it leaves behind. This module walks
every loaded sqlglot module and digests the data it holds (class attributes, module globals,
closure contents, defaults) but not function bytecode, which line coverage accounts for. The same
walk on the pinned tree is stored when the image is built; equal digests mean the patched import
produced the same state, so only tests that execute the changed lines at run time can be affected.
"""
import enum
import functools
import hashlib
import os
import pickle
import re
import sys
import types

PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "maps", "import_fp.pkl")
_PRIM = (type(None), bool, int, float, complex, str, bytes)
_SKIP_ATTRS = {"__dict__", "__weakref__", "__module__", "__qualname__", "__firstlineno__", "__static_attributes__",
               "__annotations__", "__annotate__", "__annotations_cache__", "__doc__", "__builtins__", "__spec__",
               "__loader__", "__file__", "__cached__", "__path__", "__name__", "__package__",
               "__abstractmethods__", "_abc_impl", "__parameters__", "__orig_bases__", "__type_params__"}


class Unknown(Exception):
    pass


class Walker:
    def __init__(self, shallow=False):
        self.shallow = shallow   # refer to sqlglot classes by name instead of expanding them
        self.cache = {}
        self.path = set()
        self.unknown = []
        self.defaults = {}   # (module, qualname) -> sorted digests of default-argument values

    def key(self, o, depth=0):
        t = type(o)
        if t in _PRIM:
            return (t.__name__, o)
        if depth > 60:
            return ("deep",)
        if t in (list, tuple):
            ks = tuple(self.key(x, depth + 1) for x in o)
            if o and all(isinstance(x, type) for x in o):
                # tuples of classes are isinstance() arguments, often built from a set whose order
                # depends on memory addresses: order carries no meaning
                ks = tuple(sorted(ks, key=repr))
            return ("l" if t is list else "t", ks)
        if t in (dict, types.MappingProxyType) or (isinstance(o, dict) and t.__module__ in ("collections", "builtins")):
            return ("d", tuple(sorted(((repr(self.key(k, depth + 1)), self.key(v, depth + 1)) for k, v in list(o.items())),
                                      key=lambda kv: kv[0])))
        if t in (set, frozenset):
            return ("s", tuple(sorted((repr(self.key(x, depth + 1)) for x in o))))
        if isinstance(o, enum.Enum):
            return ("e", t.__module__, t.__qualname__, o.name, self.key(o._value_, depth + 1) if type(o._value_) in _PRIM else "v")
        if isinstance(o, types.ModuleType):
            return ("m", o.__name__)
        i = id(o)
        if i in self.path:
            return ("cyc",)
        hit = self.cache.get(i)
        if hit is not None:
            return hit
        self.path.add(i)
        try:
            r = self._obj(o, t, depth)
        finally:
            self.path.discard(i)
        self.cache[i] = r
        return r

    def _obj(self, o, t, depth):
        if isinstance(o, type):
            mod = o.__module__ or ""
            if self.shallow or not mod.startswith("sqlglot"):
                return ("cref", mod, o.__qualname__)
            items = []
            for k, v in sorted(vars(o).items()):
                if k in _SKIP_ATTRS:
                    continue
                items.append((k, self.key(v, depth + 1)))
            return ("C", mod, o.__qualname__, tuple((b.__module__, b.__qualname__) for b in o.__bases__), tuple(items))
        if isinstance(o, (types.FunctionType, types.LambdaType)):
            cells = ()
            if o.__closure__:
                cs = []
                for name, c in zip(o.__code__.co_freevars, o.__closure__):
                    try:
                        cs.append((name, self.key(c.cell_contents, depth + 1)))
                    except ValueError:
                        cs.append((name, ("empty",)))
                cells = tuple(cs)
            # default values are kept apart from the state digest: a changed default is visible only
            # to callers of that function, which coverage of the function already finds
            dk = hashlib.sha1(repr((self.key(o.__defaults__, depth + 1), self.key(o.__kwdefaults__, depth + 1)))
                              .encode("utf-8", "surrogatepass")).hexdigest()[:16]
            self.defaults.setdefault((o.__module__, o.__qualname__), []).append(dk)
            return ("F", o.__module__, o.__qualname__, cells,
                    self.key({k: v for k, v in vars(o).items() if k != "__wrapped__"}, depth + 1) if vars(o) else ())
        if isinstance(o, (types.BuiltinFunctionType, types.MethodDescriptorType, types.WrapperDescriptorType,
                          types.MethodWrapperType, types.GetSetDescriptorType, types.MemberDescriptorType,
                          types.ClassMethodDescriptorType)):
            return ("b", getattr(o, "__qualname__", repr(o)))
        if isinstance(o, types.MethodType):
            return ("M", self.key(o.__func__, depth + 1), self.key(o.__self__, depth + 1))
        if isinstance(o, (classmethod, staticmethod)):
            return ("cm", t.__name__, self.key(o.__func__, depth + 1))
        if isinstance(o, property):
            return ("p", self.key(o.fget, depth + 1), self.key(o.fset, depth + 1), self.key(o.fdel, depth + 1))
        if isinstance(o, functools.partial):
            return ("P", self.key(o.func, depth + 1), self.key(o.args, depth + 1), self.key(o.keywords, depth + 1))
        if isinstance(o, re.Pattern):
            return ("re", o.pattern, o.flags)
        if isinstance(o, (functools.cached_property,)):
            return ("cp", self.key(o.func, depth + 1))
        mod = t.__module__ or ""
        if mod.startswith("sqlglot"):
            d = None
            try:
                d = vars(o)
            except TypeError:
                slots = []
                for c in t.__mro__:
                    for s in getattr(c, "__slots__", ()) or ():
                        if isinstance(s, str) and hasattr(o, s):
                            slots.append((s, getattr(o, s)))
                d = dict(slots)
            return ("O", mod, t.__qualname__, tuple((k, self.key(v, depth + 1)) for k, v in sorted(d.items())
                                                     if k not in ("parent", "_parent", "_hash")))
        if mod in ("functools", "typing", "threading", "_thread", "logging", "weakref", "abc", "builtins", "types",
                   "collections", "itertools", "operator", "_functools", "_collections", "decimal", "datetime"):
            if mod in ("decimal", "datetime"):
                return ("r", mod, repr(o))
            if hasattr(o, "__wrapped__"):
                return ("w", t.__qualname__, self.key(o.__wrapped__, depth + 1))
            return ("x", mod, t.__qualname__)   # stateless or irrelevant runtime machinery
        self.unknown.append("%s.%s" % (mod, t.__qualname__))
        return ("?", mod, t.__qualname__)


def digest_modules(prefix="sqlglot"):
    """{module name: digest} for every loaded module under prefix (None digest = could not be walked)."""
    out = {}
    w = Walker()
    for name, m in sorted(sys.modules.items()):
        if m is None or not (name == prefix or name.startswith(prefix + ".")):
            continue
        items = []
        try:
            for k, v in sorted(vars(m).items()):
                if k in _SKIP_ATTRS:
                    continue
                items.append((k, w.key(v)))
            r = repr(items)
            out[name] = hashlib.sha1(r.encode("utf-8", "surrogatepass")).hexdigest()
        except Exception:
            out[name] = None
    return out, sorted(set(w.unknown)), {k: sorted(v) for k, v in w.defaults.items()}


def digest_attrs(prefix="sqlglot"):
    """Attribute-level digests: {"module:name" or "module:Class.attr": digest}, plus default-argument
    digests per function. Classes are referenced by name, so a change shows up exactly where it lives."""
    out = {}
    w = Walker(shallow=True)

    def h(v):
        return hashlib.sha1(repr(w.key(v)).encode("utf-8", "surrogatepass")).hexdigest()[:16]

    seen_cls = set()

    def do_class(c):
        if id(c) in seen_cls:
            return
        seen_cls.add(id(c))
        name = "%s:%s" % (c.__module__, c.__qualname__)
        out[name + ".<bases>"] = h(tuple((b.__module__, b.__qualname__) for b in c.__bases__))
        for k, v in list(vars(c).items()):
            if k in _SKIP_ATTRS:
                continue
            if isinstance(v, type) and (v.__module__ or "").startswith(prefix):
                do_class(v)
            try:
                out["%s.%s" % (name, k)] = h(v)
            except Exception:
                out["%s.%s" % (name, k)] = None

    for mname, m in sorted(sys.modules.items()):
        if m is None or not (mname == prefix or mname.startswith(prefix + ".")):
            continue
        for k, v in sorted(vars(m).items()):
            if k in _SKIP_ATTRS:
                continue
            if isinstance(v, type) and (v.__module__ or "").startswith(prefix):
                do_class(v)
            try:
                out["%s:%s" % (mname, k)] = h(v)
            except Exception:
                out["%s:%s" % (mname, k)] = None
    return out, {k: sorted(v) for k, v in w.defaults.items()}, sorted(set(w.unknown))


def changed_attrs(base):
    """(attribute keys whose import-time value differs, functions whose defaults differ)."""
    now, defaults, unknown = digest_attrs()
    b = base["attrs"]
    diff = sorted(k for k in set(now) | set(b) if now.get(k) is None or b.get(k) is None or now.get(k) != b.get(k))
    bd = base.get("adefaults", {})
    dd = sorted("%s:%s" % k for k in set(bd) | set(defaults) if bd.get(k) != defaults.get(k))
    return diff, dd


def import_all():
    """Import every sqlglot module the way a full test run eventually would."""
    import importlib
    import pkgutil
    import sqlglot
    bad = []
    for info in pkgutil.walk_packages(sqlglot.__path__, prefix="sqlglot."):
        if info.name == "sqlglot.__main__":
            continue
        try:
            importlib.import_module(info.name)
        except BaseException as e:  # noqa
            bad.append(info.name)
    return bad


def changed_modules(base):
    """(modules whose import-time state differs, functions whose default arguments differ)."""
    now, unknown, defaults = digest_modules()
    baseline = base["digests"]
    diff = []
    for name in set(now) | set(baseline):
        a, b = baseline.get(name), now.get(name)
        if a is None or b is None or a != b:
            diff.append(name)
    bd = base.get("defaults", {})
    dd = sorted("%s:%s" % k for k in set(bd) | set(defaults) if bd.get(k) != defaults.get(k))
    return sorted(diff), dd


def load():
    try:
        with open(PATH, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def build(out=PATH):
    import unittest  # noqa: F401  sqlglot records at import whether it runs under unittest; tests always do
    import unittest.mock  # noqa: F401
    bad = import_all()
    d, unknown, defaults = digest_modules()
    attrs, adefaults, unknown2 = digest_attrs()
    with open(out, "wb") as f:
        pickle.dump({"digests": d, "failed": bad, "defaults": defaults, "attrs": attrs, "adefaults": adefaults}, f, protocol=5)
    print("attribute digests: %d" % len(attrs), file=sys.stderr)
    try:
        from . import maxnodes_mini
        ok, work, detail = maxnodes_mini.replica(count=True)
        with open(os.path.join(os.path.dirname(out), "mini_base.txt"), "w") as f:
            f.write(str(work if ok and work else 0))
        print("miniature max_nodes: %s work=%s" % (detail, work), file=sys.stderr)
    except Exception as e:
        print("miniature max_nodes baseline failed: %r" % (e,), file=sys.stderr)
    print("import fingerprint: %d modules, %d failed imports, unknown types: %s" % (len(d), len(bad), unknown[:12]), file=sys.stderr)


if __name__ == "__main__":
    build()
