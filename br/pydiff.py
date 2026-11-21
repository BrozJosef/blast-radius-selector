"""Structural diff of two versions of a Python module.

Classifies every difference as either
  * a change inside deferred code (a function / lambda body) -> FuncChange with the old lines involved
  * a change evaluated at import time (module or class body statement, def header) -> ImportChange
"""
import ast
from difflib import SequenceMatcher

FUNC = (ast.FunctionDef, ast.AsyncFunctionDef)


class FuncChange:
    __slots__ = ("qual", "start", "end", "lines", "whole", "header", "kind", "doc_only")

    def __init__(self, qual, start, end, kind="def"):
        self.qual = qual
        self.start = start
        self.end = end
        self.lines = set()
        self.whole = False
        self.header = False
        self.kind = kind
        self.doc_only = False

    def __repr__(self):
        return "Func(%s %d-%d lines=%s whole=%s header=%s %s)" % (
            self.qual, self.start, self.end, sorted(self.lines)[:8], self.whole, self.header, self.kind)


class ImportChange:
    __slots__ = ("scope", "names", "lines", "what")

    def __init__(self, scope, names, lines, what):
        self.scope = scope  # '' for module, 'A.B' for class body
        self.names = names  # set of assigned names, or None when unknown
        self.lines = lines  # (start, end) in old file, or None for pure insertions
        self.what = what

    def __repr__(self):
        return "Import(%s names=%s lines=%s %s)" % (self.scope, self.names, self.lines, self.what)


class ModDiff:
    def __init__(self):
        self.syntax_error = False
        self.old_syntax_error = False
        self.same = False
        self.funcs = []
        self.imports = []

    def __repr__(self):
        return "ModDiff(same=%s syn=%s funcs=%s imports=%s)" % (self.same, self.syntax_error, self.funcs, self.imports)


def _dump(n):
    return ast.dump(n)


class _NoLambda(ast.NodeTransformer):
    def visit_Lambda(self, node):
        return ast.Constant(value="<lambda>")


def _lambdas(node):
    out = []

    def rec(n):
        if isinstance(n, ast.Lambda):
            out.append(n)
            return  # nested lambdas are part of the outer one
        for c in ast.iter_child_nodes(n):
            rec(c)

    rec(node)
    return out


def _strip_lambda_dump(node):
    import copy
    return ast.dump(_NoLambda().visit(copy.deepcopy(node)))


def _targets(node):
    names = set()
    if isinstance(node, ast.Assign):
        ts = node.targets
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        ts = [node.target]
    elif isinstance(node, FUNC + (ast.ClassDef,)):
        return {node.name}
    elif isinstance(node, (ast.Import, ast.ImportFrom)):
        return {(a.asname or a.name).split(".")[0] for a in node.names}
    else:
        return None
    for t in ts:
        for x in ast.walk(t):
            if isinstance(x, ast.Name):
                names.add(x.id)
            elif isinstance(x, (ast.Attribute, ast.Subscript, ast.Starred, ast.Tuple, ast.List)):
                pass
        if not isinstance(t, (ast.Name, ast.Tuple, ast.List)):
            return None
    return names or None


def _span(node):
    start = node.lineno
    for d in getattr(node, "decorator_list", ()):
        start = min(start, d.lineno)
    return start, node.end_lineno


def _header_dump(f):
    return (_dump(f.args), [_dump(d) for d in f.decorator_list], _dump(f.returns) if f.returns else None,
            [_dump(t) for t in getattr(f, "type_params", [])], type(f).__name__)


def _is_doc(stmt):
    return isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str)


class _Differ:
    def __init__(self):
        self.md = ModDiff()

    # ---------- import-time scopes (module / class bodies) ----------
    def body_import(self, old, new, scope):
        do = [_dump(s) for s in old]
        dn = [_dump(s) for s in new]
        if do == dn:
            return
        sm = SequenceMatcher(None, do, dn, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                continue
            o, n = old[i1:i2], new[j1:j2]
            # pair up by position when same length, else by (type, name) for defs
            pairs, lo, ln = self._pair(o, n)
            for a, b in pairs:
                self.stmt_import(a, b, scope)
            for a in lo:
                self.md.imports.append(ImportChange(scope, _targets(a), _span(a), "removed " + type(a).__name__))
            for b in ln:
                anchor = None
                self.md.imports.append(ImportChange(scope, _targets(b), anchor, "added " + type(b).__name__))

    def _pair(self, o, n):
        if len(o) == len(n):
            ok = all(type(a) is type(b) and getattr(a, "name", None) == getattr(b, "name", None) for a, b in zip(o, n))
            if ok:
                return list(zip(o, n)), [], []
        pairs = []
        used = set()
        lo = []
        for a in o:
            hit = None
            if isinstance(a, FUNC + (ast.ClassDef,)):
                for k, b in enumerate(n):
                    if k not in used and type(a) is type(b) and a.name == b.name:
                        hit = k
                        break
            if hit is None:
                lo.append(a)
            else:
                used.add(hit)
                pairs.append((a, n[hit]))
        ln = [b for k, b in enumerate(n) if k not in used]
        return pairs, lo, ln

    def stmt_import(self, a, b, scope):
        if _dump(a) == _dump(b):
            return
        if isinstance(a, FUNC) and isinstance(b, FUNC) and a.name == b.name:
            self.func(a, b, scope)
            return
        if isinstance(a, ast.ClassDef) and isinstance(b, ast.ClassDef) and a.name == b.name:
            ha = ([_dump(x) for x in a.bases], [_dump(x) for x in a.keywords], [_dump(x) for x in a.decorator_list])
            hb = ([_dump(x) for x in b.bases], [_dump(x) for x in b.keywords], [_dump(x) for x in b.decorator_list])
            sc = (scope + "." if scope else "") + a.name
            if ha != hb:
                self.md.imports.append(ImportChange(sc, None, _span(a), "class header"))
            self.body_import(a.body, b.body, sc)
            return
        if type(a) is type(b) and isinstance(a, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
            # compound statement at import level: recurse when headers are equal
            if self._headers_equal(a, b):
                for fa, fb in self._subbodies(a, b):
                    self.body_import(fa, fb, scope)
                return
        if type(a) is type(b):
            # only deferred lambda bodies differ?
            la, lb = _lambdas(a), _lambdas(b)
            if la and len(la) == len(lb) and _strip_lambda_dump(a) == _strip_lambda_dump(b):
                for x, y in zip(la, lb):
                    if _dump(x) != _dump(y):
                        fc = FuncChange((scope + "." if scope else "") + "<lambda>", x.lineno, x.end_lineno, "lambda")
                        if _dump(x.args) != _dump(y.args):
                            fc.header = True
                        fc.lines.update(range(x.lineno, x.end_lineno + 1))
                        self.md.funcs.append(fc)
                return
        na, nb = _targets(a), _targets(b)
        names = None if na is None or nb is None else (na | nb)
        self.md.imports.append(ImportChange(scope, names, _span(a), "changed " + type(a).__name__))

    def _headers_equal(self, a, b):
        if isinstance(a, ast.If) or isinstance(a, ast.While):
            return _dump(a.test) == _dump(b.test)
        if isinstance(a, ast.For):
            return _dump(a.target) == _dump(b.target) and _dump(a.iter) == _dump(b.iter)
        if isinstance(a, ast.With):
            return [_dump(i) for i in a.items] == [_dump(i) for i in b.items]
        if isinstance(a, ast.Try):
            return len(a.handlers) == len(b.handlers) and all(
                (_dump(x.type) if x.type else None, x.name) == (_dump(y.type) if y.type else None, y.name)
                for x, y in zip(a.handlers, b.handlers))
        return False

    def _subbodies(self, a, b):
        if isinstance(a, ast.Try):
            yield a.body, b.body
            for x, y in zip(a.handlers, b.handlers):
                yield x.body, y.body
            yield a.orelse, b.orelse
            yield a.finalbody, b.finalbody
        else:
            yield a.body, b.body
            if hasattr(a, "orelse"):
                yield a.orelse, b.orelse

    # ---------- functions ----------
    def func(self, a, b, scope):
        qual = (scope + "." if scope else "") + a.name
        s, e = _span(a)
        fc = FuncChange(qual, s, e)
        if _header_dump(a) != _header_dump(b):
            fc.header = True
        ab, bb = a.body, b.body
        # docstring handling
        da = ab[0] if ab and _is_doc(ab[0]) else None
        db = bb[0] if bb and _is_doc(bb[0]) else None
        if (da is None) != (db is None) or (da is not None and _dump(da) != _dump(db)):
            ra = ab[1:] if da is not None else ab
            rb = bb[1:] if db is not None else bb
            if [_dump(x) for x in ra] == [_dump(x) for x in rb] and not fc.header:
                fc.doc_only = True
                fc.whole = True
                self.md.funcs.append(fc)
                return
        self.body_func(ab, bb, fc, a)
        if fc.header or fc.whole or fc.lines:
            self.md.funcs.append(fc)

    def body_func(self, old, new, fc, parent):
        do = [_dump(s) for s in old]
        dn = [_dump(s) for s in new]
        if do == dn:
            return
        sm = SequenceMatcher(None, do, dn, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                continue
            o, n = old[i1:i2], new[j1:j2]
            if len(o) == len(n):
                for a, b in zip(o, n):
                    self.stmt_func(a, b, fc)
            else:
                # insertion / deletion of statements: lines of removed stmts plus neighbours
                for a in o:
                    fc.lines.update(range(a.lineno, a.end_lineno + 1))
                if i1 > 0:
                    p = old[i1 - 1]
                    fc.lines.update(range(p.lineno, p.end_lineno + 1))
                if i2 < len(old):
                    p = old[i2]
                    fc.lines.update(range(p.lineno, p.end_lineno + 1))
                if i1 == 0 or i2 >= len(old):
                    # at a block edge: the header of the enclosing statement
                    fc.lines.update(range(parent.lineno, (old[0].lineno if old else parent.end_lineno) + 1))
                    if not old:
                        fc.whole = True

    def stmt_func(self, a, b, fc):
        if _dump(a) == _dump(b):
            return
        if type(a) is type(b):
            if isinstance(a, (ast.If, ast.While, ast.For, ast.With, ast.Try)) and self._headers_equal(a, b):
                for fa, fb in self._subbodies(a, b):
                    self.body_func(fa, fb, fc, a)
                return
            if isinstance(a, (ast.If, ast.While, ast.For, ast.With)):
                # header differs: header lines; bodies may differ too
                first = a.body[0].lineno if a.body else a.end_lineno
                fc.lines.update(range(a.lineno, max(a.lineno, first - 1) + 1))
                for fa, fb in self._subbodies(a, b):
                    self.body_func(fa, fb, fc, a)
                return
            if isinstance(a, FUNC) and a.name == b.name and _header_dump(a) == _header_dump(b):
                self.body_func(a.body, b.body, fc, a)
                return
        fc.lines.update(range(a.lineno, a.end_lineno + 1))


def diff_module(old_src, new_src):
    d = _Differ()
    md = d.md
    try:
        to = ast.parse(old_src)
    except (SyntaxError, ValueError):
        md.old_syntax_error = True
        to = None
    try:
        tn = ast.parse(new_src)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        md.syntax_error = True
        return md
    if to is None:
        md.imports.append(ImportChange("", None, None, "old unparsable"))
        return md
    if _dump(to) == _dump(tn):
        md.same = True
        return md
    d.body_import(to.body, tn.body, "")
    if not md.funcs and not md.imports:
        md.imports.append(ImportChange("", None, None, "undetected difference"))
    return md
