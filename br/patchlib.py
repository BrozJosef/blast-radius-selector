"""Unified-diff (git format) parsing and strict application."""
import re

HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class PatchError(Exception):
    pass


class FileChange:
    __slots__ = ("old", "new", "hunks", "binary", "new_file", "deleted", "rename", "mode_only", "raw")

    def __init__(self):
        self.old = self.new = None
        self.hunks = []  # (old_start, old_len, new_start, new_len, lines)
        self.binary = False
        self.new_file = self.deleted = self.rename = False
        self.mode_only = False
        self.raw = []

    @property
    def path(self):
        return self.old if self.old is not None else self.new


def _strip(p):
    p = p.strip()
    if p.startswith('"') and p.endswith('"'):
        p = p[1:-1].encode().decode("unicode_escape").encode("latin-1").decode("utf-8", "replace")
    if p == "/dev/null":
        return None
    if p.startswith(("a/", "b/")):
        p = p[2:]
    return p


def parse(text):
    """Return list[FileChange]. Raises PatchError when the patch is not understood."""
    files = []
    cur = None
    lines = text.split("\n")
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if line.startswith("diff --git "):
            cur = FileChange()
            files.append(cur)
            m = re.match(r"diff --git a/(.*) b/(.*)$", line)
            if m:
                # ambiguous with spaces; refined by ---/+++ or rename lines below
                half = (len(line) - len("diff --git ")) // 2
                rest = line[len("diff --git "):]
                a, b = rest[:half], rest[half + 1:]
                if a.startswith("a/") and b.startswith("b/") and a[2:] == b[2:]:
                    cur.old = cur.new = a[2:]
                else:
                    cur.old, cur.new = m.group(1), m.group(2)
            i += 1
            continue
        if cur is None:
            if line.startswith("--- "):
                # plain unified diff without git header
                cur = FileChange()
                files.append(cur)
            else:
                i += 1
                continue
        if line.startswith("new file mode"):
            cur.new_file = True
        elif line.startswith("deleted file mode"):
            cur.deleted = True
        elif line.startswith("rename from "):
            cur.rename = True
            cur.old = line[len("rename from "):]
        elif line.startswith("rename to "):
            cur.rename = True
            cur.new = line[len("rename to "):]
        elif line.startswith("copy from "):
            cur.rename = True
            cur.old = line[len("copy from "):]
        elif line.startswith("copy to "):
            cur.new = line[len("copy to "):]
        elif line.startswith(("old mode", "new mode")):
            cur.mode_only = True
        elif line.startswith("Binary files") or line.startswith("GIT binary patch"):
            cur.binary = True
        elif line.startswith("--- "):
            o = _strip(line[4:].split("\t")[0])
            if o is None:
                cur.new_file = True
                cur.old = None
            else:
                cur.old = o
        elif line.startswith("+++ "):
            o = _strip(line[4:].split("\t")[0])
            if o is None:
                cur.deleted = True
                cur.new = None
            else:
                cur.new = o
        elif line.startswith("@@"):
            m = HUNK.match(line)
            if not m:
                raise PatchError("bad hunk header")
            os_, ol, ns, nl = int(m.group(1)), m.group(2), int(m.group(3)), m.group(4)
            ol = 1 if ol is None else int(ol)
            nl = 1 if nl is None else int(nl)
            body = []
            i += 1
            co = cn = 0
            while i < n and (co < ol or cn < nl):
                l = lines[i]
                if l.startswith("\\"):
                    body.append(("\\", l))
                    i += 1
                    continue
                if l.startswith("+"):
                    cn += 1
                    body.append(("+", l[1:]))
                elif l.startswith("-"):
                    co += 1
                    body.append(("-", l[1:]))
                elif l.startswith(" ") or l == "":
                    co += 1
                    cn += 1
                    body.append((" ", l[1:]))
                else:
                    raise PatchError("bad hunk line")
                i += 1
            # trailing "\ No newline" marker
            while i < n and lines[i].startswith("\\"):
                body.append(("\\", lines[i]))
                i += 1
            cur.hunks.append((os_, ol, ns, nl, body))
            cur.mode_only = False
            continue
        i += 1
    for f in files:
        if f.hunks or f.binary or f.rename or f.new_file or f.deleted:
            f.mode_only = False
    return files


def apply(old_text, fc):
    """Apply FileChange hunks to old_text (str or None). Returns new text (or None if deleted)."""
    if fc.binary:
        raise PatchError("binary")
    if fc.deleted:
        return None
    if old_text is None:
        old_lines = []
        old_nl = True
    else:
        old_lines = old_text.split("\n")
        old_nl = old_text.endswith("\n")
        if old_nl:
            old_lines.pop()
    out = []
    pos = 0  # index in old_lines
    new_nl = old_nl
    for os_, ol, ns, nl, body in fc.hunks:
        start = os_ - 1 if ol > 0 else os_
        if start < pos or start > len(old_lines):
            raise PatchError("hunk out of order/range")
        out.extend(old_lines[pos:start])
        pos = start
        last = None
        for k, (tag, txt) in enumerate(body):
            if tag == " ":
                if pos >= len(old_lines) or old_lines[pos] != txt:
                    raise PatchError("context mismatch")
                out.append(txt)
                pos += 1
            elif tag == "-":
                if pos >= len(old_lines) or old_lines[pos] != txt:
                    raise PatchError("removal mismatch")
                pos += 1
            elif tag == "+":
                out.append(txt)
            elif tag == "\\":
                # "\ No newline at end of file" applies to the previous line
                if last == "+" or last == " ":
                    new_nl = False
                if last == "-":
                    # old had no newline at end; new gets one unless a later marker says otherwise
                    new_nl = True
                continue
            last = tag
    out.extend(old_lines[pos:])
    if not out:
        return ""
    return "\n".join(out) + ("\n" if new_nl else "")


def changed_old_lines(fc):
    """(removed old line numbers, insertion points as (old_line_before, n_added))"""
    removed = []
    inserts = []
    for os_, ol, ns, nl, body in fc.hunks:
        o = os_ if ol > 0 else os_ + 1
        for tag, txt in body:
            if tag == " ":
                o += 1
            elif tag == "-":
                removed.append(o)
                o += 1
            elif tag == "+":
                inserts.append(o - 1)
    return removed, inserts


def changed_ranges(fc):
    """Tight line ranges touched by the hunks: (old ranges, new ranges), inclusive.
    A pure insertion is reported on the old side as the two lines around it."""
    olds, news = [], []
    for os_, ol, ns, nl, body in fc.hunks:
        o = os_ if ol > 0 else os_ + 1
        n = ns if nl > 0 else ns + 1
        for tag, txt in body:
            if tag == " ":
                o += 1
                n += 1
            elif tag == "-":
                olds.append((o, o))
                news.append((n - 1, n))
                o += 1
            elif tag == "+":
                news.append((n, n))
                olds.append((o - 1, o))
                n += 1
    return olds, news
