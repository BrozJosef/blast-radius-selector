"""Materialise a patched source tree as a symlink farm (only changed paths are real files)."""
import os
import shutil


def build(src, root, files, renames=()):
    """files: {relpath: content or None(deleted)}. Returns root."""
    if os.path.exists(root):
        shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root)
    changed_dirs = set()
    for p in files:
        d = os.path.dirname(p)
        while True:
            changed_dirs.add(d)
            if not d:
                break
            d = os.path.dirname(d)
    deleted = {p for p, c in files.items() if c is None}
    for d in sorted(changed_dirs, key=lambda x: x.count("/") if x else -1):
        real = os.path.join(root, d)
        os.makedirs(real, exist_ok=True)
        srcd = os.path.join(src, d)
        if os.path.isdir(srcd):
            for name in os.listdir(srcd):
                rel = os.path.join(d, name) if d else name
                if rel in changed_dirs or rel in files:
                    continue
                os.symlink(os.path.join(srcd, name), os.path.join(real, name))
    for p, c in files.items():
        if c is None:
            continue
        with open(os.path.join(root, p), "w", encoding="utf-8", newline="") as f:
            f.write(c)
    return root
