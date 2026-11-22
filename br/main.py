"""Blast Radius selector entry point."""
import json
import os
import sys
import time

T0 = time.time()
os.environ.setdefault("SKIP_INTEGRATION", "0")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

from . import select, pack as packer, overlay, hot, slices  # noqa: E402

DATA = os.environ.get("BR_DATA", "/data")
SRC = select.SRC
MAPS = os.environ.get("BR_MAPS", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "maps", "maps.bin"))
TMP = os.environ.get("BR_TMP", "/tmp")
DEBUG = os.environ.get("BR_DEBUG")

# policy knobs
RUN_MAX_TEST_MS = int(os.environ.get("BR_RUN_MAX_TEST_MS", "4000"))    # never execute a test slower than this
RUN_MAX_TOTAL_MS = int(os.environ.get("BR_RUN_MAX_TOTAL_MS", "12000"))
CPU_SCALE = float(os.environ.get("BR_CPU_SCALE", "1.0"))   # offline CPU map -> expected CPU on the scorer
P_FAIL = float(os.environ.get("BR_P_FAIL", "0.5"))  # nor a batch bigger than this
WALL_BUDGET = float(os.environ.get("BR_WALL", "150"))                  # stop executing after this many seconds
NOEXEC = bool(os.environ.get("BR_NOEXEC"))
HOT = not os.environ.get("BR_NOHOT")
NOSLICE = bool(os.environ.get("BR_NOSLICE"))
SURE_RUN_MS = float(os.environ.get("BR_SURE_RUN_MS", "700"))
P_FAIL2 = float(os.environ.get("BR_P_FAIL2", "0.15"))
_ZY = []


def zygote():
    if not _ZY:
        _ZY.append(hot.Zygote(SRC, debug=bool(DEBUG)))
    return _ZY[0]


def log(*a):
    print(*a, file=sys.stderr, flush=True)


_pre = set()
LAST_INFO = {}


def preimport(ids):
    """Import third-party modules in the parent once, so every forked child inherits them."""
    if "base" not in _pre:
        _pre.add("base")
        import unittest, unittest.mock, concurrent.futures, concurrent.futures.process, pickle, signal, importlib  # noqa
        import logging, doctest, datetime, decimal, typing, enum, functools, itertools, collections, re, math, textwrap  # noqa
    if "data" not in _pre and any(t.startswith(("tests/test_optimizer.py", "tests/test_executor.py")) for t in ids):
        _pre.add("data")
        try:
            import duckdb
            import numpy  # noqa
            import pandas  # noqa
            import pandas.testing  # noqa
            _c = duckdb.connect

            def _connect(*a, **k):
                cfg = dict(k.pop("config", None) or {})
                cfg.setdefault("threads", 1)
                return _c(*a, config=cfg, **k)
            duckdb.connect = _connect
        except Exception as e:
            log("preimport failed", repr(e))


def execute(cid, plan, ids, limits, timeout, allow=None):
    """Fork a child that runs `ids` against the patched tree. Returns failing set or None on trouble."""
    preimport(ids)
    root = os.path.join(TMP, "ov_" + cid)
    try:
        overlay.build(SRC, root, plan.files, plan.renames)
    except Exception as e:
        log("overlay failed", cid, repr(e))
        return None
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        code = 1
        try:
            os.close(r)
            devnull = os.open(os.devnull, os.O_RDWR)
            os.dup2(devnull, 0)
            os.dup2(devnull, 1)
            if not DEBUG:
                os.dup2(devnull, 2)
            os.chdir(root)
            sys.path.insert(0, root)
            from . import inlinepool, runner
            inlinepool.install()
            runner.MEMO_OK = not plan.renames and not any(p.startswith("tests/") for p in plan.files)
            if allow:
                from . import slices as _s
                _s.ALLOW.clear()
                _s.ALLOW.update({k: set(v) for k, v in allow.items()})
                _s.install()
            broken = []
            for path in sorted(plan.import_probe):
                if not path.endswith(".py"):
                    continue
                name = path[:-3].replace("/", ".")
                if name.endswith(".__init__"):
                    name = name[:-9]
                try:
                    __import__(name)
                except BaseException as e:  # noqa
                    broken.append(path)
            if broken:
                bad, info = set(), {"import_broken": broken}
            else:
                bad, info = runner.run_ids(ids, limits)
            os.write(w, json.dumps({"failing": sorted(bad), "info": info, "broken": broken}).encode())
            code = 0
        except BaseException as e:  # noqa
            try:
                os.write(w, json.dumps({"error": repr(e)}).encode())
            except Exception:
                pass
        finally:
            os._exit(code)
    os.close(w)
    import select as _sel
    buf = b""
    deadline = time.time() + timeout
    res = None
    while True:
        left = deadline - time.time()
        if left <= 0:
            break
        rl, _, _ = _sel.select([r], [], [], left)
        if not rl:
            break
        chunk = os.read(r, 65536)
        if not chunk:
            break
        buf += chunk
    os.close(r)
    try:
        import signal
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass
    try:
        os.waitpid(pid, 0)
    except OSError:
        pass
    try:
        import shutil
        shutil.rmtree(root, ignore_errors=True)
    except Exception:
        pass
    if buf:
        try:
            res = json.loads(buf.decode())
        except Exception:
            res = None
    if not res or "error" in res:
        log("exec trouble", cid, (res or {}).get("error", "no result/timeout"))
        return None
    LAST_INFO.clear()
    LAST_INFO.update(res.get("info") or {})
    if res.get("info"):
        log("exec info", cid, res["info"])
    if res.get("broken"):
        return "BROKEN"
    return set(res["failing"])


_SL = []


def slicemap():
    if not _SL:
        import pickle
        import zlib
        try:
            with open(os.path.join(os.path.dirname(MAPS), "slices.bin"), "rb") as f:
                _SL.append(pickle.loads(zlib.decompress(f.read())))
        except Exception as e:
            log("no slice map", repr(e))
            _SL.append({"tests": {}, "lines": {}, "singles": set()})
    return _SL[0]


FIX = "tests/fixtures/"


def plan_slices(plan, cand_ids, M):
    """Which fixture entries each fixture-driven candidate needs. Returns (allow, est_cpu_ms per test)."""
    if NOSLICE or plan.renames or any(p.startswith("tests/") and p.endswith(".py") for p in plan.files):
        return {}, {}
    S = slicemap()
    need = {}      # fixture name -> set of entries, or None for everything
    est = {}
    cset = set(cand_ids)
    # (1) edited fixture files: only the entries whose text changed
    for path, (old, new) in plan.data_files.items():
        if not path.startswith(FIX) or old is None or new is None:
            continue
        name = path[len(FIX):]
        ch = slices.changed_entries(old, new, name in S["singles"])
        need[name] = ch
    # (2) code changes inside functions: entries whose run touches a changed line
    code = plan.lines_ok and not plan.overlay and not plan.import_probe
    for t in cand_ids:
        rec = S["tests"].get(t)
        if rec is None:
            continue
        mine = {}
        whole = False
        if plan.lines:
            if not code:
                whole = True
            else:
                for path, ls in plan.lines.items():
                    if rec["resid"].get(path) and (rec["resid"][path] & ls):
                        whole = True
                        break
                    d = S["lines"].get(path, {})
                    for l in ls:
                        for (tid, F, mask) in d.get(l, ()):
                            if tid == t:
                                mine[F] = mine.get(F, 0) | mask
                if not whole and not mine and M.idx[t] not in plan.data_cands:
                    whole = True   # coverage says it is a candidate but no entry explains why
        elif not plan.data_files:
            whole = True
        if plan.overlay and not plan.lines and plan.files and not all(p in plan.data_files for p in plan.files):
            whole = True           # some other kind of change (import level): no slicing
        c = rec["resid_cpu"]
        for F, n in rec["files"].items():
            if whole:
                want = None
            else:
                want = {k for k in range(n) if mine.get(F, 0) >> k & 1}
                if F in need and need[F] is not None and FIX + F in plan.data_files:
                    want |= need[F]
                elif FIX + F in plan.data_files:
                    want = None
            if want is None:
                need[F] = None
                c += sum(rec["cpu"][F])
            else:
                if need.get(F, set()) is not None:
                    need[F] = need.get(F, set()) | want
                c += sum(rec["cpu"][F][k] for k in want if k < n)
        full = max(1.0, rec["cpu_full"])
        est[t] = M.cpu.get(t, full) * min(1.0, c / full)
    # a fixture restricted here must not be read in full by a candidate we cannot slice
    allow = {}
    for F, want in need.items():
        if want is None:
            continue
        readers = {M.tests[i] for i in M.readers(FIX + F)} & cset
        if any(r not in S["tests"] and FIX + F not in plan.data_files for r in readers):
            continue
        allow[F] = want
    for t in list(est):
        rec = S["tests"][t]
        if any(F not in allow for F in rec["files"]):
            # some file of this test stays unsliced: fall back to the measured full cost share
            c = rec["resid_cpu"] + sum(sum(rec["cpu"][F]) if F not in allow else sum(rec["cpu"][F][k] for k in allow[F] if k < n)
                                       for F, n in rec["files"].items())
            est[t] = M.cpu.get(t, rec["cpu_full"]) * min(1.0, c / max(1.0, rec["cpu_full"]))
    return allow, est


def decide(cid, patch_text, M, tests, num_of, secs_ms):
    """Returns list of test numbers to run for this commit."""
    plan = select.analyze(patch_text, M)
    all_nums = [t["number"] for t in tests]
    if plan.all:
        log(cid, "ALL", plan.reasons[-1:])
        return all_nums, plan
    cand_ids = [M.tests[i] for i in plan.cands]
    cand_ids = [t for t in cand_ids if t in num_of]
    incl = lambda ts: [num_of[t] for t in ts]  # noqa: E731
    broad = all_nums if plan.broad_all else sorted({num_of[M.tests[i]] for i in plan.broad if M.tests[i] in num_of} | set(incl(cand_ids)))
    fallback = broad if plan.import_probe else incl(cand_ids)
    if NOEXEC or plan.no_exec or time.time() - T0 > WALL_BUDGET:
        return sorted(set(fallback) | set(cascade(cand_ids, M, num_of))), plan
    if not cand_ids:
        if plan.import_probe:
            r = execute(cid, plan, [], {}, 20.0)
            if r is None or r == "BROKEN":
                log(cid, "import broken -> broad", len(broad))
                return broad, plan
        return [], plan
    allow, sl_est = plan_slices(plan, cand_ids, M)
    cpu = M.cpu

    def cost(t):
        c = sl_est.get(t)
        if c is None:
            c = cpu.get(t, secs_ms[num_of[t]])
        return c * CPU_SCALE

    # Readers of an edited data file nearly always fail. Unless slicing makes confirming them
    # cheap, include them without paying to run them.
    sure = {t for t in cand_ids if M.idx[t] in plan.data_cands and cost(t) > SURE_RUN_MS}
    order = sorted((t for t in cand_ids if t not in sure), key=lambda t: cost(t) / max(1.0, secs_ms[num_of[t]] + 20.0))

    def blind(ts):
        """Cost of including these tests without running them (plus what they could drag down)."""
        ts = list(ts)
        if not ts:
            return 0
        return packer.pack(set(incl(ts)) | set(cascade(ts, M, num_of)), secs_ms)[1]

    def choose(order, fixed, p_fail, spent_classes):
        best = (blind(list(order) + list(fixed)), 0)
        run_ms = 60.0
        classes = set(spent_classes)
        for k, t in enumerate(order, 1):
            c = cost(t)
            if c > RUN_MAX_TEST_MS:
                break
            cl = t.rsplit("::", 1)[0]
            if cl not in classes:
                classes.add(cl)
                run_ms += M.cpu_class.get(cl, 0) * CPU_SCALE
            run_ms += c
            if run_ms > RUN_MAX_TOTAL_MS:
                break
            rest = list(order[k:]) + list(fixed)
            est = run_ms + (blind(rest) if rest else p_fail * 2100)
            if est <= best[0]:
                best = (est, k)
        return best[1]

    k = choose(order, sure, P_FAIL, ())
    light, heavy = order[:k], order[k:] + sorted(sure)
    if DEBUG:
        log(cid, "PLAN run", [(t.split("::")[-1], round(cost(t))) for t in light][:6], "incl", [(t.split("::")[-1], round(cost(t)), secs_ms[num_of[t]]) for t in heavy][:6], "allow", {k_: len(v) for k_, v in allow.items()})
    if not light:
        if plan.import_probe:
            r = execute(cid, plan, [], {}, 20.0)
            if r is None or r == "BROKEN":
                return broad, plan
        return sorted(set(incl(cand_ids)) | set(cascade(cand_ids, M, num_of))), plan

    def run(ts):
        total = sum(cost(t) for t in ts)
        limits = {t: max(15.0, 20 * max(cpu.get(t, 0), secs_ms[num_of[t]]) / 1000.0) for t in ts}
        timeout = max(30.0, 8 * total / 1000.0 + 20.0)
        bad = None
        if HOT and hot.eligible(plan):
            try:
                files = {f: plan.files[f] for f in {f for f, _ in plan.hot}}
                bad, why = zygote().run(cid, files, ts, limits, timeout, allow=allow)
                LAST_INFO.clear()
                LAST_INFO.update(getattr(zygote(), "last_info", None) or {})
                if bad is None:
                    log(cid, "hot fallback:", why)
            except Exception as e:
                log(cid, "hot error", repr(e))
                bad = None
        if bad is None:
            bad = execute(cid, plan, ts, limits, timeout, allow)
        return bad

    bad = run(light)
    if bad == "BROKEN":
        log(cid, "import broken -> broad", len(broad))
        return broad, plan
    if bad is None:
        return sorted(set(fallback) | set(cascade(cand_ids, M, num_of))), plan
    broke = dict(LAST_INFO.get("pool_broken") or {})
    # Second look: when everything cheap passed, the rest is far less likely to fail, so paying
    # CPU to clear it (instead of a worker's time) becomes worthwhile for more tests.
    rest = [t for t in heavy if t not in sure]
    if not bad and rest and not broke and time.time() - T0 < WALL_BUDGET:
        k2 = choose(rest, sure, P_FAIL2, {t.rsplit("::", 1)[0] for t in light})
        if k2:
            more = rest[:k2]
            bad2 = run(more)
            if bad2 == "BROKEN" or bad2 is None:
                return sorted(set(fallback) | set(cascade(cand_ids, M, num_of))), plan
            bad = set(bad) | set(bad2)
            broke.update(LAST_INFO.get("pool_broken") or {})
            light = light + more
            heavy = rest[k2:] + sorted(sure)
    plan.executed = light
    plan.failed = bad
    plan.allow = allow
    extra = cascade(heavy, M, num_of)
    for cl, first in broke.items():
        extra += [num_of[t] for t in M.pool_users.get(cl, ()) if t in num_of and t.rsplit("::", 1)[-1] >= first]
        log(cid, "pool broke in", cl, "at", first)
    return sorted(set(incl(heavy)) | {num_of[t] for t in bad if t in num_of} | set(extra)), plan


def cascade(not_run, M, num_of):
    """Pool users we did not execute could break their class-wide pool; later pool users would then fail."""
    out = []
    for cl, users in M.pool_users.items():
        firsts = [t.rsplit("::", 1)[-1] for t in not_run if t in users]
        if firsts:
            first = min(firsts)
            out += [num_of[t] for t in users if t in num_of and t.rsplit("::", 1)[-1] >= first]
    return out


def main():
    tj = json.load(open(os.path.join(DATA, "tests.json")))
    tests = tj["tests"]
    num_of = {t["id"]: t["number"] for t in tests}
    secs_ms = {t["number"]: int(round(t["seconds"] * 1000)) for t in tests}
    id_of = {t["number"]: t["id"] for t in tests}
    M = select.Maps(MAPS)
    cdir = os.path.join(DATA, "commits")
    out = []
    for name in sorted(os.listdir(cdir)):
        if not name.endswith(".patch"):
            continue
        cid = name[:-6]
        try:
            with open(os.path.join(cdir, name), encoding="utf-8", errors="surrogateescape", newline="") as f:
                text = f.read()
            nums, plan = decide(cid, text, M, tests, num_of, secs_ms)
        except Exception as e:  # last line of defence: never crash, never miss
            log("FATAL", cid, repr(e))
            nums, plan = [t["number"] for t in tests], None
        if DEBUG:
            import resource
            a = resource.getrusage(resource.RUSAGE_SELF); b = resource.getrusage(resource.RUSAGE_CHILDREN)
            now = (a.ru_utime + a.ru_stime, b.ru_utime + b.ru_stime)
            prev = getattr(main, "_cpu", (0, 0)); main._cpu = now
            log(cid, "CPU self %.3f child %.3f | ran %d" % (now[0] - prev[0], now[1] - prev[1], len(getattr(plan, "executed", ()) or ()) if plan else -1))
        drop = [x for x in os.environ.get("BR_PROBE_DROP", "").split(",") if x]
        if drop and len(set(nums)) < len(tests):
            nums = [n for n in nums if not any(d in id_of[n] for d in drop)]
        workers, cost = packer.pack(set(nums), secs_ms)
        out.append(json.dumps({"commit": cid, "workers": workers}, separators=(",", ":")))
        if DEBUG and plan is not None:
            log(cid, "cost", cost, "n", len(nums), "| ".join(plan.reasons))
            if getattr(plan, "failed", None):
                log(cid, "FAILED", sorted(plan.failed)[:12], "ALLOW", {k: sorted(v)[:8] for k, v in (getattr(plan, "allow", None) or {}).items()})
    if _ZY:
        try:
            _ZY[0].close()
        except Exception:
            pass
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))
    sys.stdout.flush()
    try:
        log("cgroup", open("/sys/fs/cgroup/cpu.stat").read().split("\n")[0])
    except Exception:
        pass
    log("done in %.2fs" % (time.time() - T0))


if __name__ == "__main__":
    main()
