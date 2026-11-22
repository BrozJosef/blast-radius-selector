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
RUN_MAX_TEST_MS = int(os.environ.get("BR_RUN_MAX_TEST_MS", "20000"))    # never execute a test slower than this
RUN_MAX_TOTAL_MS = int(os.environ.get("BR_RUN_MAX_TOTAL_MS", "30000"))
CPU_SCALE = float(os.environ.get("BR_CPU_SCALE", "1.6"))   # offline CPU map -> expected CPU on the scorer
P_FAIL = float(os.environ.get("BR_P_FAIL", "0.3"))  # nor a batch bigger than this
WALL_BUDGET = float(os.environ.get("BR_WALL", "140"))                  # stop executing after this many seconds
WALL_HARD = float(os.environ.get("BR_WALL_HARD", "165"))               # no child may run past this
NOEXEC = bool(os.environ.get("BR_NOEXEC"))
HOT = not os.environ.get("BR_NOHOT")
NOSLICE = bool(os.environ.get("BR_NOSLICE"))
NOMINI = bool(os.environ.get("BR_NOMINI"))
NOEVIDENCE = bool(os.environ.get("BR_CASCADE_ALWAYS"))   # keep the conservative cascade rule everywhere
ASSUME_UNAFFECTED = bool(os.environ.get("BR_ASSUME_UNAFFECTED"))   # simulation only
FP = not os.environ.get("BR_NOFP")
NONARROW = bool(os.environ.get("BR_NONARROW"))   # fingerprint may only prove "identical", never narrow by attribute
FP_MIN_GAIN_MS = float(os.environ.get("BR_FP_MIN_GAIN_MS", "6000"))
SURE_RUN_MS = float(os.environ.get("BR_SURE_RUN_MS", "700"))
P_FAIL2 = float(os.environ.get("BR_P_FAIL2", "0.15"))
B_AFTER_FAIL = float(os.environ.get("BR_B_AFTER_FAIL", "1.5"))
P_FAIL_AFTER = float(os.environ.get("BR_P_FAIL_AFTER", "0.6"))
HEAVY_MS = float(os.environ.get("BR_HEAVY_MS", "1200"))
INFO_BUDGET_MS = float(os.environ.get("BR_INFO_BUDGET_MS", "1500"))
_ZY = []


def zygote():
    if not _ZY:
        preimport(["tests/test_optimizer.py"])   # so the zygote inherits the heavy third-party imports
        _ZY.append(hot.Zygote(SRC, debug=bool(DEBUG)))
    return _ZY[0]


def log(*a):
    print(*a, file=sys.stderr, flush=True)


_pre = set()
LAST_INFO = {}
ORACLE = None    # offline simulation hook: (cid, tests) -> failing subset
SPENT = [0.0]    # simulated CPU ms


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


LAST_FP = [None]


def execute(cid, plan, ids, limits, timeout, allow=None, fp=False, mini=None):
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
            runner.MINI = mini
            if allow:
                from . import slices as _s
                _s.ALLOW.clear()
                _s.ALLOW.update({k: set(v) for k, v in allow.items()})
                _s.install()
            broken = []
            bad_imports = None
            if fp:
                # same import order as when the baseline fingerprint was taken
                try:
                    from . import fingerprint
                    bad_imports = fingerprint.import_all()
                except BaseException:  # noqa
                    bad_imports = None
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
            changed = None
            if fp and not broken:
                # did importing the patched code leave any state different from the pinned tree's?
                try:
                    from . import fingerprint
                    base = fingerprint.load()
                    if base is not None and bad_imports is not None:
                        mods, defs = fingerprint.changed_attrs(base)
                        if sorted(bad_imports) != sorted(base["failed"]):
                            mods.append("<imports>")
                        changed = [mods[:200], defs[:200]]
                except BaseException as e:  # noqa
                    changed = None
            if broken:
                bad, info = set(), {"import_broken": broken}
            else:
                bad, info = runner.run_ids(ids, limits)
            os.write(w, json.dumps({"failing": sorted(bad), "info": info, "broken": broken, "fp": changed}).encode())
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
    LAST_FP[0] = res.get("fp")
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
                d = pickle.load(f)
            d["_blobs"] = d["lines"]
            d["lines"] = select._PerKey(d["_blobs"])
            _SL.append(d)
        except Exception as e:
            log("no slice map", repr(e))
            _SL.append({"tests": {}, "lines": {}, "singles": set()})
    return _SL[0]


FIX = "tests/fixtures/"
_SV = []


_ST = []


def slice_classes(plan, t):
    """Entries of a fixture-driven test that instantiate a class whose attributes the patch re-assigns."""
    if not _ST:
        import pickle
        try:
            with open(os.path.join(os.path.dirname(MAPS), "slicetypes.bin"), "rb") as f:
                _ST.append(select._PerKey(pickle.load(f)))
        except Exception as e:
            log("no slice class usage", repr(e))
            _ST.append(None)
    st = _ST[0]
    if st is None or not plan.attr_classes:
        return None
    rec = st.get(t)
    if rec is None or not rec["sound"]:
        return None
    if rec["resid"] & plan.attr_classes:
        return True, {}
    mine = {}
    for F, d in rec["files"].items():
        m = 0
        for c in plan.attr_classes:
            m |= d.get(c, 0)
        if m:
            mine[F] = m
    return False, mine


def slice_vp(plan, t, M):
    """Entries of a fixture-driven test in which the swapped operator(s) can yield a different value.
    Returns (whole, {fixture: bitmask}) or None when value profiles cannot decide."""
    if not plan.sites_only or not plan.sites:
        return None
    if not _SV:
        import pickle
        try:
            with open(os.path.join(os.path.dirname(MAPS), "slicevp.bin"), "rb") as f:
                _SV.append(select._PerKey(pickle.load(f)))
        except Exception as e:
            log("no slice value profiles", repr(e))
            _SV.append(None)
    sv = _SV[0]
    if sv is None:
        return None
    rec = sv.get(t)
    if rec is None:
        return None
    mine = {}
    for path, sites in plan.sites.items():
        for site in sites:
            sp = select.site_pred(M, path, site)
            if sp is None:
                return None
            sid, which, differs = sp
            for m in rec["resid"][which].get(sid, ()):
                if differs(m):
                    return True, {}
            for F, fr in rec["files"].items():
                for m, mask in fr[which].get(sid, {}).items():
                    if differs(m):
                        mine[F] = mine.get(F, 0) | mask
    return False, mine


def plan_slices(plan, cand_ids, M):
    """Which fixture entries each fixture-driven candidate needs. Returns (allow, est_cpu_ms per test)."""
    if NOSLICE or plan.renames or any(p.startswith("tests/") and p.endswith(".py") for p in plan.files):
        return {}, {}
    if any(not (p.startswith(FIX) and p.endswith(".sql")) for p in plan.data_files):
        return {}, {}   # some other data file changed: nothing tells us which entries read it
    S = slicemap()
    need = {}      # fixture name -> set of entries, or None for everything
    est = {}
    frac = {}
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
    # a patch that only re-assigns class attributes: entries that never build an instance of the
    # class (or a subclass) cannot observe it
    by_class = bool(plan.attr_classes) and plan.attr_change and not plan.lines and not plan.data_files \
        and not plan.hot and plan.fp_ok and all(p.startswith("sqlglot/") for p in plan.files)
    for t in cand_ids:
        rec = S["tests"].get(t)
        if rec is None:
            continue
        mine = {}
        whole = False
        vp = slice_vp(plan, t, M) if (code and plan.lines) else None
        if vp is None and by_class:
            vp = slice_classes(plan, t)
        if vp is not None:
            whole, mine = vp
            if not whole and not mine and M.idx[t] not in plan.data_cands:
                whole = True
        elif plan.lines:
            if not code:
                whole = True
            else:
                for path, ls in plan.lines.items():
                    if rec["resid"].get(path) and (rec["resid"][path] & ls):
                        whole = True
                        break
                    d = S["lines"].get(path, {})
                    cov = M.lines.get(path) or {}
                    ti = M.idx[t]
                    for l in ls:
                        hit = False
                        for (tid, F, mask) in d.get(l, ()):
                            if tid == t:
                                mine[F] = mine.get(F, 0) | mask
                                hit = True
                        if not hit:
                            b = cov.get(l)
                            if b and ti in M.un(b):
                                whole = True   # the test runs this line but no single entry accounts for it
                    if whole:
                        break
                if not whole and not mine and M.idx[t] not in plan.data_cands:
                    whole = True   # coverage says it is a candidate but no entry explains why
        elif not plan.data_files:
            whole = True
        if (vp is None or not by_class) and plan.overlay and not plan.lines and plan.files \
                and not all(p in plan.data_files for p in plan.files):
            whole = True           # some other kind of change (import level): no slicing
        # per-entry cost net of the fixed part every run of the test pays (class setup, scaffolding)
        base = rec["resid_cpu"]
        tot_net = 0.0
        want_net = 0.0
        for F, n in rec["files"].items():
            net = [max(0.0, x - base) for x in rec["cpu"][F]]
            tot_net += sum(net)
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
                want_net += sum(net)
            else:
                if need.get(F, set()) is not None:
                    need[F] = need.get(F, set()) | want
                want_net += sum(net[k] for k in want if k < n)
        share = (want_net / tot_net) if tot_net > 0 else 1.0
        fixed = min(0.5, base / max(1.0, rec["cpu_full"]))       # part of the test that slicing cannot remove
        est[t] = M.cpu.get(t, rec["cpu_full"]) * min(1.0, fixed * 0.3 + (1 - fixed * 0.3) * share)
        frac[t] = (rec, base, tot_net)
    # a fixture restricted here must not be read in full by a candidate we cannot slice
    allow = {}
    for F, want in need.items():
        if want is None:
            continue
        readers = {M.tests[i] for i in M.readers(FIX + F)} & cset
        # a candidate that reads this fixture but was never checked against slicing needs it whole
        if not ASSUME_UNAFFECTED and any(r not in S["tests"] and r not in S.get("unaffected", ()) and FIX + F not in plan.data_files for r in readers):
            continue
        allow[F] = want
    for t in list(est):
        rec, base, tot_net = frac[t]
        if any(F not in allow for F in rec["files"]):
            # some file of this test stays unsliced after all: recompute its share
            w = 0.0
            for F, n in rec["files"].items():
                net = [max(0.0, x - base) for x in rec["cpu"][F]]
                w += sum(net) if F not in allow else sum(net[k] for k in allow[F] if k < n)
            share = (w / tot_net) if tot_net > 0 else 1.0
            fixed = min(0.5, base / max(1.0, rec["cpu_full"]))
            est[t] = M.cpu.get(t, rec["cpu_full"]) * min(1.0, fixed * 0.3 + (1 - fixed * 0.3) * share)
    return allow, est


def decide(cid, patch_text, M, tests, num_of, secs_ms):
    """Returns list of test numbers to run for this commit."""
    plan = select.analyze(patch_text, M)
    CASCADE_PLAN[0] = plan
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
    if ORACLE is not None and plan.import_probe:
        SPENT[0] += 300.0
        if ORACLE(cid, None) == "BROKEN":
            return broad, plan
        if plan.fp_ok and FP:
            # simulation stand-in for the import-state fingerprint (optimistic: equal whenever truth allows)
            rt_ids = [M.tests[i] for i in plan.rt if M.tests[i] in num_of]
            gain = packer.pack(set(incl(cand_ids)), secs_ms)[1] - packer.pack(set(incl(rt_ids)), secs_ms)[1] if cand_ids else 0
            if gain >= FP_MIN_GAIN_MS:
                SPENT[0] += 3500.0
                if not plan.attr_change and ORACLE(cid, ("FP", rt_ids)):
                    cand_ids = rt_ids
                    plan.cands = set(plan.rt)
                    fallback = incl(cand_ids)
                elif plan.attr_change and plan.attr_keys is not None and not NONARROW:
                    art = [M.tests[i] for i in (plan.rt | plan.attr_rt) if M.tests[i] in num_of]
                    if ORACLE(cid, ("FP", art)):
                        cand_ids = art
                        plan.cands = set(plan.rt | plan.attr_rt)
                        fallback = incl(cand_ids)
        plan.import_probe = set()
    if plan.import_probe and plan.fp_ok and FP and ORACLE is None and (not plan.attr_change or plan.attr_keys is not None):
        rt_ids = [M.tests[i] for i in ((plan.rt | plan.attr_rt) if plan.attr_change else plan.rt) if M.tests[i] in num_of]
        gain = packer.pack(set(incl(cand_ids)), secs_ms)[1] - packer.pack(set(incl(rt_ids)), secs_ms)[1] if cand_ids else 0
        if gain >= FP_MIN_GAIN_MS:
            LAST_FP[0] = None
            r = execute(cid, plan, [], {}, 25.0, fp=True)
            if r is None or r == "BROKEN":
                log(cid, "import broken -> broad", len(broad))
                return broad, plan
            fpres = LAST_FP[0]
            same = bool(fpres) and fpres[0] == [] and set(fpres[1]) <= plan.header_funcs
            if (not same and not NONARROW and bool(fpres) and plan.attr_keys is not None and len(fpres[0]) < 200
                    and set(fpres[0]) <= plan.attr_keys and set(fpres[1]) <= plan.header_funcs):
                # only the attributes the patch assigns differ: nothing was derived from them at import,
                # so only tests that read them at run time can see the change
                art = [M.tests[i] for i in (plan.rt | plan.attr_rt) if M.tests[i] in num_of]
                log(cid, "only %s changed at import: %d -> %d candidates" % (sorted(fpres[0])[:3], len(cand_ids), len(art)))
                plan.why("import-time state differs only in the assigned attributes")
                cand_ids = art
                plan.cands = set(plan.rt | plan.attr_rt)
            elif same:
                log(cid, "import-time state unchanged: %d -> %d candidates" % (len(cand_ids), len(rt_ids)))
                plan.why("import-time state identical to the pinned tree")
                cand_ids = rt_ids
                plan.cands = set(plan.rt)
            else:
                log(cid, "import-time state differs in", (fpres[0][:5], fpres[1][:5]) if fpres else "?")
            if cand_ids is not None and (same or plan.cands == set(plan.rt | plan.attr_rt)):
                # the digest is taken with every module loaded, so it cannot see a change in what
                # `import sqlglot` alone loads; the one test that checks this is kept
                plan.must |= {i for i, t in enumerate(M.tests) if t.endswith("::test_lazy_load")}
            plan.import_probe = set()      # already probed
            fallback = incl(cand_ids)
    if not cand_ids:
        if plan.import_probe:
            r = execute(cid, plan, [], {}, 12.0)
            if r is None or r == "BROKEN":
                log(cid, "import broken -> broad", len(broad))
                return broad, plan
        return [], plan
    allow, sl_est = plan_slices(plan, cand_ids, M)
    cpu = M.cpu

    lvl = exec_memo_ok(plan, None) if HOT else False
    mini = mini_opts(plan)

    seen_fail = [False]

    def cost(t):
        c = sl_est.get(t)
        if c is None:
            c = cpu.get(t, secs_ms[num_of[t]])
        if mini is not None and t == MINI_ID:
            c = 350.0            # the miniature run, not the 29 s original
        if lvl and t.endswith("::test_execute_tpcds"):
            # measured share left once the executor stage is memoised; a change already seen to alter
            # behaviour mostly misses the plan-level memo and the mutated plans tend to run longer
            c *= 0.14 if lvl == "A" else (B_AFTER_FAIL if seen_fail[0] else 0.62)
        return c * CPU_SCALE

    # Readers of an edited data file nearly always fail. Unless slicing makes confirming them
    # cheap, include them without paying to run them.
    sure = {t for t in cand_ids if M.idx[t] in plan.data_cands and cost(t) > SURE_RUN_MS}
    order = sorted((t for t in cand_ids if t not in sure), key=lambda t: cost(t) / max(1.0, secs_ms[num_of[t]] + 20.0))

    casc_all = NOEVIDENCE or not plan.lines_ok or plan.overlay or bool(plan.import_probe)

    def blind(ts, nothing_run=False):
        """Cost of including these tests without running them (plus what they could drag down)."""
        ts = list(ts)
        if not ts:
            return 0
        extra_ = cascade(ts, M, num_of) if (casc_all or nothing_run) else []
        return packer.pack(set(incl(ts)) | set(extra_), secs_ms)[1]

    def choose(order, fixed, p_fail, spent_classes):
        full = blind(list(order) + list(fixed), nothing_run=not spent_classes)
        best = (full, 0)
        run_ms = 60.0
        classes = set(spent_classes)
        n = len(order)
        # packing is not free: evaluate a handful of prefix lengths, not every one
        marks = {n}
        k = 1
        while k < n:
            marks.add(k)
            k = k + 1 if k < 4 else int(k * 1.6) + 1
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
            if k not in marks and not (k < n and cost(order[k]) > 4 * max(c, 50.0)):
                continue
            rest = list(order[k:]) + list(fixed)
            br = blind(rest) if rest else 0
            # What we run still fails with probability p_fail and is then included, but a test that ran
            # to completion cannot have broken its pool, so it drags no cascade along.
            ran_fail = packer.pack(set(incl(order[:k])) | set(incl(rest)) | set(cascade(rest, M, num_of) if casc_all else ()), secs_ms)[1]
            est = run_ms + br + p_fail * max(0, ran_fail - br)
            if DEBUG and os.environ.get("BR_TRACE_CHOOSE"):
                log("   choose k=%d run_ms=%.0f br=%d ran_fail=%d est=%.0f full=%d" % (k, run_ms, br, ran_fail, est, full))
            if est <= best[0]:
                best = (est, k)
        return best[1]

    # Stage 1 settles the cheap candidates; expensive ones wait until we know whether anything failed.
    cheap_order = [t for t in order if cost(t) <= HEAVY_MS]
    dear = [t for t in order if cost(t) > HEAVY_MS]
    k = choose(cheap_order, list(sure) + dear, P_FAIL, ()) if cheap_order else 0
    if k == 0 and dear and cheap_order:
        # cheap candidates are worth running for what they tell us about the expensive ones
        acc = 0.0
        for t in cheap_order:
            acc += cost(t)
            if acc > INFO_BUDGET_MS:
                break
            k += 1
    light, heavy = cheap_order[:k], cheap_order[k:] + dear + sorted(sure)
    if not light and dear:
        # nothing cheap to learn from: decide the expensive ones on their own merits
        k = choose(order, sure, P_FAIL, ())
        light, heavy = order[:k], order[k:] + sorted(sure)
    if DEBUG:
        log(cid, "PLAN run", [(t.split("::")[-1], round(cost(t))) for t in light][:6], "incl", [(t.split("::")[-1], round(cost(t)), secs_ms[num_of[t]]) for t in heavy][:6], "allow", {k_: len(v) for k_, v in allow.items()})
    if not light:
        if plan.import_probe:
            r = execute(cid, plan, [], {}, 12.0)
            if r is None or r == "BROKEN":
                return broad, plan
        return sorted(set(incl(cand_ids)) | set(cascade(cand_ids, M, num_of))), plan

    def run(ts):
        if ORACLE is not None:
            SPENT[0] += 60.0 + sum(cost(t) for t in ts) + sum(M.cpu_class.get(c_, 0) * CPU_SCALE for c_ in {t.rsplit("::", 1)[0] for t in ts})
            LAST_INFO.clear()
            return ORACLE(cid, ts)
        total = sum(cost(t) for t in ts)
        # past its expected time plus a margin, including the test is cheaper than waiting for it
        # a test gets about four times its expected CPU on the scorer before it counts as hung
        limits = {t: max(6.0, min(0.004 * cost(t), 0.0015 * cost(t) + 5.0)) for t in ts}
        timeout = min(max(5.0, WALL_HARD - (time.time() - T0)), max(15.0, 0.004 * total + 10.0))
        bad = None
        fx = fixture_only(plan, ts, allow)
        if HOT and (fx is not None or hot.eligible(plan)):
            try:
                preimport(ts)
                files = {} if fx is not None else {f: plan.files[f] for f in {f for f, _ in plan.hot}}
                bad, why = zygote().run(cid, files, ts, limits, timeout, allow=allow, override=fx,
                                        opts={"exec_memo": exec_memo_ok(plan, fx), "mini": mini},
                                        ranges={f: plan.ranges[f] for f in files if f in plan.ranges})
                LAST_INFO.clear()
                LAST_INFO.update(getattr(zygote(), "last_info", None) or {})
                if bad is None:
                    log(cid, "hot fallback:", why)
                    if "timeout" in str(why):
                        return None      # do not burn the same time again in an overlay child
            except Exception as e:
                log(cid, "hot error", repr(e))
                bad = None
        if bad is None:
            bad = execute(cid, plan, ts, limits, timeout, allow, mini=mini)
        return bad

    timeouts = [0]
    bad = run(light)
    timeouts[0] += int(LAST_INFO.get("timeouts", 0) or 0) + (1 if bad is None else 0)
    if bad == "BROKEN":
        log(cid, "import broken -> broad", len(broad))
        return broad, plan
    if bad is None:
        return sorted(set(fallback) | set(cascade(cand_ids, M, num_of))), plan
    broke = dict(LAST_INFO.get("pool_broken") or {})
    # Second look: when everything cheap passed, the rest is far less likely to fail, so paying
    # CPU to clear it (instead of a worker's time) becomes worthwhile for more tests.
    rest = [t for t in heavy if t not in sure]
    seen_fail[0] = bool(bad)
    if rest and not broke and time.time() - T0 < WALL_BUDGET:
        # with failures already seen, an expensive candidate probably fails too: running it rarely pays
        k2 = choose(rest, list(sure) + [t for t in bad if t in num_of], P_FAIL2 if not bad else P_FAIL_AFTER,
                    {t.rsplit("::", 1)[0] for t in light})
        if k2:
            more = rest[:k2]
            bad2 = run(more)
            if bad2 == "BROKEN" or bad2 is None:
                return sorted(set(fallback) | set(cascade(cand_ids, M, num_of))), plan
            bad = set(bad) | set(bad2)
            timeouts[0] += int(LAST_INFO.get("timeouts", 0) or 0)
            broke.update(LAST_INFO.get("pool_broken") or {})
            light = light + more
            heavy = rest[k2:] + sorted(sure)
    plan.executed = light
    plan.failed = bad
    plan.allow = allow
    # Unrun pool users could, in principle, hang their class-wide pool. When the patched code has just
    # run to completion in other tests without a single timeout and the change is confined to function
    # bodies, that risk is far smaller than the cost of always adding every later pool user.
    evidence = bool(light) and not timeouts[0] and plan.lines_ok and not plan.overlay and not plan.import_probe
    extra = [] if (evidence and not NOEVIDENCE) else cascade(heavy, M, num_of)
    for cl, first in broke.items():
        extra += [num_of[t] for t in M.pool_users.get(cl, ()) if t in num_of and t.rsplit("::", 1)[-1] >= first]
        log(cid, "pool broke in", cl, "at", first)
    return sorted(set(incl(heavy)) | {num_of[t] for t in bad if t in num_of} | set(extra)), plan


_STAGE = []
MINI_ID = "tests/test_parser.py::TestParser::test_max_nodes"
_MINI_BASE = []


def mini_opts(plan):
    """Options for the miniature max_nodes test: only when every changed line is known."""
    if NOMINI or not plan.lines_ok or plan.overlay or plan.import_probe or plan.renames or not plan.lines:
        return None
    if any(not p.startswith("sqlglot/") or not p.endswith(".py") for p in plan.files):
        return None
    if not _MINI_BASE:
        try:
            with open(os.path.join(os.path.dirname(MAPS), "mini_base.txt")) as f:
                _MINI_BASE.append(int(f.read().strip() or 0) or None)
        except Exception:
            _MINI_BASE.append(None)
    if _MINI_BASE[0] is None:
        return None
    return {"changed": {p: sorted(ls) for p, ls in plan.lines.items()}, "base": _MINI_BASE[0]}



def exec_memo_ok(plan, fx):
    """Which executor memo level is valid: "A" (whole query) when no changed line runs inside
    sqlglot.executor.execute, "B" (plan execution) when none runs inside PythonExecutor.execute."""
    if fx is not None or os.environ.get("BR_NOEXECMEMO"):
        return False         # fixture text changed: queries differ; let them run
    if not hot.eligible(plan) or not plan.lines_ok:
        return False
    if not _STAGE:
        import pickle
        try:
            with open(os.path.join(os.path.dirname(MAPS), "exec_stage.pkl"), "rb") as f:
                _STAGE.append(pickle.load(f))
        except Exception:
            _STAGE.append(None)
    st = _STAGE[0]
    if st is None:
        return False
    level = "A"
    for path, ls in plan.lines.items():
        if path.startswith("tests/"):
            return False
        if st["B"].get(path) and (st["B"][path] & ls):
            return False
        if path.startswith("sqlglot/executor/"):
            return False
        if (st["A"].get(path) and (st["A"][path] & ls)) or path.startswith(("sqlglot/planner.py", "sqlglot/optimizer/")):
            level = "B"
    return level


def fixture_only(plan, ts, allow):
    """Patched .sql fixtures that only loader-driven tests read can be served from the warm zygote."""
    if not plan.data_files or len(plan.data_files) != len(plan.files) or plan.renames or plan.import_probe:
        return None
    S = slicemap()
    out = {}
    for path, (old, new) in plan.data_files.items():
        if not (path.startswith(FIX) and path.endswith(".sql")) or new is None:
            return None
        out[path[len(FIX):]] = new
    if any(t not in S["tests"] for t in ts):
        return None
    return out


CASCADE_PLAN = [None]


def cascade_possible(plan, M):
    """A class-wide pool breaks only when patched code raises while a task or its reply is pickled or
    unpickled. If every changed line is known and none runs inside that stage, it cannot happen."""
    # A hung or dying task also starves the class-wide pool, whatever line it runs: an unrun pool user
    # can never be cleared by line analysis alone. Kept conservative on purpose (red-team finding).
    return True
    if plan is None or not plan.lines_ok or plan.overlay or plan.import_probe or plan.renames:
        return True
    pk = getattr(M, "pk_lines", None)
    if pk is None:
        return True
    if any(not p.endswith(".py") for p in plan.files):
        return True
    for path, ls in plan.lines.items():
        if path.startswith("tests/") or (pk.get(path) and (pk[path] & ls)):
            return True
    return False


def cascade(not_run, M, num_of):
    """Pool users we did not execute could break their class-wide pool; later pool users would then fail."""
    out = []
    if not cascade_possible(CASCADE_PLAN[0], M):
        return out
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
            zc = _ZY[0].total_cpu() if _ZY else 0.0
            zp = getattr(main, "_zcpu", 0.0); main._zcpu = zc
            log(cid, "CPU self %.3f child %.3f zygote %.3f | ran %d" % (now[0] - prev[0], now[1] - prev[1], zc - zp, len(getattr(plan, "executed", ()) or ()) if plan else -1))
        if plan is not None and getattr(plan, "executed", None) is not None and not plan.all:
            # diagnostic probe: keep only what failed when executed, plus readers of edited data files
            keep = {num_of[t] for t in (getattr(plan, "failed", None) or ()) if t in num_of}
            keep |= {num_of[M.tests[i]] for i in plan.data_cands if M.tests[i] in num_of}
            nums = sorted(set(nums) & keep)
        if plan is not None and plan.must:
            nums = sorted(set(nums) | {num_of[M.tests[i]] for i in plan.must if M.tests[i] in num_of})
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
