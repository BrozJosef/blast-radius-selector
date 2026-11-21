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

from . import select, pack as packer, overlay  # noqa: E402

DATA = os.environ.get("BR_DATA", "/data")
SRC = select.SRC
MAPS = os.environ.get("BR_MAPS", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "maps", "maps.bin"))
TMP = os.environ.get("BR_TMP", "/tmp")
DEBUG = os.environ.get("BR_DEBUG")

# policy knobs
RUN_MAX_TEST_MS = int(os.environ.get("BR_RUN_MAX_TEST_MS", "1200"))    # never execute a test slower than this
RUN_MAX_TOTAL_MS = int(os.environ.get("BR_RUN_MAX_TOTAL_MS", "6000"))  # nor a batch bigger than this
WALL_BUDGET = float(os.environ.get("BR_WALL", "150"))                  # stop executing after this many seconds
NOEXEC = bool(os.environ.get("BR_NOEXEC"))


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def execute(cid, plan, ids, limits, timeout):
    """Fork a child that runs `ids` against the patched tree. Returns failing set or None on trouble."""
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
            bad, info = runner.run_ids(ids, limits)
            os.write(w, json.dumps({"failing": sorted(bad), "info": info}).encode())
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
    if res.get("info"):
        log("exec info", cid, res["info"])
    return set(res["failing"])


def decide(cid, patch_text, M, tests, num_of, secs_ms):
    """Returns list of test numbers to run for this commit."""
    plan = select.analyze(patch_text, M)
    all_nums = [t["number"] for t in tests]
    if plan.all:
        log(cid, "ALL", plan.reasons[-1:] )
        return all_nums, plan
    # candidates as runtime ids/numbers (map index -> id -> runtime number)
    cand_ids = [M.tests[i] for i in plan.cands]
    cand_ids = [t for t in cand_ids if t in num_of]
    if not cand_ids:
        return [], plan
    if NOEXEC or plan.no_exec or time.time() - T0 > WALL_BUDGET:
        return [num_of[t] for t in cand_ids], plan
    light = [t for t in cand_ids if secs_ms[num_of[t]] <= RUN_MAX_TEST_MS]
    heavy = [t for t in cand_ids if secs_ms[num_of[t]] > RUN_MAX_TEST_MS]
    total = sum(secs_ms[num_of[t]] for t in light)
    if total > RUN_MAX_TOTAL_MS:
        return [num_of[t] for t in cand_ids], plan
    limits = {t: max(3.0, 20 * secs_ms[num_of[t]] / 1000.0) for t in light}
    timeout = max(10.0, 6 * total / 1000.0 + 5.0)
    bad = execute(cid, plan, light, limits, timeout)
    if bad is None:
        return [num_of[t] for t in cand_ids], plan
    plan.executed = light
    plan.failed = bad
    return [num_of[t] for t in heavy] + [num_of[t] for t in bad if t in num_of], plan


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
        workers, cost = packer.pack(set(nums), secs_ms)
        out.append(json.dumps({"commit": cid, "workers": workers}, separators=(",", ":")))
        if DEBUG and plan is not None:
            log(cid, "cost", cost, "n", len(nums), "| ".join(plan.reasons))
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))
    sys.stdout.flush()
    log("done in %.2fs" % (time.time() - T0))


if __name__ == "__main__":
    main()
