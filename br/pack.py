def pack(tests, seconds):
    """tests: iterable of numbers. Returns (workers, cost_ms) minimising max load + 2s per worker."""
    ts = sorted(tests, key=lambda n: -seconds[n])
    if not ts:
        return [], 0
    best = None
    for w in range(1, min(8, len(ts)) + 1):
        loads = [0] * w
        ws = [[] for _ in range(w)]
        for n in ts:
            i = loads.index(min(loads))
            loads[i] += seconds[n]
            ws[i].append(n)
        cost = max(loads) + 2000 * w
        if best is None or cost < best[0]:
            best = (cost, ws)
    return best[1], best[0]
