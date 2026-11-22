"""replay.py <log> [policy ...]: replay the request stream of a TEST_LOG image through cache policies.

The stream is policy-independent: M or U is an acquire of <bytes>, C is a cacheable release, F a release that is
never cached (noreuse, untracked, or above the byte bound; under the logged policies make_room always makes room for
a cacheable mapping of at most MR_FREE_BYTES, so an F of at most that size is a noreuse mapping). E/R/H/P are the
logged policy's own evictions and are dropped. X (Plain only) resizes a live mapping from <old> to <bytes>. Alignment is not modelled: a hit needs equal bytes only.

Policies:
  ctl          Full's rule: 512 entries, 64 MiB, make_room oldest-first on a put
  hwm          ctl + mr_fit_peak before a fresh mapping (the pre-registered H)
  hwm+<MiB>    hwm with <MiB> of slack above the live peak
  frac<pct>    ctl while live + cached + new stays under <pct> % of --mem kB, else the hwm rule
Per policy: hits, fresh maps, munmaps (spills + evictions), peak live, peak live + cached, the excess.
Positive control: --check <policy> replays the logged policy and compares its hits and evictions with the log's own
U and E+R+H counts; a mismatch exits 1."""
import re
import sys

FREE_N, FREE_B = 512, 64 << 20


def stream(path):
    ev = []
    for m in re.finditer(r"^@@HM ([MUCFERHPXY]) (\d+) (\d+) (\d+) (\d+)(?: (\d+))?$", open(path, errors="replace").read(), re.M):
        op, b = m.group(1), int(m.group(2))
        if op in "XY": ev.append((op, b, int(m.group(6)))); continue
        if op in "MU": ev.append(("A", b))
        elif op == "C": ev.append(("R", b, True))
        elif op == "F": ev.append(("R", b, False))
    if not ev:
        print("NO DATA:", path); sys.exit(2)
    return ev


def run(ev, pol, mem_kb):
    free = []          # cached sizes, put order
    live = cached = peak = foot = 0
    hits = fresh = unmaps = 0
    slack = 0; frac = None
    if pol.startswith("hwm+"): slack = int(pol[4:]) << 20
    if pol.startswith("frac"): frac = int(pol[4:]) / 100 * mem_kb * 1024
    hwm = pol.startswith("hwm") or frac is not None
    for e in ev:
        if e[0] in "XY":
            g = e[1] - e[2]   # an in-place growth runs the rule first, as a fresh mapping of g bytes would (plain_reuse.c __plain_remap_fit)
            if hwm and g > 0:
                cap = max(live + g, peak) + slack
                if frac is not None:
                    cap = max(cap, frac)
                while free and live + cached + g > cap:
                    need = live + cached + g - cap
                    pick = 0
                    for i in range(1, len(free)):
                        a, c = free[i], free[pick]
                        if (c < need or a < c) if a >= need else (c < need and a > c): pick = i
                    cached -= free.pop(pick); unmaps += 1
            if e[0] == "X":   # Y: the growth was tried and failed; the rule ran, nothing grew
                live += g; peak = max(peak, live)
        elif e[0] == "A":
            b = e[1]
            for i in range(len(free) - 1, -1, -1):
                if free[i] == b:
                    free[i] = free[-1]; free.pop(); cached -= b; live += b; hits += 1
                    break
            else:
                if hwm:
                    cap = max(live + b, peak) + slack
                    if frac is not None:
                        cap = max(cap, frac)
                    while free and live + cached + b > cap:
                        need = live + cached + b - cap
                        pick = 0
                        for i in range(1, len(free)):
                            a, c = free[i], free[pick]
                            if (c < need or a < c) if a >= need else (c < need and a > c): pick = i
                        cached -= free.pop(pick); unmaps += 1
                live += b; fresh += 1
            peak = max(peak, live)
        else:
            b, ok = e[1], e[2]
            live -= b
            if ok and b <= FREE_B:
                while free and (len(free) >= FREE_N or cached + b > FREE_B):
                    cached -= free.pop(0); unmaps += 1
                free.append(b); cached += b
            else:
                unmaps += 1
        foot = max(foot, live + cached)
    return dict(hits=hits, fresh=fresh, unmaps=unmaps, peak=peak, foot=foot, excess=foot - peak)


def main():
    a = sys.argv[1:]
    mem = 921488
    if "--mem" in a: i = a.index("--mem"); mem = int(a[i + 1]); del a[i:i + 2]
    chk = None
    if "--check" in a: i = a.index("--check"); chk = a[i + 1]; del a[i:i + 2]
    path, pols = a[0], a[1:] or ["ctl", "hwm", "hwm+8", "hwm+16", "hwm+32", "frac50", "frac75"]
    ev = stream(path)
    if chk:
        text = open(path, errors="replace").read()
        u = len(re.findall(r"^@@HM U ", text, re.M))
        ev_ = len(re.findall(r"^@@HM [ERH] ", text, re.M)) + len(re.findall(r"^@@HM F ", text, re.M))
        r = run(ev, chk, mem)
        ok = r["hits"] == u and r["unmaps"] == ev_
        print(f"CHECK {chk}: model hits {r['hits']} unmaps {r['unmaps']}; log U {u} E+R+H+F {ev_}: {'MATCH' if ok else 'MISMATCH'}")
        if not ok: sys.exit(1)
    for p in pols:
        r = run(ev, p, mem)
        print(f"{p:8s} hits {r['hits']:6d} fresh {r['fresh']:6d} unmaps {r['unmaps']:6d} peak {r['peak']/2**20:8.1f} MiB "
              f"foot {r['foot']/2**20:8.1f} MiB excess {r['excess']/2**20:7.1f} MiB")


main()
