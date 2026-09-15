#!/usr/bin/env python3
"""Which variable does the M1 no-reclamation release cost actually track?

The baseline boots measure give_cyc/n rising with allocations. Three candidate
independent variables are confounded in a single boot's log and this script
separates them:

  (a) nodes RETAINED / live         -- the four arms differ by 0 / 16 / 2048
  (b) the GLOBAL rev-node pool head -- monotonic across a boot (R-12: the pool is
                                       65536 and is NOT reclaimed between runs)
  (c) nodes minted in THIS DOMAIN   -- resets at every domain entry

Reads the run-scoped boot.txt captures and reports all three. "No data" is an
ERROR, not a zero: a parse that finds no snapshots exits non-zero, and lines
corrupted by interleaved monitor markers are COUNTED and reported, never
silently dropped (about 1 % of snapshots on the boots of record).

Usage: m1-cost-variable.py <boot.txt> [boot.txt ...]
"""
import re
import sys

NUM = ["alloc", "minted", "live", "n", "give_cyc", "take_cyc", "retained", "revoked", "init_n"]


def invocations(path):
    """Split one capture into per-invocation segments; each domain entry is one."""
    data = open(path, "rb").read().decode("latin1")
    out, bad = [], 0
    for seg in re.split(r"R1 m1 start ", data)[1:]:
        body = seg.split("R1 m1 end")[0]
        arm = re.search(r"arm=(\w+)", seg)
        snaps = []
        for m in re.finditer(r"R1 m1 snap ([^\r\n]*)", body):
            d = dict(kv.split("=", 1) for kv in m.group(1).split() if "=" in kv)
            try:
                r = {k: int(d[k]) for k in NUM}
            except (KeyError, ValueError):
                bad += 1
                continue
            if r["n"] <= 0:
                bad += 1
                continue
            snaps.append(r)
        end = re.search(r"R1 m1 end[^\n]*minted=(\d+)", seg)
        out.append({
            "arm": arm.group(1) if arm else "?",
            "snaps": sorted(snaps, key=lambda x: x["minted"]),
            "minted_total": int(end.group(1)) if end else None,
        })
    return out, bad


def slope(pts):
    n = len(pts)
    if n < 3:
        return None
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    den = sum((p[0] - mx) ** 2 for p in pts)
    return None if den == 0 else sum((p[0] - mx) * (p[1] - my) for p in pts) / den


def main(paths):
    total, bad_total = 0, 0
    for path in paths:
        invs, bad = invocations(path)
        bad_total += bad
        clean = sum(len(i["snaps"]) for i in invs)
        total += clean
        print(f"== {path}")
        print(f"   {len(invs)} invocations, {clean} clean snapshots, {bad} corrupted")
        if not invs:
            continue

        # (b) vs (c): the pool head is cumulative across the boot; per-domain minting is not.
        print(f"   {'#':>3} {'arm':>9} {'first give/n':>13} {'last give/n':>12} {'pool head BEFORE':>17}")
        cum = 0
        for k, inv in enumerate(invs, 1):
            if not inv["snaps"]:
                print(f"   {k:>3} {inv['arm']:>9} {'NO CLEAN SNAPSHOTS':>27}")
                continue
            fi, la = inv["snaps"][0], inv["snaps"][-1]
            print(f"   {k:>3} {inv['arm']:>9} {fi['give_cyc']/fi['n']:>13.1f} "
                  f"{la['give_cyc']/la['n']:>12.1f} {cum:>17}")
            cum += inv["minted_total"] if inv["minted_total"] is not None else la["minted"]
        print(f"   pool head after the boot: {cum} nodes (R-12: 65536, not reclaimed between runs)")

        # (a): compare arms at MATCHED per-domain minting, where retention differs.
        by_arm = {}
        for inv in invs:
            by_arm.setdefault(inv["arm"], []).extend(inv["snaps"])
        arms = sorted(by_arm)
        print(f"   give_cyc/n at matched per-domain minted (retention differs by arm):")
        print(f"   {'minted':>8} " + " ".join(f"{a:>20}" for a in arms))
        for target in (250, 500, 1000, 1500, 2000, 2079, 2300, 2590):
            cells = []
            for a in arms:
                pool = by_arm[a]
                if not pool:
                    cells.append(f"{'-':>20}")
                    continue
                best = min(pool, key=lambda x: abs(x["minted"] - target))
                if abs(best["minted"] - target) > 60:
                    cells.append(f"{'out of range':>20}")
                else:
                    cells.append(f"{best['give_cyc']/best['n']:>11.1f} r={best['retained']:<6}")
            print(f"   {target:>8} " + " ".join(cells))

        # the curve's shape, per arm, as least-squares windows
        print(f"   window slopes, cycles per 1000 per-domain minted:")
        for a in arms:
            pts = [(x["minted"], x["give_cyc"] / x["n"]) for x in by_arm[a]]
            cells = []
            for lo in range(0, 2600, 520):
                s = slope([p for p in pts if lo <= p[0] < lo + 520])
                cells.append(f"{s*1000:>7.0f}" if s else f"{'--':>7}")
            print(f"     {a:>9}: " + " ".join(cells))
        print()

    if total == 0:
        sys.exit("ERROR: no snapshots parsed from " + ", ".join(paths)
                 + " -- looked for lines matching 'R1 m1 snap <k>=<v> ...'")
    print(f"TOTAL {total} clean snapshots, {bad_total} corrupted "
          f"({100*bad_total/(total+bad_total):.1f} % of the capture)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
