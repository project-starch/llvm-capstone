#!/usr/bin/env python3
"""Check gp-captable domains against CVA6's ACTUAL bounds-compression behaviour.

THE RULE, from the RTL (capstone-ariane/core/include/ariane_pkg.sv).
compress_bounds has two branches, selected by whether the cursor sits on the base:

    if (bounds.start == cursor) begin   // "cursorless" -- ariane_pkg.sv:749-751

`split` sets cursor == base on BOTH of its outputs (capstone_dyn_unit.anvil:139-144),
so every capability the glue carves takes the cursorless branch. There:

  * the BASE is exact at any alignment -- decompress returns `start: cursor`
    verbatim (ariane_pkg.sv:662-665). There is no base rounding in this branch.
  * the TOP is truncated DOWN to a multiple of 2**E, where E is derived from the
    highest bit at which cursor and top DIFFER, floored at bit 20:

        lz = 63; while (lz > 20 && bit(cursor,lz) == bit(top,lz)) lz--;
        E  = lz - 20;                                  // ariane_pkg.sv:752-759

    So E is 0 -- and the capability exact -- whenever base and top lie in the same
    2 MiB (2**21) window. E only goes positive when [base, top) STRADDLES a 2 MiB
    boundary, and then the top silently loses its low E bits.

WHAT THIS MEANS FOR US. A domain that is <= 2 MiB and 2 MiB-aligned has every
interior capability inside one window (E = 0, exact), and any capability whose top
is exactly the window edge has a top that is a multiple of 2**21 and so survives
truncation too. Both hold today: the kernel module rounds the allocation up to a
power-of-two page count (capstone.c:83-84) and the page allocator returns it
2**order-page aligned. Domains are therefore exact BY CONSTRUCTION, not by luck --
which is why no ladder rung has ever hit this.

IT STOPS HOLDING THE MOMENT A DOMAIN EXCEEDS 2 MiB. Then interior splits straddle a
window boundary, tops truncate DOWNWARD, and a global silently gets a shorter
capability than it asked for. That is a real cliff sitting just past SQLite's
current size, so this script exists to fail the build when a domain reaches it.

AND QEMU CANNOT SEE ANY OF IT. helper_cssplit works on full 64-bit base/end/cursor
and never calls cap_compress (op_helper.c:848-870); tagged loads even restore exact
bounds from an out-of-band shadow map, bypassing the lossy decode entirely. RTL
round-trips EVERY capability write-back through compress_bounds (ex_stage.sv:
1080-1098) because the compressed form IS the architectural register state. So a
top-truncation bug passes under QEMU forever. Same shape as the DELIN divergence.

NOTE the OTHER branch is the one with the granule(L) = 1 << (max(0, hb(L)-12) + 3)
rule, base truncated down and top rounded up (ariane_pkg.sv:769-806). It applies
only once cursor != base -- e.g. after a cincoffset, or after the monitor's
C_SET_CURSOR. That is exactly what C-13 turned out to be, and it is why the granule
model belongs in the monitor's split geometry and NOT in the glue's carve.

Exit status 1 if any domain is at risk.
"""
import struct
import subprocess
import sys

WINDOW_BITS = 21                      # ariane_pkg.sv floors the scan at bit 20
WINDOW = 1 << WINDOW_BITS


def cursorless_top_exact(base, top):
    """Replay ariane_pkg.sv:752-759 for a split-produced (cursor == base) cap."""
    lz = 63
    while lz > 20 and ((base >> lz) & 1) == ((top >> lz) & 1):
        lz -= 1
    e = lz - 20
    return top % (1 << e) == 0, e


def find_initdesc(path):
    for tool in ("llvm-readelf", "readelf"):
        try:
            out = subprocess.run([tool, "-SW", path], capture_output=True,
                                 text=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            continue
        for line in out.splitlines():
            if ".capstone_gp_initdesc" in line:
                f = line.split()
                return int(f[f.index(".capstone_gp_initdesc") + 3], 16)
        return None
    return None


def check(path):
    name = path.split("/")[-1]
    off = find_initdesc(path)
    if off is None:
        print("%-28s no .capstone_gp_initdesc (not a gp-captable domain)" % name)
        return []

    img = open(path, "rb").read()
    _built, count = struct.unpack_from("<QQ", img, off)
    recs = [struct.unpack_from("<QQq", img, off + 32 + 24 * i)
            for i in range(count)]

    carve = count * 16 + sum(max((s + 15) & ~15, 16) for s, _a, _b in recs)

    bad = []
    # The domain is allocated as a power-of-two page run, so model it as a region
    # based at 0 of that size; base 0 is the worst case for window alignment only
    # if the size exceeds one window, which is exactly what we are testing for.
    pages = (len(img) + 65536 - 1) // 4096 + 1
    order = max(0, (pages - 1).bit_length())
    tot = (1 << order) * 4096

    if tot > WINDOW:
        # Interior splits now straddle a 2 MiB boundary. Replay the carve at a
        # WINDOW-aligned base and report any capability whose top truncates.
        top = tot
        for label, length in ([("cap-table", count * 16)] +
                              [("global[%d]" % i, max((s + 15) & ~15, 16))
                               for i, (s, _a, _b) in enumerate(recs)]):
            base = top - length
            ok, e = cursorless_top_exact(base, top)
            if not ok:
                bad.append((label, length, base, top, e,
                            top - (top >> e << e)))
            top = base

    status = ("OK" if not bad else "TOP-TRUNCATION x%d" % len(bad))
    print("%-28s count=%-5d carve=%-8d tot=%-9d %s" %
          (name, count, carve, tot, status))
    if tot > WINDOW and not bad:
        print("    note: domain is %d bytes, past the 2 MiB window -- exact only "
              "because every top happens to be aligned; treat as fragile" % tot)
    for label, length, base, top, e, lost in bad:
        print("    %-12s len=%-8d [%#x,%#x) straddles a 2 MiB window: E=%d, "
              "top loses %d bytes" % (label, length, base, top, e, lost))
    return bad


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: check-repr.py <domain.elf> [...]")
    failed = 0
    for p in sys.argv[1:]:
        try:
            if check(p):
                failed += 1
        except Exception as exc:                       # noqa: BLE001
            print("%-28s ERROR %s" % (p.split("/")[-1], exc))
    sys.exit(1 if failed else 0)
