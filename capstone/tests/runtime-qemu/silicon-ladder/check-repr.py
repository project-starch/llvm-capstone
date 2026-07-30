#!/usr/bin/env python3
"""Report gp-captable capabilities the silicon cannot represent exactly.

The glue (start-gp-captable-interp.S) carves the cap table and then every global
DOWNWARD from sp.END, 16 bytes at a time:

    t1 = sp.END - count*16      ; split -> cap table  [t1, END)
    loop i: t1 -= max(align_up(size_i,16),16) ; split -> global i [t1, prev_t1)

CVA6's compress_bounds only encodes a base exactly when it is a multiple of the
region's representability granule

    granule(L) = 1 << (max(0, floor(log2 L) - 12) + 3)

so a length below 8192 needs 8-byte alignment (always satisfied -- the carve is
16-aligned), 8192..16383 needs 16, 16384..32767 needs 32, and so on.  A base that
is not granule-aligned is truncated DOWNWARD, silently overlapping its neighbour.
That is the same failure mode as C-13, which was fixed in the monitor's split
geometry but never in the glue's per-global carve.

Only two things can trip it in practice: a single global >= 16384 bytes, and a cap
table of >= 1024 entries (count*16 >= 16384).  Everything smaller is covered by the
16-byte carve alignment.

Exit status 1 if any capability is unrepresentable, so this can gate a build.
"""
import struct
import subprocess
import sys


def granule(length):
    hb = length.bit_length() - 1 if length > 1 else 0
    return 1 << (max(0, hb - 12) + 3)


def find_initdesc(path):
    """Return the descriptor's file offset via readelf, or None if absent."""
    for tool in ("llvm-readelf", "readelf"):
        try:
            out = subprocess.run([tool, "-SW", path], capture_output=True,
                                 text=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            continue
        for line in out.splitlines():
            if ".capstone_gp_initdesc" not in line:
                continue
            f = line.split()
            i = f.index(".capstone_gp_initdesc")
            # [Nr] Name Type Address Off Size ...
            return int(f[i + 3], 16)
        return None
    return None


def check(path):
    off = find_initdesc(path)
    if off is None:
        print("%-28s no .capstone_gp_initdesc (not a gp-captable domain)" %
              path.split("/")[-1])
        return []

    blob = open(path, "rb").read()
    _built, count = struct.unpack_from("<QQ", blob, off)
    recs = [struct.unpack_from("<QQq", blob, off + 32 + 24 * i)
            for i in range(count)]

    bad = []
    tbl = count * 16
    if tbl % granule(tbl):
        bad.append(("cap-table", tbl, granule(tbl), tbl, tbl % granule(tbl)))

    cur = tbl                       # base of the current capability = END - cur
    for i, (size, _align, _blob_off) in enumerate(recs):
        stor = max((size + 15) & ~15, 16)
        cur += stor
        g = granule(stor)
        if cur % g:
            bad.append(("global[%d]" % i, stor, g, cur, cur % g))

    status = "OK" if not bad else "UNREPRESENTABLE x%d" % len(bad)
    print("%-28s count=%-5d carve=%-8d %s" %
          (path.split("/")[-1], count, cur, status))
    for name, length, g, base_off, mis in bad:
        print("    %-12s len=%-8d granule=%-4d base=END-%-8d misaligned by %d"
              % (name, length, g, base_off, mis))
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
