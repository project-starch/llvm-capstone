#!/usr/bin/env python3
"""Compute the dom_data budget for a built domain, offline.

Everything the entry glue carves comes out of ONE region, dom_data, and today the
only way to discover it does not fit is a capability fault on the board -- or, until
2026-07-28, no fault at all and silently uninitialized globals. This reproduces the
monitor's and kernel module's arithmetic from the ELF so the answer is available at
build time.

    usage: domdata-budget.py <domain.dom> [--fail-under-stack BYTES]

The arithmetic mirrors, and must stay in step with:
  module/capstone.c:83-85   dom_tot = pow2pages(code_len + 64 KiB); order-N alloc
  sbi_capstone.c            dom_data = base+code_size+DOMAIN_DATA_SIZE .. base+tot_size
                            code_size is rounded UP to a multiple of 16 first
  gen-gp-captable-glue.py / start-gp-captable-interp.S
                            cap table = N*16 off the top, then per-global
                            align_up(size,16) storage below it, stack below that
  the monitor's blob copy    [globals_off, code_size) copied to the FRONT of dom_data
"""
import sys, struct, subprocess, os, re

PAGE = 4096
# TWO DIFFERENT CONSTANTS, BOTH NAMED DOMAIN_DATA_SIZE. Conflating them is why the
# first version of this script reported a 29 KB dom_data where the board showed ~94 KB.
MODULE_HEADROOM = 4096 * 16         # module/capstone.c:24 -- added before rounding to pow2 pages
MONITOR_SEAL_SIZE = 16 * 96         # sbi_capstone.c:33-34 -- the sealed-slot region


def readelf(path, *args):
    tool = os.path.join(os.environ.get("CAPSTONE_LLVM_BIN", ""), "llvm-readelf")
    return subprocess.run([tool, *args, path], capture_output=True, text=True).stdout


def load_extent(path):
    """loadable_start / loadable_end over PT_LOAD, as libcapstone.c:111-129 computes."""
    lo, hi = None, 0
    for l in readelf(path, "-lW").splitlines():
        m = re.match(r'\s+LOAD\s+(0x[0-9a-f]+)\s+(0x[0-9a-f]+)\s+\S+\s+(0x[0-9a-f]+)\s+(0x[0-9a-f]+)', l)
        if m:
            va, memsz = int(m.group(2), 16), int(m.group(4), 16)
            lo = va if lo is None else min(lo, va)
            hi = max(hi, va + memsz)
    return lo, hi


def sections(path):
    out = {}
    for l in readelf(path, "-SW").splitlines():
        m = re.match(r'\s*\[\s*\d+\]\s+(\S+)\s+\S+\s+([0-9a-f]+)\s+([0-9a-f]+)\s+([0-9a-f]+)', l)
        if m:
            out[m.group(1)] = (int(m.group(2), 16), int(m.group(3), 16), int(m.group(4), 16))
    return out


def descriptor(path):
    """(count, [(size, align, blob_off)]) from .capstone_gp_initdesc, or None."""
    secs = sections(path)
    if ".capstone_gp_initdesc" not in secs:
        return None
    _addr, off, size = secs[".capstone_gp_initdesc"]
    blob = open(path, "rb").read()[off:off + size]
    if len(blob) < 32:
        return None
    _built, count = struct.unpack_from("<QQ", blob, 0)
    recs = []
    for i in range(count):
        base = 32 + 24 * i
        if base + 24 > len(blob):
            break
        recs.append(struct.unpack_from("<QQq", blob, base))
    return count, recs


def main():
    if len(sys.argv) < 2:
        print(__doc__); return 2
    path = sys.argv[1]
    fail_under = 0
    if "--fail-under-stack" in sys.argv:
        fail_under = int(sys.argv[sys.argv.index("--fail-under-stack") + 1])

    lo, hi = load_extent(path)
    code_len = hi - lo
    code_size = (((code_len - 1) >> 4) + 1) << 4      # monitor rounds up to 16

    # Headroom SCALES with the image since 2026-08-12 (module/capstone.c:83):
    #   dom_headroom = max(code_len, DOMAIN_DATA_SIZE)
    # Images below 64 KiB are unchanged; larger ones double. Keep this in step with the
    # module or this predictor silently reports the wrong order and the wrong verdict --
    # which it did, reporting "DOES NOT FIT" for a build the module now gives 4 MiB.
    dom_tot = code_len + max(code_len, MODULE_HEADROOM)   # capstone.c:83
    pages = (dom_tot - 1) // PAGE + 1
    order = 0 if pages == 1 else (pages - 1).bit_length()
    tot_size = (1 << order) * PAGE
    dom_data = tot_size - code_size - MONITOR_SEAL_SIZE

    secs = sections(path)
    gbase = secs.get(".capstone_gp_initdesc", (None,))[0]
    globals_off = (gbase - lo) if gbase is not None else 0x1000
    blob = max(0, code_size - globals_off)

    d = descriptor(path)
    if d is None:
        count, recs = 0, []
    else:
        count, recs = d
    table = count * 16
    storage = sum(max(16, (s + 15) & ~15) for s, _a, _o in recs)
    stack = dom_data - blob - table - storage

    print(f"  image           code_len={code_len:>10}  code_size={code_size:>10}")
    print(f"  allocation      pages={pages:<6} order={order:<3} tot_size={tot_size:>10}")
    print(f"  dom_data        {dom_data:>10}")
    print(f"    - blob        {blob:>10}   (globals_off=0x{globals_off:x} .. code_size)")
    print(f"    - cap table   {table:>10}   ({count} globals)")
    print(f"    - storage     {storage:>10}")
    print(f"    = STACK       {stack:>10}")
    if stack <= 0:
        print("  VERDICT: DOES NOT FIT -- the carve overruns dom_data.")
        return 1
    if stack < fail_under:
        print(f"  VERDICT: stack {stack} is under the required {fail_under}.")
        return 1
    print("  VERDICT: fits.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
