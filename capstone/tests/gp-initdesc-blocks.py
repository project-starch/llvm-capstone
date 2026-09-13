#!/usr/bin/env python3
"""I-8: under -capstone-gp-captable an image must have ONE .capstone_gp_initdesc block, and this
refuses one that has more.

WHAT GOES WRONG. The section carries a 32-byte header, (built, count), then 24 bytes per global,
and the linker concatenates one such block per translation unit that has globals. The entry glue
and capstone/tests/runtime-qemu/silicon-ladder/domdata-budget.py both read the FIRST header and
stop. So an image built from several translation units gets a capability table carved for the
globals of whichever object the linker put first, every global past that has no slot, and the
first access to one walks off the end of gp.

WHAT IT LOOKS LIKE IF YOU DO NOT CHECK. Not a link error and not a compiler warning. The image
links, every existing gate passes -- no cjalr, a gp access present, exactly one gp-table header,
__capstone_gct_end defined -- and the domain faults inside the entry glue before any of the
program runs, with "Cap mem access OOB" and a cursor sixty-four bytes short of its own end. It
cost half a day on the nginx port, and it would cost the same again on the next one.

WHY EVERY EXISTING PORT IS FINE. They all amalgamate. SQLite uses the upstream amalgamation and
MicroPython's build says so in its own words, "amalgamating py/ + the port into one translation
unit". That was taken for a convenience. It is a requirement, and this is the first time it is
written down.

    usage: gp-initdesc-blocks.py IMAGE...
    exit 0 one block, 1 more than one, 2 the section is unreadable or absent
"""
import os
import re
import struct
import subprocess
import sys

REFUSED, BROKEN = 1, 2


def readelf():
    """The build's own llvm-readelf when the environment names one, PATH otherwise. Bare
    llvm-readelf is not enough: everything else in the ports reaches into the build directory, so
    a caller with a working toolchain and an unprepared PATH used to get a traceback here rather
    than a verdict, and the caller above it read the previous run's answer."""
    prefix = os.environ.get("CAPSTONE_LLVM_BIN")
    if prefix:
        cand = os.path.join(prefix, "llvm-readelf")
        if os.path.exists(cand):
            return cand
    return "llvm-readelf"


def section(path, name):
    try:
        out = subprocess.run([readelf(), "-SW", path], capture_output=True, text=True).stdout
    except FileNotFoundError:
        print("llvm-readelf not found; set CAPSTONE_LLVM_BIN or put it on PATH", file=sys.stderr)
        sys.exit(BROKEN)
    m = re.search(re.escape(name) + r"\s+\S+\s+(\S+)\s+(\S+)\s+(\S+)", out)
    return None if m is None else tuple(int(m.group(i), 16) for i in (1, 2, 3))


def blocks(path):
    """Every descriptor block in the image, as (offset, count)."""
    sec = section(path, ".capstone_gp_initdesc")
    if sec is None:
        return None
    _addr, off, size = sec
    blob = open(path, "rb").read()[off:off + size]
    out, pos = [], 0
    while pos + 32 <= len(blob):
        _built, count = struct.unpack_from("<QQ", blob, pos)
        span = 32 + 24 * count
        # A count that cannot fit is not a block: stop rather than invent one, so a section this
        # tool does not understand reports as unreadable instead of as a pass.
        if count > (len(blob) // 24) or pos + span > len(blob):
            return None
        out.append((pos, count))
        pos += span
    return out


def main(argv):
    if not argv:
        print(__doc__.strip().splitlines()[-2].strip(), file=sys.stderr)
        return BROKEN
    worst = 0
    for path in argv:
        found = blocks(path)
        if not found:
            # None is an absent or unparsable section; an empty list is a section too short to
            # hold even one header. Neither is a pass, and neither may be an exception either:
            # a gate that crashes is a gate somebody switches off.
            print(f"{path}: no readable .capstone_gp_initdesc; cannot say how many blocks",
                  file=sys.stderr)
            worst = max(worst, BROKEN)
            continue
        if len(found) == 1:
            print(f"{path}: one block, {found[0][1]} globals")
            continue
        total = sum(c for _, c in found)
        print(f"{path}: {len(found)} blocks, {total} globals in all, and the glue will carve for "
              f"{found[0][1]}", file=sys.stderr)
        for off, count in found:
            print(f"    offset {off:>6}  count {count}", file=sys.stderr)
        print("  Build the image as ONE translation unit with globals, the way every other port "
              "in this repository does.", file=sys.stderr)
        worst = max(worst, REFUSED)
    return worst


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
