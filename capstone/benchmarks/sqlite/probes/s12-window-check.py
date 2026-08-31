#!/usr/bin/env python3
"""Does this build still contain the S-12 fault window? A GATE, not a report.

Every reduction, every source-injection knob and every codegen change is capable of removing the
instruction sequence it was built to test, and the resulting board run is then a clean, monotone,
entirely void result: the domain runs, it returns, and the arm looks like a cure. That failure has
already cost this investigation an arm (`CAPSTONE_ENTRY_MARK`, whose injected counter changed the
faulting function's register allocation enough that the window moved out of it entirely and into
sqlite3WhereRightJoinLoop).

So: before spending an 8-minute boot on a variant, check the artifact. The window is

    movc          rD, zero
    stc           rD, <imm>(rX)
    ldc           rD, <imm>(rY)          <- the reload of the caller's pWInfo
    cincoffsetimm rD, rD, 0xb0           <- &pWInfo->sWC, the faulting instruction

contiguous, inside sqlite3WhereCodeOneLoopStart. 0xb0 is the offset of sWC within WhereInfo, which
is why the constant is part of the signature rather than incidental.

Exit status is the point:
    0  window PRESENT      -- the variant tests what it claims to
    1  window ABSENT       -- the variant tests NOTHING; do not spend a boot on it
    2  could not look      -- objdump failed, symbol missing, file unreadable

"no data" exits 2 and says where it looked, rather than printing an empty result that reads like a
finding. Invoke by ABSOLUTE PATH and never through a pipe: a gate that fails to START exits 127 and
reads exactly like one that ran and passed, and a pipe replaces its exit status with the last
stage's.
"""
import re
import subprocess
import sys

OBJDUMP = "/home/alexey/dev/llvm-capstone/llvm/cmake-build-debug/bin/llvm-objdump"
TRIPLE = "capstone64-unknown-elf"
FUNC = "sqlite3WhereCodeOneLoopStart"


def disassemble(path):
    try:
        out = subprocess.run([OBJDUMP, "-d", f"--triple={TRIPLE}", path],
                             capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"GATE ERROR: could not run {OBJDUMP}: {exc}", file=sys.stderr)
        raise SystemExit(2)
    if out.returncode != 0 or not out.stdout.strip():
        print(f"GATE ERROR: objdump produced nothing for {path}", file=sys.stderr)
        raise SystemExit(2)
    return out.stdout.splitlines()


def body(lines, func):
    """Instructions of `func` as (addr, text), or None if the symbol is absent."""
    start = end = None
    for i, line in enumerate(lines):
        m = re.match(r"^([0-9a-f]+)\s+<(.+)>:", line)
        if m:
            if m.group(2) == func:
                start = i
            elif start is not None and end is None:
                end = i
    if start is None:
        return None
    out = []
    for line in lines[start + 1:end]:
        m = re.match(r"\s+([0-9a-f]+):\s+(?:[0-9a-f]{2} ){4}\s*(.*)", line)
        if m:
            out.append((m.group(1), m.group(2).strip()))
    return out


def find_window(insns):
    """Every contiguous movc/stc/ldc/cincoffsetimm+0xb0 site, with its registers."""
    hits = []
    for i in range(len(insns) - 3):
        texts = [insns[i + k][1] for k in range(4)]
        m0 = re.match(r"movc\s+(\w+),\s*zero$", texts[0])
        m1 = re.match(r"stc\s+(\w+),\s*(-?0x[0-9a-f]+)\((\w+)\)$", texts[1])
        m2 = re.match(r"ldc\s+(\w+),\s*(-?0x[0-9a-f]+)\((\w+)\)$", texts[2])
        m3 = re.match(r"cincoffsetimm\s+(\w+),\s*(\w+),\s*0xb0$", texts[3])
        if not (m0 and m1 and m2 and m3):
            continue
        # The consumer must READ what the ldc just wrote. Its DESTINATION is irrelevant, and
        # requiring dst == src was a real defect in the first version of this gate: it encoded
        # the baseline's exact allocation, so the very first reduction -- which produced
        # `ldc a0,0(a0)` followed by `cincoffsetimm a1, a0, 0xb0` -- was reported WINDOW ABSENT
        # while the window was plainly there. A gate keyed to one register shape silently misses
        # another, which is the failure class this file exists to prevent, committed inside the
        # prevention itself.
        if m3.group(2) != m2.group(1):
            continue
        hits.append({
            "addr": insns[i][0],
            "movc_rd": m0.group(1),
            "stc_src": m1.group(1),
            "ldc_rd": m2.group(1),
            "ldc_base": m2.group(3),
            "paired": m0.group(1) == m2.group(1),
            "text": [f"{insns[i + k][0]}  {texts[k]}" for k in range(4)],
        })
    return hits


def main(argv):
    if len(argv) != 2:
        print(f"usage: {argv[0]} <domain.dom>", file=sys.stderr)
        return 2
    path = argv[1]
    insns = body(disassemble(path), FUNC)
    if insns is None:
        print(f"GATE ERROR: symbol {FUNC} not present in {path}", file=sys.stderr)
        return 2
    hits = find_window(insns)
    print(f"{path}: {FUNC} = {len(insns)} instructions, {len(hits)} S-12 window site(s)")
    for h in hits:
        for line in h["text"]:
            print(f"    {line}")
        print(f"    -> movc {h['movc_rd']} / stc from {h['stc_src']} / ldc into {h['ldc_rd']}"
              f" via {h['ldc_base']};  register match: "
              f"{'PRESERVED' if h['paired'] else 'BROKEN'}")
    if not hits:
        print("WINDOW ABSENT -- this build tests NOTHING about S-12. Do not spend a boot on it.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
