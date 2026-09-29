#!/usr/bin/env python3
"""Judge the corpus's cases as the FFmpeg app port's domain runs them (run-sublet-port.sh).

    sublet-port-verdict.py <serial.log> <expect.txt> <arm> <image-dir> <fixture>...

Fixture 40 + 2 * case + fixed is case.c with upstream's defect (fixed = 0) or with the fix
(fixed = 1). Each fixture's section of the log (__FFAPP_BEGIN_FX<n>__ .. __FFAPP_END_FX<n>__; a
fault ends the emulator, so a faulting section runs to the end of the log) must match its
registered outcome:

  COMPLETE <verdict>  the case printed "case=<c> arm=<buggy|fixed>" and "VERDICT <verdict>", the
                      host reports capstone_main = 0 (the case's own check held), and there is
                      no capability fault
  FAULT <lines>       the case printed its "case=<c> arm=buggy" line and NO verdict line; the log
                      shows exactly one capability fault, cause 24 or 25, and its pc, taken into
                      the image's own addresses, is an instruction the image's line table
                      attributes to case.c at one of <lines> -- the lines that dereference the
                      case's stale pointer. Where the fault line reports value_hi, it must be
                      non-zero: a zero would be an integer used as an address, not a capability
                      that lost its authority.

The case's arm is the fixture's, so a section that prints another case or arm is an error. An
image with no case.c line information is an ERROR, never a pass: the attribution would then
rest on nothing. Exit 0 only if every requested fixture matches.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

HALT = re.compile(r"domain halted by capability fault: cause = (\d+), pc = 0x([0-9a-f]+)")
DONE = re.compile(r"ffapp-host: DONE, serviced \d+ request\(s\), capstone_main = (-?\d+)")
BASE = re.compile(r"Domain block \(buddy, NOT a capability region\) vaddr = [0-9a-f]+, paddr = ([0-9a-f]+)")
VALUE_HI = re.compile(r"Cap mem access requires capability:.*value_hi = ([0-9a-f]+)")
IMAGE_ENTRY = 0x10000   # link.ld: the image is linked at 0x10000 and loaded at the block's base


def section(lines, n):
    begin, end = f"__FFAPP_BEGIN_FX{n}__", f"__FFAPP_END_FX{n}__"
    out, cur, seen = [], False, False
    for line in lines:
        t = line.strip()
        if t == begin:
            cur, seen = True, True
            continue
        if cur and t == end:
            break
        if cur:
            out.append(line.rstrip("\r\n"))
    return out if seen else None


def llvm_bin():
    if os.environ.get("CAPSTONE_LLVM_BIN"):
        return Path(os.environ["CAPSTONE_LLVM_BIN"])
    return Path(os.environ["CAPSTONE_LLVM_BUILD_DIR"]) / "bin"


def case_lines(image):
    """Map every instruction address the line table attributes to case.c to its line."""
    out = subprocess.run([str(llvm_bin() / "llvm-objdump"), "-d", "-l", "--no-show-raw-insn", str(image)],
                         capture_output=True, text=True).stdout
    where, table = None, {}
    for line in out.splitlines():
        m = re.match(r"^; (\S+):(\d+)", line)
        if m:
            where = int(m.group(2)) if m.group(1).endswith("/case.c") else None
            continue
        m = re.match(r"^\s+([0-9a-f]+):\s", line)
        if m and where is not None:
            table[int(m.group(1), 16)] = where
        elif re.match(r"^[0-9a-f]+ <", line):
            where = None
    return table


def judge(sec, n, want, image):
    c, fixed = (n - 40) // 2, (n - 40) % 2
    arm_line = f"case={c} arm={'fixed' if fixed else 'buggy'}"
    if not any(l.strip() == arm_line for l in sec):
        return False, f"no '{arm_line}' line: the image did not run this case and arm"
    others = [l for l in sec if re.match(r"^case=\d+ arm=", l.strip()) and l.strip() != arm_line]
    if others:
        return False, f"another case or arm ran: {others[0].strip()}"
    verdict = next((l.strip() for l in sec if l.strip().startswith("VERDICT ")), None)
    halts = [HALT.search(l) for l in sec if HALT.search(l)]
    done = next((int(m.group(1)) for m in map(DONE.search, sec) if m), None)
    kind, detail = want
    if kind == "COMPLETE":
        if halts:
            return False, f"faulted: cause {halts[0].group(1)} pc 0x{halts[0].group(2)}"
        if verdict is None or not verdict.startswith(f"VERDICT {detail}"):
            return False, f"verdict {verdict!r}, not VERDICT {detail}"
        if done != 0:
            return False, f"capstone_main = {done}, not 0: the case's own check failed"
        return True, verdict
    # FAULT <lines>
    if verdict is not None:
        return False, f"the case printed its verdict, so nothing stopped it: {verdict}"
    if len(halts) != 1:
        return False, f"{len(halts)} capability faults, not exactly one"
    cause, pc = int(halts[0].group(1)), int(halts[0].group(2), 16)
    if cause not in (24, 25):
        return False, f"cause {cause}, not a capability fault on an access"
    fault_at = next(i for i, l in enumerate(sec) if HALT.search(l))
    if not any(l.strip() == arm_line for l in sec[:fault_at]):
        return False, "the fault came before the case started"
    bases = [int(m.group(1), 16) for m in map(BASE.search, sec[:fault_at]) if m]
    if not bases:
        return False, "no domain base in the section: cannot place the pc in the image"
    vaddr = pc - bases[-1] + IMAGE_ENTRY
    table = case_lines(image)
    if not table:
        return False, f"ERROR: {image} has no case.c line information; the fault cannot be attributed"
    line = table.get(vaddr)
    allowed = {int(x) for x in detail.split(",")}
    his = [VALUE_HI.search(l) for l in sec if VALUE_HI.search(l)]
    if his and int(his[-1].group(1), 16) == 0:
        return False, f"the faulting value has value_hi = 0: an integer, not a revoked capability"
    if line not in allowed:
        return False, f"fault at image 0x{vaddr:x}, case.c line {line}, not one of {sorted(allowed)}"
    return True, f"cause {cause} at image 0x{vaddr:x}, case.c:{line}" + (
        f", value_hi {his[-1].group(1)}" if his else "")


def main(argv):
    if len(argv) < 6:
        sys.exit(__doc__)
    log, expect_path, arm, image_dir, fixtures = argv[1], argv[2], argv[3], Path(argv[4]), argv[5:]
    lines = open(log, errors="replace").readlines()
    expect = {}
    for raw in open(expect_path):
        f = raw.split("#", 1)[0].split()
        if len(f) == 4 and f[0] == arm:
            expect[int(f[1])] = (f[2], f[3])
    ok = True
    for n in map(int, fixtures):
        if n not in expect:
            print(f"fx{n}: ERROR no prediction registered for arm {arm}")
            ok = False
            continue
        sec = section(lines, n)
        if sec is None:
            print(f"fx{n}: ERROR no section in the log -- the image never started")
            ok = False
            continue
        good, why = judge(sec, n, expect[n], image_dir / f"ffapp_fx{n}.dom")
        ok = ok and good
        print(f"fx{n}: {'AS PREDICTED' if good else 'DIFFERS'}  predicted: {' '.join(expect[n])}  got: {why}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(sys.argv)
