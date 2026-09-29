#!/usr/bin/env python3
"""Judge the corpus's cases as the FFmpeg app port's domain runs them (run-sublet-port.sh).

    sublet-port-verdict.py <result-dir> <expect.txt> <arm> <image-dir> <fixture>...

Fixture 40 + 2 * case + fixed is case.c with upstream's defect (fixed = 0) or with the fix
(fixed = 1). <result-dir> is what common/application/check-safety.py collected for each fixture:
fx<n>.json (capstone-job's exit or signal, and the launcher's fault record), fx<n>.stdout and
fx<n>.qemu (the QEMU log over that run). Each must match its registered outcome:

  COMPLETE <verdict>  the case printed "case=<c> arm=<buggy|fixed>" and "VERDICT <verdict>", the
                      application exited 0 (the case's own check held), and there is no fault
  FAULT <lines>       the case printed its "case=<c> arm=buggy" line and NO verdict line; the
                      application died on SIGSEGV with a fault record, cause 24 or 25, for this
                      very image (its SHA-256), and its pc, taken into the image's own addresses,
                      is an instruction the image's line table attributes to case.c at one of
                      <lines> -- the lines that dereference the case's stale pointer. Where the
                      QEMU log reports value_hi, it must be non-zero: a zero would be an integer
                      used as an address, not a capability that lost its authority.

The case's arm is the fixture's, so output that names another case or arm is an error. An image
with no case.c line information is an ERROR, never a pass: the attribution would then rest on
nothing. Exit 0 only if every requested fixture matches.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# capstone-exec's fault record: the pc, and the image's code range as the launcher mapped it.
FAULT = re.compile(r"domain fault cause=(\d+) pc=0x([0-9a-f]+) .*?code=0x([0-9a-f]+)-0x([0-9a-f]+)")
VALUE_HI = re.compile(r"Cap mem access requires capability:.*value_hi = ([0-9a-f]+)")


def llvm_bin():
    if os.environ.get("CAPSTONE_LLVM_BIN"):
        return Path(os.environ["CAPSTONE_LLVM_BIN"])
    return Path(os.environ["CAPSTONE_LLVM_BUILD_DIR"]) / "bin"


def link_base(image):
    """The lowest PT_LOAD address: where the image's first byte is linked."""
    out = subprocess.run([str(llvm_bin() / "llvm-readelf"), "-lW", str(image)],
                         capture_output=True, text=True).stdout
    loads = [int(l.split()[2], 16) for l in out.splitlines() if l.split()[:1] == ["LOAD"]]
    return min(loads) if loads else None


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


def judge(result, stdout, diagnostics, n, want, image):
    c, fixed = (n - 40) // 2, (n - 40) % 2
    arm_line = f"case={c} arm={'fixed' if fixed else 'buggy'}"
    if not any(l.strip() == arm_line for l in stdout):
        return False, f"no '{arm_line}' line: the image did not run this case and arm"
    others = [l for l in stdout if re.match(r"^case=\d+ arm=", l.strip()) and l.strip() != arm_line]
    if others:
        return False, f"another case or arm ran: {others[0].strip()}"
    verdict = next((l.strip() for l in stdout if l.strip().startswith("VERDICT ")), None)
    fault = result.get("fault")
    kind, detail = want
    if kind == "COMPLETE":
        if fault or result.get("kind") != "exit":
            return False, f"did not exit normally: {result.get('kind')} {result.get('value')} {fault or ''}".rstrip()
        if verdict is None or not verdict.startswith(f"VERDICT {detail}"):
            return False, f"verdict {verdict!r}, not VERDICT {detail}"
        if result.get("value") != 0:
            return False, f"exit status {result.get('value')}, not 0: the case's own check failed"
        return True, verdict
    # FAULT <lines>
    if verdict is not None:
        return False, f"the case printed its verdict, so nothing stopped it: {verdict}"
    if result.get("kind") != "signal" or result.get("value") != 11 or not fault:
        return False, f"no domain fault: {result.get('kind')} {result.get('value')}"
    m = FAULT.search(fault)
    if not m:
        return False, f"unreadable fault record: {fault}"
    with open(image, "rb") as stream:
        digest = hashlib.sha256(stream.read()).hexdigest()
    if result.get("image_sha256") != digest:
        return False, f"the fault record is for another image ({result.get('image_sha256')}), not {image}"
    cause, pc, lo, hi = int(m.group(1)), int(m.group(2), 16), int(m.group(3), 16), int(m.group(4), 16)
    if cause not in (24, 25):
        return False, f"cause {cause}, not a capability fault on an access"
    if not lo <= pc < hi:
        return False, f"pc 0x{pc:x} is outside the image's code 0x{lo:x}-0x{hi:x}"
    base = link_base(image)
    if base is None:
        return False, f"ERROR: {image} has no PT_LOAD; the pc cannot be placed"
    vaddr = pc - lo + base
    table = case_lines(image)
    if not table:
        return False, f"ERROR: {image} has no case.c line information; the fault cannot be attributed"
    line = table.get(vaddr)
    allowed = {int(x) for x in detail.split(",")}
    his = [VALUE_HI.search(l) for l in diagnostics if VALUE_HI.search(l)]
    if his and int(his[-1].group(1), 16) == 0:
        return False, f"the faulting value has value_hi = 0: an integer, not a revoked capability"
    if line not in allowed:
        return False, f"fault at image 0x{vaddr:x}, case.c line {line}, not one of {sorted(allowed)}"
    return True, f"cause {cause} at image 0x{vaddr:x}, case.c:{line}" + (
        f", value_hi {his[-1].group(1)}" if his else "")


def main(argv):
    if len(argv) < 6:
        sys.exit(__doc__)
    results, expect_path, arm, image_dir, fixtures = Path(argv[1]), argv[2], argv[3], Path(argv[4]), argv[5:]
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
        record = results / f"fx{n}.json"
        if not record.is_file():
            print(f"fx{n}: ERROR no result record -- the image never ran")
            ok = False
            continue
        stdout = (results / f"fx{n}.stdout").read_text(errors="replace").splitlines()
        diagnostics = (results / f"fx{n}.qemu").read_text(errors="replace").splitlines()
        good, why = judge(json.loads(record.read_text()), stdout, diagnostics, n, expect[n],
                          image_dir / f"ffapp_fx{n}.dom")
        ok = ok and good
        print(f"fx{n}: {'AS PREDICTED' if good else 'DIFFERS'}  predicted: {' '.join(expect[n])}  got: {why}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(sys.argv)
