#!/usr/bin/env python3
"""vm-contract's pre-registered branches, read from the board transcript's vm-contract segment (vp2 Tier 1 boot 2).

identity      VIRTUAL_APPLICATION_OK, then the stale access denied with 25 (exit 139): the run is virtual
layout        29 before VIRTUAL_APPLICATION_OK: header or layout, kit side (to the RTL lane with the table dump)
injected      the monitor handed a trap from S or U to the kernel (EXCX/ILLX then INJS); its cause and pc are the reading
monitor-stop  NODM or INJL: the monitor's own stop
delin         cause 26 at map 0 base + 0x1f60: the unfixed image-gp delin in __capstone_cap_init (C-27), kit side (boot 3)
qemu-csr      cause 2 whose tval is a Zicsr instruction on CSR 0xcc0 or 0x5c0: a QEMU-only CSR read not removed (Q-21/Q-22), kit side
other         anything else, printed with its exits and causes
NO DATA       the segment is missing or ambiguous (exit 2)

  python3 vm-contract-branch.py <board.txt>       python3 vm-contract-branch.py --selftest
"""
import re, sys

TAG = r"[0-9A-Fa-f]+"


def branch(t):
    t = t.replace("\r", "")
    starts = list(re.finditer(r"^### RUNG \d+/\d+ vm-contract ###$", t, re.M))
    if len(starts) != 1:
        return "NO DATA", f"{len(starts)} vm-contract start markers"
    rest = t[starts[0].end():]
    end = re.search(r"^### RUNG \d+ vm-contract END rc=-?\d+ ###$", rest, re.M)
    nxt = re.search(r"^### RUNG \d+/\d+ \S+ ###$", rest, re.M)
    if end and (not nxt or end.start() < nxt.start()):
        s, note = rest[:end.start()], ""
    else:   # a rung that did not return has no END line: read to the next rung, or to the end of the transcript
        s, note = (rest[:nxt.start()] if nxt else rest), " [no END marker: the rung did not return]"
    k, why = classify(s)
    return k, why + note


def delin_fault(s):
    """A domain fault with cause 26 at map 0 base + 0x1f60, the base taken from the ADD_OK slot=0 line before it."""
    base = None
    for l in s.splitlines():
        m = re.search(r"CAPSTONE_VM_ADD_OK slot=0 base=0x([0-9a-f]+)", l)
        if m: base = int(m.group(1), 16)
        m = re.search(r"domain fault cause=26 pc=0x([0-9a-f]+)", l)
        if m and base is not None and int(m.group(1), 16) - base == 0x1f60:
            return l.strip()
    return None


def qemu_csr_fault(s):
    """A domain fault with cause 2 whose tval decodes as csrr*/csrw* on 0xcc0 (urevavail) or 0x5c0 (scapctl)."""
    for m in re.finditer(r"domain fault cause=2 pc=0x[0-9a-f]+ address=0x([0-9a-f]+)", s):
        w = int(m.group(1), 16)
        if w & 0x7F == 0x73 and (w >> 12) & 7 in (1, 2, 3, 5, 6, 7) and (w >> 20) in (0xcc0, 0x5c0):
            return f"tval {w:#010x}, CSR {w >> 20:#x}"
    return None


def classify(s):
    ok = "VIRTUAL_APPLICATION_OK" in s
    pre = s.split("VIRTUAL_APPLICATION_OK")[0] if ok else s
    mon = " ".join(re.findall(rf"(?:EXCX|ILLX|MCAU|MEPC|MTVL|MSTA|INJS|NODM|INJL):{TAG}", s))
    if re.search(rf"(?:NODM|INJL):{TAG}", s):
        return "monitor-stop", mon
    if ok and re.search(r"VM_EXIT:stale:139", s) and "domain fault cause=25" in s and not re.search(rf"INJS:{TAG}", s):
        return "identity", "VIRTUAL_APPLICATION_OK, then the stale access denied with 25 (exit 139)"
    q = qemu_csr_fault(s)
    if q and not ok:
        return "qemu-csr", f"cause 2 on a QEMU-only CSR instruction ({q}): the read was not removed (Q-21/Q-22), kit side"
    d = delin_fault(s)
    if d and not ok:
        return "delin", "cause 26 at map 0 base + 0x1f60: the unfixed image-gp delin (C-27), kit side"
    if not ok and "domain fault cause=29" in pre:
        return "layout", "29 before VIRTUAL_APPLICATION_OK (kit side; to the RTL lane with the table dump)"
    if re.search(rf"INJS:{TAG}", s):
        return "injected", mon
    return "other", f"ok-line={ok} exits={re.findall(r'VM_EXIT:\S+', s)} causes={re.findall(r'cause=\d+', s)} monitor=[{mon}]"


def seg(body, rc=0):
    return f"$ echo '### RUNG 3/4 vm-contract ###'; ...\n### RUNG 3/4 vm-contract ###\n{body}\n### RUNG 3 vm-contract END rc={rc} ###\nDN_{rc}\n"


def selftest():
    ident = "VIRTUAL_APPLICATION_OK 1\nVM_EXIT:ok:0\ndomain fault cause=25 pc=0x10\nVM_EXIT:stale:139\nRESULT vm-contract retval=0"
    inj = "EXCX:0000E002\nMCAU:00000018\nMEPC:FFFFFFFF01234567\nMSTA:0000000A00000900\nINJS:00000018\nOops\nRESULT vm-contract retval=922"
    cases = [(seg(ident), "identity"),
             (seg("domain fault cause=29 pc=0x20\nVM_EXIT:ok:139\nRESULT vm-contract retval=922"), "layout"),
             (seg(inj), "injected"),
             (seg(ident + "\nEXCX:0000E002\nINJS:00000018"), "injected"),   # an injection taints the identity reading
             (seg("EXCX:0000E002\nMCAU:00000019\nNODM:00000019"), "monitor-stop"),
             (seg("VM_EXIT:ok:1\nRESULT vm-contract retval=922"), "other"),
             ("$ echo '### RUNG 3/4 vm-contract ###'; x; echo \"### RUNG 3 vm-contract END rc=$rc ###\"\n", "NO DATA"),
             (seg(inj)[:seg(inj).index("### RUNG 3 vm-contract END")].replace("INJS:00000018\n", ""), "other"),   # wedged
             (seg(ident) + seg(ident), "NO DATA"),
             (seg("capstone-exec: domain fault cause=2 pc=0x3f92210ccc address=0xcc002573\nVM_EXIT:normal:139\nRESULT vm-contract retval=922"), "qemu-csr"),
             (seg("capstone-exec: domain fault cause=2 pc=0x3f92210ccc address=0xc0102573\nVM_EXIT:normal:139\nRESULT vm-contract retval=922"), "other"),
             (seg("CAPSTONE_VM_ADD_OK slot=0 base=0x3f87280000 bytes=524288 reg=3 perms=7 linear=0\ncapstone-exec: domain fault cause=26 pc=0x3f87281f60 address=0x3f872a1ff0\nVM_EXIT:normal:139\nRESULT vm-contract retval=922"), "delin"),
             (seg("CAPSTONE_VM_ADD_OK slot=0 base=0x3f87280000 bytes=524288 reg=3 perms=7 linear=0\ncapstone-exec: domain fault cause=26 pc=0x3f87281f64 address=0x3f872a1ff0\nVM_EXIT:normal:139\nRESULT vm-contract retval=922"), "other")]
    bad = 0
    for t, want in cases:
        got = branch(t)[0]
        bad += got != want
        print(f"{'ok  ' if got == want else 'FAIL'} want {want:13} got {got}")
    print("PASS" if not bad else f"{bad} FAILED")
    return 1 if bad else 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        sys.exit(selftest())
    k, why = branch(open(sys.argv[1], errors="replace").read())
    print(f"VM-CONTRACT {'NO DATA' if k == 'NO DATA' else 'BRANCH ' + k}: {why}")
    sys.exit(2 if k == "NO DATA" else 0)
