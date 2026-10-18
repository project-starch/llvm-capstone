#!/usr/bin/env python3
"""score-b5.py <boot 5 board.txt> <boot 4c board.txt>: boot 5 against its pre-registration, by script.
- the 20 exits, in order, against the prediction (b5-prediction.txt);
- the 17 returning steps' fault lines against 4c's, matched by step: cause, pc-code and entry-code offsets (the fault
  address is a heap address and moves on every boot, so it is printed, not compared);
- the read-backs, the kit lines, the BEGUN list.
Exit 0 only if every compared field equals; 2 on no data (the vm-suite segment missing)."""
import re, sys
PRED = [("normal", 0), ("threads", 1), ("vm", 1), ("nested", 0), ("nested-stale", 139), ("heap-threads", 1), ("stale", 139),
        ("bounds", 139), ("retired", 139), ("vm-ro-write", 139), ("vm-none-read", 139), ("vm-guard", 139),
        ("vm-padding", 139), ("preempt", 143), ("parallel_one", 0), ("parallel_two", 0), ("sqlite_write", 0),
        ("sqlite_read", 0), ("mruby", 0), ("cleanup", 1)]
KIT = ["SQLITE_SUM=5050", "SQLITE_PERSIST=100", "MRUBY_SUM=5050", "MRUBY_FILE=virtual"]
FAULT = re.compile(r"domain fault cause=(\d+) pc=0x([0-9a-f]+) address=0x([0-9a-f]+) entry=0x([0-9a-f]+) code=0x([0-9a-f]+)-")


def segment(path):
    t = open(path, errors="replace").read().replace("\r", "")
    st = list(re.finditer(r"^### RUNG \d+/\d+ vm-suite ###$", t, re.M))
    if len(st) != 1:
        return None
    rest = t[st[0].end():]
    end = re.search(r"^### RUNG \d+ vm-suite END rc=-?\d+ ###$", rest, re.M)
    return rest[:end.start()] if end else rest


def faults_by_step(seg):
    """{step: [(cause, pc-code, entry-code, addr-code)]}: each fault line belongs to the next VM_EXIT after it."""
    out, pending = {}, []
    for line in seg.split("\n"):
        m = FAULT.search(line)
        if m and not line.startswith("KMSG"):
            c, pc, ad, en, lo = m.groups()
            pc, ad, en, lo = (int(x, 16) for x in (pc, ad, en, lo))
            pending.append((int(c), pc - lo, en - lo, ad - lo))
        e = re.match(r"^VM_EXIT:([a-z_-]+):(\d+)$", line.strip())
        if e:
            out[e.group(1)] = pending
            pending = []
    return out


def main():
    b5, b4c = segment(sys.argv[1]), segment(sys.argv[2])
    if b5 is None or b4c is None:
        print("NO DATA: a vm-suite segment is missing"); return 2
    bad = 0
    exits = re.findall(r"^VM_EXIT:([a-z_-]+):(\d+)$", b5, re.M)
    got = [(n, int(v)) for n, v in exits]
    print(f"exits: {len(got)} read, {len(PRED)} predicted")
    for i, p in enumerate(PRED):
        g = got[i] if i < len(got) else None
        ok = g == p
        bad += not ok
        print(f"  {'ok  ' if ok else 'DIFF'} {p[0]:14} predicted {p[1]:3}  read {g[1] if g and g[0] == p[0] else g}")
    f5, f4 = faults_by_step(b5), faults_by_step(b4c)
    print("returning steps' fault lines against 4c (cause, pc-code, entry-code; addr-code printed only):")
    for name, _ in PRED[:16]:
        a = [x[:3] for x in f5.get(name, [])]
        b = [x[:3] for x in f4.get(name, [])]
        ok = a == b
        bad += not ok
        if a or b or not ok:
            print(f"  {'ok  ' if ok else 'DIFF'} {name:14} b5 {[(c, hex(p), hex(e)) for c, p, e in a]}  4c {[(c, hex(p), hex(e)) for c, p, e in b]}"
                  f"  addr b5 {[hex(x[3]) for x in f5.get(name, [])]}")
    for k in KIT:
        n = len(re.findall(rf"^{re.escape(k)}$", b5, re.M))
        bad += n != 1
        print(f"  {'ok  ' if n == 1 else 'DIFF'} {k} x{n}")
    for rb, want in (("ARENA_PIN", ["ARENA_PIN:Y:suite"]),
                     ("RESOLVE_DIRTY", ["RESOLVE_DIRTY:N:default", "RESOLVE_DIRTY:Y:before-vm", "RESOLVE_DIRTY:N:after-vm"])):
        g = re.findall(rf"^{rb}:[0-9YN]+:[a-z-]+$", b5, re.M)
        bad += g != want
        print(f"  {'ok  ' if g == want else 'DIFF'} {rb} {g}")
    print("BEGUN:", re.findall(r"^VM_BEGIN:([a-z_]+)$", b5, re.M))
    print("PASS: every compared field equals" if not bad else f"{bad} DIFFERENT")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
