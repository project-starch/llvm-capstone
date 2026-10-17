#!/usr/bin/env python3
"""Count QEMU-only CSR accesses (0xcc0 urevavail, 0x5c0 scapctl) in a domain image.

Q-21: 0xcc0 is ABSENT on the RTL, so a read is an illegal instruction on silicon while QEMU
implements it. An image containing one passes on QEMU and faults on vp2 -- the C-27 shape one level
out. Gate: 0. The pre-fix images are the positive control.

Counted by ENCODING, not by mnemonic: objdump prints some custom words as .4byte, and a mnemonic
grep reads 0 (this project has been bitten by exactly that). A SYSTEM/csr instruction is opcode
0x73 with funct3 in {1,2,3,5,6,7}; the CSR number is bits [31:20].
"""
import re, subprocess, sys, hashlib, pathlib, collections

OBJDUMP = "/mnt/hdd/$USER/capstone/llvm-lane-c27/bin/llvm-objdump"
WATCH = {0xcc0: "urevavail(0xcc0)", 0x5c0: "scapctl(0x5c0)"}
INSN = re.compile(r"^\s*([0-9a-f]+):\s+((?:[0-9a-f]{2} ){4})\s*(.*)$")

def words(path):
    r = subprocess.run([OBJDUMP, "-d", str(path)], capture_output=True, text=True)
    out = []
    for l in r.stdout.splitlines():
        m = INSN.match(l)
        if m:
            b = m.group(2).split()
            out.append((int(m.group(1), 16), int("".join(reversed(b)), 16), m.group(3)))
    return out

def funcs(path):
    r = subprocess.run([OBJDUMP.replace("objdump", "nm"), str(path)],
                       capture_output=True, text=True)
    s = []
    for l in r.stdout.splitlines():
        p = l.split()
        if len(p) == 3 and p[1].upper() == "T" and not p[2].startswith(".L"):
            try: s.append((int(p[0], 16), p[2]))
            except ValueError: pass
    s.sort()
    return s

def owner(syms, a):
    name, base = "?", 0
    for s, n in syms:
        if s <= a: name, base = n, s
        else: break
    return f"{name}+0x{a-base:x}"

def main(argv):
    if not argv:
        print("NO DATA: no image given", file=sys.stderr); return 2
    rc = 0
    for a in argv:
        p = pathlib.Path(a)
        if not p.exists():
            print(f"  {p.name:<26} ABSENT"); rc = 2; continue
        ws = words(p)
        if not ws:
            print(f"NO DATA: {p} disassembled to 0 instructions", file=sys.stderr); return 2
        syms = funcs(p)
        hits = collections.Counter(); sites = []
        for addr, w, txt in ws:
            if (w & 0x7f) == 0x73 and ((w >> 12) & 7) in (1, 2, 3, 5, 6, 7):
                csr = w >> 20
                if csr in WATCH:
                    hits[WATCH[csr]] += 1
                    sites.append(f"0x{addr:x} {owner(syms, addr)} {WATCH[csr]}")
        sha = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        tot = sum(hits.values())
        print(f"  {p.name:<26} {sha} insns={len(ws):>7} QEMU-only-CSR={tot}"
              f"{'  <-- FAULTS ON vp2' if tot else '  ok'}")
        for s in sites: print(f"       {s}")
        if tot: rc = 1
    return rc

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
