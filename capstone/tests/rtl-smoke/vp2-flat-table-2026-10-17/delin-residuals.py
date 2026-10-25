#!/usr/bin/env python3
"""delin-residuals.py <image>...: list the delins the census does NOT count as gp-derived.

The board lane's delin-census.py counts `delin rd` whose NEAREST writer within 5 instructions is
scc/cincoffset/cincoffsetimm with gp as rs2. Those are being dropped by the C-27 fix. Every OTHER
delin survives the fix, so each one is either a legitimate LINEAR operand or the next cause 26 --
which is the question the board needs answered before an image goes near the board.

Parsing and the window rule are lifted from delin-census.py deliberately, so the residual set is
exactly the complement of its count; the script asserts that (residuals == delins - gp_derived)
per image and exits non-zero if it does not, because a complement that does not add up means the
two scripts disagree about the population and neither number means anything.
"""
import re, subprocess, sys, os, hashlib, collections

WINDOW = 5
INSN = re.compile(r"\s*([0-9a-f]+):\s+(?:[0-9a-f]{2} ){4}\s*(\S+)\s*(.*)")
DERIVE = {"cincoffset", "cincoffsetimm", "scc"}

def disas(objdump, path):
    r = subprocess.run([objdump, "-d", path], capture_output=True, text=True)
    return r.stdout.splitlines()

def analyse(lines):
    ins = []
    for l in lines:
        m = INSN.match(l)
        if m:
            ins.append((m.group(1), m.group(2), m.group(3)))
    delins = gp = 0
    residuals = []
    for i, (addr, mn, args) in enumerate(ins):
        if mn != "delin":
            continue
        delins += 1
        rd = args.split(",")[0].strip()
        writer = None
        for j in range(i - 1, max(i - 1 - WINDOW, -1), -1):
            regs = [x.strip() for x in ins[j][2].split(",")]
            if regs and regs[0] == rd:
                writer = ins[j]
                break
        if writer and writer[1] in DERIVE:
            regs = [x.strip() for x in writer[2].split(",")]
            if len(regs) > 1 and regs[1] == "gp":
                gp += 1
                continue
        residuals.append((addr, rd, writer, ins[max(0, i - 10):i]))
    return len(ins), delins, gp, residuals

def main(argv):
    objdump = os.environ.get("LLVM_OBJDUMP", "llvm-objdump")
    rc = 0
    for path in argv:
        lines = disas(objdump, path)
        n, d, g, res = analyse(lines)
        if n == 0:
            print(f"NO DATA {path}: 0 instructions from {objdump}"); return 2
        sha = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
        print(f"=== {os.path.basename(path)} ({sha}): {d} delin, {g} gp-derived, "
              f"{len(res)} RESIDUAL")
        if len(res) != d - g:
            print(f"  COMPLEMENT MISMATCH: {len(res)} != {d} - {g}; the two scripts disagree")
            rc = 1
        kinds = collections.Counter()
        for addr, rd, writer, ctx in res:
            w = f"{writer[1]} {writer[2]}" if writer else "NO WRITER IN WINDOW"
            kinds[(writer[1] if writer else "none")] += 1
            print(f"  {addr}: delin {rd}   <- nearest writer: {w}")
            for a, m2, ar in ctx:
                print(f"        {a}: {m2} {ar}")
        print(f"  residual writers: {dict(kinds)}")
    return rc

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
