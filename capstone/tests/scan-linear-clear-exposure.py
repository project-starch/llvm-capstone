#!/usr/bin/env python3
"""Scan built Capstone artifacts for sites affected by the missing linear-source clear.

Two independent RTL findings motivate this:

  A  CINCOFFSET / CINCOFFSETIMM / SCC / TIGHTEN / SHRINKTO / INIT do not cnull a
     non-NONLIN rs1, though the spec defines them as "MOVC rd, rs1" + an edit.
  B  STC does not cnull a non-NONLIN rs2 (mem-access-insn.adoc:105).

The question this answers is NOT "is the RTL wrong" -- that is settled elsewhere -- but
"if the RTL were MADE CONFORMANT, would code we actually build break?".  LLVM does not
model either clear (CapstoneInstrInfo.td: STC has an empty (outs); CIncOffset takes rs1 as
a pure use), so every site where a cleared register is read afterwards is a latent break.

Each rule is reported at two severities:

  SURFACE   the instruction would behave differently under a conformant RTL
  BREAK     ... and the affected register is READ again before being redefined,
            so the difference is observable and the code is wrong under the fix

Method and its limits, stated plainly: this is a LINEAR scan of disassembly within each
symbol, with no control-flow graph.  Branches are ignored, so a read on a path not taken
is still counted (over-report) and a read reached only via a back-edge from before the
site is missed (under-report).  Over-reporting is the safe direction here and BREAK hits
are meant to be triaged by hand, not trusted as a count.

Hits are classified by the PROVENANCE of the affected register: 'linear' when its last
definition mints one (split/mrev/init) with no delin since, 'nonlin' after a delin, and
'unknown' when it arrives from outside the function.  Only 'linear' is reported by default.
`unknown` is dominated by sp and gp -- which the entry glue delins, invisibly to an
intra-function scan -- and reporting it buries every real hit under thousands.

Exit status: 0 clean, 1 hits found, 2 the scan could not run (no input, no objdump).
"no data" is an ERROR here, never a clean result.
"""

import argparse
import collections
import re
import subprocess
import sys

# rd is operand 0, rs1 is operand 1 for all of these.
FINDING_A_OPS = {"cincoffset", "cincoffsetimm", "scc", "tighten", "shrinkto", "init"}

# Instructions that MINT a linear capability into their destination.
#
# llvm-objdump does not know these encodings and prints "<unknown>", so provenance is
# decoded from the raw bytes instead.  Discovered by a control that STOPPED FIRING when the
# scan was retargeted -- without the raw decode every split/mrev/init is invisible and the
# scan reports a clean zero having examined nothing.  Keyed on (funct7, funct3) of the R-type
# 0x5B opcode, per verif/tests/custom/capstone/asm_insn.h:13-24.
LINEAR_MINTING = {"split", "mrev", "init"}
RAW_R_OPS = {(0x6, 0x1): "split", (0x8, 0x1): "mrev", (0x9, 0x1): "init",
             (0x1, 0x1): "shrink", (0x2, 0x1): "tighten", (0x7, 0x1): "seal",
             (0xa, 0x1): "movc", (0xc, 0x1): "cincoffset", (0x3, 0x1): "delin"}
# The entry glue delins these before calling anything (my_first_domain/start.S:35,41;
# start-fpga-nogp.S:59-60), so an operand arriving from outside a function in one of these
# is NONLIN by construction.  Everything else arriving from outside stays 'unknown' -- which
# is where cross-function linear provenance lives, e.g. start-fpga-nogp.S carves a LINEAR a2
# in __test_entry and stores it in `test`, a site the intra-function scan cannot otherwise
# see.  That site is this scan's specified positive control, so this list must stay MINIMAL.
ASSUMED_NONLIN = {"sp", "gp", "tp"}
ABI = ["zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2", "s0", "s1",
       "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "s2", "s3", "s4",
       "s5", "s6", "s7", "s8", "s9", "s10", "s11", "t3", "t4", "t5", "t6"]


def decode_raw(byte_str):
    """Decode an R-type 0x5B instruction objdump printed as <unknown>."""
    try:
        b = bytes(int(x, 16) for x in byte_str.split())
    except ValueError:
        return None
    if len(b) != 4:
        return None
    word = int.from_bytes(b, "little")
    if word & 0x7F != 0x5B:
        return None
    name = RAW_R_OPS.get(((word >> 25) & 0x7F, (word >> 12) & 0x7))
    if name is None:
        return None
    rd, rs1, rs2 = (word >> 7) & 0x1F, (word >> 15) & 0x1F, (word >> 20) & 0x1F
    return name, [ABI[rd], ABI[rs1], ABI[rs2]]

# Instructions that do NOT define their first operand.
STORES = {"sd", "sw", "sh", "sb", "stc", "fsd", "fsw"}
BRANCHES = {"beq", "bne", "blt", "bge", "bltu", "bgeu", "beqz", "bnez",
            "blez", "bgez", "bltz", "bgtz", "j", "jr", "ret", "cjalr"}

LINE = re.compile(r"^\s*([0-9a-f]+):\s+((?:[0-9a-f]{2} )+)\s*\t(\S+)\s*(.*)$")
SYMBOL = re.compile(r"^[0-9a-f]+\s+<(.+)>:\s*$")
# `stc t2, 0x0(gp)` / `ldc a0, 0x50(sp)`
MEMOP = re.compile(r"^(\S+),\s*(-?0x[0-9a-f]+|-?\d+)\((\S+)\)$")

Insn = collections.namedtuple("Insn", "addr mnem ops raw")


def disassemble(objdump, path):
    out = subprocess.run(
        [objdump, "-d", "--triple=capstone64-unknown-elf", path],
        capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"objdump failed on {path}: {out.stderr.strip()[:200]}")
    funcs, cur = collections.OrderedDict(), None
    for line in out.stdout.splitlines():
        m = SYMBOL.match(line)
        if m:
            cur = m.group(1)
            funcs.setdefault(cur, [])
            continue
        m = LINE.match(line)
        if m and cur is not None:
            addr, raw_bytes, mnem, ops = m.group(1), m.group(2), m.group(3), m.group(4).strip()
            if mnem == "<unknown>":
                dec = decode_raw(raw_bytes)
                if dec is None:
                    continue
                mnem, oplist = dec
                ops = ", ".join(oplist)
                funcs[cur].append(Insn(addr, mnem, oplist, ops))
                continue
            funcs[cur].append(Insn(addr, mnem, [o.strip() for o in ops.split(",")], ops))
    return funcs


def operand_regs(insn):
    """(defined_reg | None, [used_regs])."""
    if insn.mnem in BRANCHES:
        return None, [o for o in insn.ops if o and not o.startswith("0x")]
    if insn.mnem in STORES:
        m = MEMOP.match(insn.raw)
        if m:
            return None, [m.group(1), m.group(3)]
        return None, [o for o in insn.ops if o]
    m = MEMOP.match(insn.raw)          # loads: `ldc a0, 0x50(sp)`
    if m:
        return m.group(1), [m.group(3)]
    if not insn.ops or not insn.ops[0]:
        return None, []
    defd = insn.ops[0]
    if not re.match(r"^[a-z][a-z0-9]*$", defd):
        return None, []
    used = [o for o in insn.ops[1:] if re.match(r"^[a-z][a-z0-9]*$", o)]
    return defd, used


def provenance(defs, insns, idx, reg):
    """Classify reg at idx: 'linear' | 'nonlin' | 'unknown'.

    'linear' only when reg's last definition before idx is an instruction that MINTS a
    linear capability and no delin followed.  That is the only class this scan reports by
    default: it is the one where a conformant RTL would demonstrably change behaviour.

    'unknown' covers registers arriving from outside the function -- including sp and gp,
    which the domain entry glue delins (my_first_domain/start.S:35,41) but which no
    intra-function scan can see.  Reporting those drowns the result: they are the
    overwhelmingly common operand and are safe by construction.
    """
    if reg == "zero":
        return "nonlin"
    for j in range(idx - 1, -1, -1):
        if insns[j].mnem == "delin" and insns[j].ops and insns[j].ops[0] == reg:
            return "nonlin"
        if defs[j] == reg:
            return "linear" if insns[j].mnem in LINEAR_MINTING else "unknown"
    return "nonlin" if reg in ASSUMED_NONLIN else "unknown"


def read_before_redef(defs, uses, idx, reg):
    """First index > idx that READS reg, or None if redefined/end first."""
    for j in range(idx + 1, len(defs)):
        if reg in uses[j]:
            return j
        if defs[j] == reg:
            return None
    return None


def scan_function(path, fname, insns, classes):
    hits = []
    decoded = [operand_regs(ins) for ins in insns]      # once, not per query
    defs = [d for d, _ in decoded]
    uses = [frozenset(u) for _, u in decoded]
    for i, ins in enumerate(insns):
        rule = None
        if ins.mnem in FINDING_A_OPS and len(ins.ops) >= 2:
            rd, rs1 = ins.ops[0], ins.ops[1]
            prov = provenance(defs, insns, i, rs1)
            if rd != rs1 and prov in classes:
                rule, victim = "A", rs1
        elif ins.mnem == "stc":
            m = MEMOP.match(ins.raw)
            prov = provenance(defs, insns, i, m.group(1)) if m else "nonlin"
            if m and prov in classes:
                rule, victim = "B", m.group(1)
        if rule is None:
            continue
        j = read_before_redef(defs, uses, i, victim)
        severity = "BREAK" if j is not None else "SURFACE"
        hits.append(dict(file=path, func=fname, addr=ins.addr, rule=rule,
                         severity=severity, victim=victim, prov=prov, insn=f"{ins.mnem} {ins.raw}",
                         read_at=insns[j].addr if j is not None else None,
                         read_insn=f"{insns[j].mnem} {insns[j].raw}" if j is not None else None))
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--objdump", default="llvm/cmake-build-debug/bin/llvm-objdump")
    ap.add_argument("--severity", choices=["SURFACE", "BREAK"], default="SURFACE")
    ap.add_argument("--quiet", action="store_true", help="totals only")
    ap.add_argument("--classes", default="linear",
                    help="comma-separated provenance classes to report: linear,unknown,nonlin. "
                         "Default 'linear' -- the only class where a conformant RTL demonstrably "
                         "changes behaviour. Add 'unknown' for the conservative sweep, which "
                         "reports every sp/gp operand and is dominated by them.")
    args = ap.parse_args()
    classes = set(args.classes.split(","))

    all_hits, scanned, failed = [], 0, []
    for path in args.files:
        try:
            funcs = disassemble(args.objdump, path)
        except Exception as exc:                      # noqa: BLE001
            failed.append((path, str(exc)))
            continue
        if not funcs:
            failed.append((path, "no disassembly produced"))
            continue
        scanned += 1
        for fname, insns in funcs.items():
            all_hits.extend(scan_function(path, fname, insns, classes))

    if scanned == 0:
        print("ERROR: nothing was disassembled -- the scan checked NOTHING.", file=sys.stderr)
        for p, e in failed:
            print(f"  {p}: {e}", file=sys.stderr)
        return 2

    keep = [h for h in all_hits if args.severity == "SURFACE" or h["severity"] == "BREAK"]
    if not args.quiet:
        for h in sorted(keep, key=lambda h: (h["severity"] != "BREAK", h["file"], h["addr"])):
            print(f"{h['severity']:7s} rule-{h['rule']}  {h['file']}  <{h['func']}>+{h['addr']}")
            print(f"          {h['insn']}   [{h['victim']} provenance={h['prov']}]")
            if h["read_at"]:
                print(f"          -> READ at {h['read_at']}: {h['read_insn']}")

    nbreak = sum(1 for h in all_hits if h["severity"] == "BREAK")
    print(f"\nscanned {scanned} file(s); "
          f"{sum(1 for h in all_hits if h['rule'] == 'A')} rule-A, "
          f"{sum(1 for h in all_hits if h['rule'] == 'B')} rule-B; "
          f"{nbreak} BREAK, {len(all_hits) - nbreak} SURFACE")
    for p, e in failed:
        print(f"NOT SCANNED: {p}: {e}", file=sys.stderr)
    return 1 if keep else 0


if __name__ == "__main__":
    sys.exit(main())
