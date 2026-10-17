#!/usr/bin/env python3
"""vp2-silicon-gate.py [--table TSV] [--column COL] <image.dom | module.ko>...: every instruction of a staged file against
the RTL lane's vp2 silicon census (capstone/docs/plans/2026-10-18-vp2-silicon-census.tsv), read at run time.

Three classes cost a boot each on 2026-10-18, each one QEMU implements and the RTL refuses:
- an LDC/STC with an integer base (Q-23);
- a DELIN of a NONLIN capability (Q-25, C-27);
- a read of CSR 0xcc0 or 0x5c0 (Q-21, Q-22).
This gate checks a file against the table's columns, so a change to the table changes the gate. Nothing about the
decoder is written here. The checks:
- every custom-2 (0x5b) and custom-3 (0x7b) instruction is matched to its most specific `insn` row (opcode, funct3,
  funct7; the "other" rows last), and its cell in COLUMN is read;
- SFENCE.VMA, MRET and SRET by their rows;
- every Zicsr instruction: by its `csr` row (a range such as 0xfc0-0xfc3 included). A CSR not in the table falls under
  the `CSR privilege` rule (csr[9:8] above the column's privilege is illegal; csr[11:10] = 11 written is illegal);
- two operand rules that static analysis can see:
  - DELIN whose operand was last written by cincoffset, cincoffsetimm or scc off gp, within 5 instructions;
  - LDC/STC whose base was last written by an integer instruction, within 8.
How a cell reads:
- one that starts with "ok", or names EMULATED, admits the instruction; conditional ones are counted separately;
- one that starts with "illegal" refuses it;
- anything else ("n/a", "see note") is a candidate to attribute by hand, and is reported.
COLUMN defaults to in_context for a .dom (applications run inside a CSRUNV context) and trusted_S for a .ko.
Exit codes: 0 when nothing is refused and there are no candidates; 1 when something is refused or there are
candidates; 2 on no data (a missing table or objdump, or a file with no instruction). No data is never a pass.
"""
import argparse, csv, os, re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve()
DEFAULT_TABLE = HERE.parents[4] / "capstone/docs/plans/2026-10-18-vp2-silicon-census.tsv"
LLVM = os.environ.get("LLVM_OBJDUMP", os.path.expanduser("~/dev/llvm-capstone/llvm/cmake-build-debug/bin/llvm-objdump"))
CROSS = os.environ.get("CROSS_OBJDUMP", os.path.expanduser(
    "~/dev/llvm-capstone/capstone/caplifive-system/sw/buildroot/build-fpga/host/bin/riscv64-buildroot-linux-gnu-objdump"))
COLUMNS = ("U", "trusted_S", "M_scalar", "M_legacy_C", "in_context")
PRIV = {"U": 0, "in_context": 0, "trusted_S": 1, "M_scalar": 3, "M_legacy_C": 3}
GP = 3


def load_table(path):
    rows = list(csv.reader((l for l in open(path) if not l.startswith("#")), delimiter="\t"))
    hdr, body = rows[0], [dict(zip(rows[0], r)) for r in rows[1:] if r and r[0]]
    for c in COLUMNS + ("class", "name", "encoding"):
        if c not in hdr: raise SystemExit(f"NO DATA: table {path} has no column {c!r}")
    insn, csr = [], []
    for r in body:
        if r["class"] == "insn":
            e = r["encoding"]
            m = re.match(r"(0x[0-9a-f]+)(?:\s+f3=([01]{3}))?(?:\s+f7=(0x[0-9a-f]+|other))?", e)
            if m:
                op = int(m.group(1), 16)
                f3 = int(m.group(2), 2) if m.group(2) else None
                f7 = None if m.group(3) in (None, "other") else int(m.group(3), 16)
                catch = "other" in e
                insn.append({"op": op, "f3": f3, "f7": f7, "catch": catch, "row": r})
            elif r["name"].startswith("SFENCE"):
                insn.append({"std": "sfence", "row": r})
            elif r["name"].startswith("MRET"):
                insn.append({"std": "xret", "row": r})
        elif r["class"] == "csr":
            m = re.match(r"(0x[0-9a-f]+)(?:-(0x[0-9a-f]+))?", r["encoding"])
            if m:
                lo = int(m.group(1), 16); hi = int(m.group(2), 16) if m.group(2) else lo
                csr.append({"lo": lo, "hi": hi, "row": r})
    if not insn or not csr: raise SystemExit(f"NO DATA: table {path} has no insn or csr rows")
    return insn, csr


def match_insn(insn_rows, w):
    op, f3, f7 = w & 0x7F, (w >> 12) & 7, w >> 25
    best = None
    for e in insn_rows:
        if "std" in e or e["op"] != op: continue
        if e["f3"] is not None and e["f3"] != f3: continue
        if e["f7"] is not None and e["f7"] != f7: continue
        if e["f3"] is not None and e["f7"] is None and not e["catch"] and f3 == 1: continue   # f3=001 rows name an f7
        score = (e["f3"] is not None) + (e["f7"] is not None) - 2 * e["catch"]
        if best is None or score > best[0]: best = (score, e)
    return best[1] if best else None


def verdict(cell):
    c = cell.strip()
    if "EMULATED" in c: return "ok"
    if c.startswith("ok") or c in ("RW", "RO") or c.startswith("RW") or c.startswith("RO"):
        return "ok" if c in ("ok", "RW", "RO") or c.startswith(("RW ", "RO ")) else "conditional"
    if c.startswith("illegal"): return "illegal"
    return "candidate"


def words(lines):
    """(address, function, word, starts_block) for each 4-byte instruction. A block starts after any label line, so a
    backward scan for an operand's writer stops there instead of reading another function's or path's code."""
    fn, brk = None, True
    for l in lines:
        h = re.match(r"[0-9a-f]+ <([^>]*)>:", l)
        if h:
            if not h.group(1).startswith("."): fn = h.group(1)
            brk = True
            continue
        m = re.match(r"\s*([0-9a-f]+):\s+((?:[0-9a-f]{2} ){4})\s", l)
        if m:
            yield int(m.group(1), 16), fn, int.from_bytes(bytes.fromhex(m.group(2).replace(" ", "")), "little"), brk; brk = False; continue
        m = re.match(r"\s*([0-9a-f]+):\s+([0-9a-f]{8})\s", l)
        if m:
            yield int(m.group(1), 16), fn, int(m.group(2), 16), brk; brk = False


def rd_of(w):
    op = w & 0x7F
    if op in (0x23, 0x63, 0x0F): return None       # store, branch, fence: no rd
    if op == 0x5B and (w >> 12) & 7 == 4: return None   # STC
    return (w >> 7) & 31


def is_transfer(w):
    return (w & 0x7F) in (0x63, 0x6F, 0x67) or ((w & 0x7F) == 0x5B and (w >> 12) & 7 in (5, 6))   # B, JAL, JALR, CJALR, CBNZ


UNIMP = 0xC0001073   # csrrw x0, cycle, x0: the canonical deliberately-illegal instruction (__builtin_trap); traps alike on QEMU and the RTL


def writer_before(seq, i, reg, window):
    """The nearest earlier instruction in the same straight-line block that writes reg, or None."""
    for j in range(i - 1, max(i - 1 - window, -1), -1):
        if seq[j + 1][3] if j + 1 <= i else False: return None      # seq[j+1] starts a block: nothing before it counts
        w = seq[j][2]
        if is_transfer(w): return None
        if rd_of(w) == reg: return w
    return None


def integer_kind(w):
    """'integer' when the writer certainly produces an integer (lui, auipc, a plain load, li = addi from x0, a link);
    'alu' when it is integer ALU arithmetic on a register that may hold a capability (addi/add/mv from rs1 != x0),
    whose result type on the RTL the census table has no row for; None otherwise."""
    op = w & 0x7F
    if op in (0x37, 0x17, 0x03, 0x6F, 0x67): return "integer"
    if op in (0x13, 0x1B) and (w >> 15) & 31 == 0: return "integer"
    if op in (0x13, 0x33, 0x1B, 0x3B): return "alu"
    return None


def is_gp_derive(w):
    op, f3, f7, rs1 = w & 0x7F, (w >> 12) & 7, w >> 25, (w >> 15) & 31
    if op != 0x5B or rs1 != GP: return False
    return (f3 == 1 and f7 in (0x0C, 0x05)) or f3 == 2     # CINCOFFSET, SCC, CINCOFFSETIMM


def gate(lines, insn_rows, csr_rows, column):
    seq = list(words(lines))
    refused, cand, cond, n, traps = [], [], 0, len(seq), 0
    priv = PRIV[column]
    for i, (addr, fn, w, _brk) in enumerate(seq):
        op, f3 = w & 0x7F, (w >> 12) & 7
        hit = None
        if op in (0x5B, 0x7B):
            e = match_insn(insn_rows, w)
            if e is None:
                cand.append((addr, fn, w, "no table row matches")); continue
            v = verdict(e["row"][column]); hit = e["row"]["name"]
            if v == "illegal": refused.append((addr, fn, w, f"{hit}: {column} = {e['row'][column]}"))
            elif v == "candidate": cand.append((addr, fn, w, f"{hit}: {column} = {e['row'][column]}"))
            elif v == "conditional": cond += 1
            # operand rules
            if op == 0x5B and f3 == 1 and (w >> 25) == 0x03:                     # DELIN: operand = rd
                wr = writer_before(seq, i, (w >> 7) & 31, 5)
                if wr is not None and is_gp_derive(wr):
                    refused.append((addr, fn, w, "DELIN of a gp-derived operand (gp is NONLIN by ABI; Q-25)"))
            if op == 0x5B and f3 in (3, 4) and column == "trusted_S":
                # Kernel code: a register is an integer unless a capability instruction produced it, and v7's contract
                # (Q-23) is that trusted S makes no capability memory access. So the base's nearest writer in the same
                # function must be a capability instruction (custom-2/3 with an rd); otherwise the LDC/STC is refused.
                r = (w >> 15) & 31
                src = None
                for j in range(i - 1, -1, -1):
                    if seq[j][1] != fn: break
                    if rd_of(seq[j][2]) == r: src = seq[j][2]; break
                if src is None or (src & 0x7F) not in (0x5B, 0x7B):
                    refused.append((addr, fn, w, f"{'LDC' if f3 == 3 else 'STC'} in trusted S with base x{r} not from a capability instruction (Q-23)"))
                continue
            if op == 0x5B and f3 in (3, 4):                                       # LDC / STC base
                r = (w >> 15) & 31
                wr = writer_before(seq, i, r, 8)
                k = integer_kind(wr) if wr is not None else None
                if k == "integer":
                    refused.append((addr, fn, w, f"{'LDC' if f3 == 3 else 'STC'} with an integer base x{r} (Q-23)"))
                elif k == "alu":
                    cand.append((addr, fn, w, f"{'LDC' if f3 == 3 else 'STC'} base x{r} written by integer ALU arithmetic "
                                              f"(word {wr:#010x}); its type on the RTL is not in the table"))
        elif op == 0x73 and f3 == 0:
            if w == 0x30200073 or w == 0x10200073:
                e = next(x for x in insn_rows if x.get("std") == "xret")
                cell = e["row"][column]
                ok = (w == 0x10200073 and "SRET ok" in cell) or verdict(cell) == "ok"
                if not ok and column in ("trusted_S",) and w == 0x30200073:
                    refused.append((addr, fn, w, f"MRET: {column} = {cell}"))
                elif not ok and verdict(cell) != "ok" and not (w == 0x10200073 and "SRET ok" in cell):
                    (refused if verdict(cell) == "illegal" else cand).append((addr, fn, w, f"{'MRET' if w == 0x30200073 else 'SRET'}: {column} = {cell}"))
            elif (w >> 25) == 0x09:
                e = next(x for x in insn_rows if x.get("std") == "sfence")
                v = verdict(e["row"][column])
                if v != "ok": (refused if v == "illegal" else cand).append((addr, fn, w, f"SFENCE.VMA: {column} = {e['row'][column]}"))
        elif w == UNIMP:
            traps += 1
        elif op == 0x73 and f3 in (1, 2, 3, 5, 6, 7):
            c = w >> 20
            write = f3 in (1, 5) or ((w >> 15) & 31) != 0
            e = next((x for x in csr_rows if x["lo"] <= c <= x["hi"]), None)
            if e:
                cell = e["row"][column]; v = verdict(cell)
                if v == "illegal": refused.append((addr, fn, w, f"CSR {c:#x} {e['row']['name']}: {column} = {cell}"))
                elif v == "candidate": cand.append((addr, fn, w, f"CSR {c:#x} {e['row']['name']}: {column} = {cell}"))
                elif cell.strip().startswith("RO") and write: refused.append((addr, fn, w, f"CSR {c:#x} {e['row']['name']} is RO, written"))
            else:
                if (c >> 8) & 3 > priv: refused.append((addr, fn, w, f"CSR {c:#x}: privilege {(c >> 8) & 3} above {column} ({priv})"))
                elif (c >> 10) & 3 == 3 and write: refused.append((addr, fn, w, f"CSR {c:#x}: read-only, written"))
    return n, refused, cand, cond, traps


def check_evidence(f, ev):
    """Verify an attribution's evidence against the image itself. Known forms: fini_array_size=0."""
    if ev == "fini_array_size=0":
        ro = os.path.join(os.path.dirname(LLVM), "llvm-readelf")
        out = subprocess.run([ro, "-S", "-W", f], capture_output=True, text=True).stdout
        m = re.search(r"\]\s+\.fini_array\s+\S+\s+([0-9a-f]+)\s+([0-9a-f]+)\s+([0-9a-f]+)", out)
        if not m: return False, "no .fini_array section found"
        size = int(m.group(3), 16)
        return size == 0, f".fini_array size {size:#x} at {m.group(1)}"
    return False, "unknown evidence form"


def main(argv):
    ap = argparse.ArgumentParser(); ap.add_argument("--table", default=str(DEFAULT_TABLE)); ap.add_argument("--column")
    ap.add_argument("--attributions", help="TSV: image sha256/16, address (hex), evidence, reason. A matching candidate "
                                           "is reported as attributed, and does not fail the gate, only if its evidence "
                                           "checks out against the image; a refusal is never attributed")
    ap.add_argument("files", nargs="*"); a = ap.parse_args(argv)
    if not os.path.exists(a.table): print(f"NO DATA: no table at {a.table}"); return 2
    insn_rows, csr_rows = load_table(a.table)
    attr = {}
    if a.attributions:
        for l in open(a.attributions):
            if l.startswith("#") or not l.strip(): continue
            sha, ad, ev, why = l.rstrip("\n").split("\t", 3)
            attr[(sha, int(ad, 16))] = (ev, why)
    if not a.files: print(__doc__); return 2
    worst = 0
    for f in a.files:
        col = a.column or ("trusted_S" if f.endswith(".ko") else "in_context")
        od = CROSS if f.endswith(".ko") else LLVM
        if not os.access(od, os.X_OK): print(f"NO DATA: no objdump at {od}"); return 2
        args = [od, "-d", f] if od == CROSS else [od, "-d", "--triple=capstone64-unknown-elf", f]
        r = subprocess.run(args, capture_output=True, text=True)
        n, refused, cand, cond, traps = gate(r.stdout.splitlines(), insn_rows, csr_rows, col)
        sha = subprocess.run(["sha256sum", f], capture_output=True, text=True).stdout[:16]
        if n == 0: print(f"NO DATA {f} ({sha}): 0 instructions disassembled"); worst = 2; continue
        attributed, keep = [], []
        for ad, fn, w, why in cand:
            row = attr.get((sha, ad))
            ok, note = (False, "") if row is None else check_evidence(f, row[0])
            if ok: attributed.append((ad, fn, w, why, f"{row[1]} [evidence {row[0]}: {note}]"))
            else:
                keep.append((ad, fn, w, why + (f" (attribution REJECTED: evidence {row[0]}: {note})" if row else "")))
        cand = keep
        groups = {}
        for addr, fn, w, why in refused: groups.setdefault(why, []).append((addr, fn))
        status = "FAIL" if refused else ("CAND" if cand else "ok  ")
        print(f"{status} {f} ({sha}) as {col}: {len(refused)} refused, {len(cand)} candidates, {len(attributed)} attributed, {cond} conditional, {traps} unimp, {n} instructions")
        for why, sites in sorted(groups.items(), key=lambda x: -len(x[1])):
            fns = sorted({fn or '?' for _, fn in sites})
            print(f"       {len(sites):6} x {why}  [{', '.join(fns[:4])}{', ...' if len(fns) > 4 else ''}] first {sites[0][0]:#x}")
        for addr, fn, w, why in cand[:5]:
            print(f"       candidate {addr:#x} {fn}: {why}")
        for addr, fn, w, why, because in attributed:
            print(f"       attributed {addr:#x} {fn}: {why} -- {because}")
        if refused or cand: worst = max(worst, 1)
    return worst


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
