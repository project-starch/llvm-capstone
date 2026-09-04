#!/usr/bin/env python3
"""Minimise S-12 by removing INSTRUCTIONS from the real faulting binary.

Not a reconstruction and not a source truncation. Both of those have now failed for the same
reason: they regenerate the function, so every address, allocation and instruction count moves at
once and the result is a different program that has to be argued about. Twelve hand-written
reconstructions came back clean, and truncating the source made SQLite return a garbage Bitmask
that killed the run in memcpy long before the site under test.

This patches the ACTUAL binary that faults, replacing chosen instructions with NOP
(addi x0,x0,0 = 0x00000013) and leaving every other byte, address and offset identical. What
survives removal is not needed; what cannot be removed is the repro.

The window, entry to fault, is 36 instructions and branch-free -- the same path on every call:

    [14] cincoffsetimm a0, s0, -0x70    a0 = the spill slot
    [16] stc  a2, 0x0(a0)               store the incoming pWInfo into it
    [34] ldc  a4, 0x0(a0)               reload it
    [35] cincoffsetimm a4, a4, 0xb0     FAULTS, mcause 25, tval 0

[34] and [35] are the fault itself and are refused as targets. [0]-[4] establish the frame. The
other 27 are candidates, and bisecting them costs ~log2(27) rounds rather than 27.

This also answers, directly and without statistics, the question the arm comparisons could not:
NOP [32] and [33] -- the `movc a4,zero` / `stc a4,0(a5)` pair -- and if the fault persists, the
pair was never required. The whole register-match correlation, which has been retracted twice and
re-derived twice, is one boot away from being settled by deletion instead of by p-values.

USAGE
    s12-minimize.py <in.dom> <out.dom> --nop 5,6,7,8      remove instructions by index
    s12-minimize.py <in.dom> --list                       print the window with indices

The anchor is the 20-byte sequence around the fault, required to occur exactly once, so a drifted
offset refuses rather than patching the wrong place.
"""
import argparse
import hashlib
import re
import subprocess
import sys

OBJDUMP = "/home/alexey/dev/llvm-capstone/llvm/cmake-build-debug/bin/llvm-objdump"
TRIPLE = "capstone64-unknown-elf"
FUNC = "sqlite3WhereCodeOneLoopStart"
NOP = bytes.fromhex("13000000")          # addi x0, x0, 0, little-endian
FAULT_VA = 0x104814
# The base is pinned by CONTENT, not by path. A second build on this disk also contains the
# anchor, and the whole campaign is defined by byte-identity to this exact image; a run patched
# from the wrong base would be internally consistent and completely wrong. Durable copy lives at
# ~/fpga-artifacts/s12-base.dom because /tmp does not survive.
BASE_SHA256 = "69fe70b767e76b2ba34cca2ec160f3567e1a641ded7baa639803220087f9ca20"
# cincoffsetimm a5,s0,-0x120 / movc a4,zero / stc a4,0(a5) / ldc a4,0(a0) / cincoffsetimm a4,a4,0xb0
ANCHOR = bytes.fromhex("db2704ee" "5b170014" "5bc0e700" "5b370500" "5b27070b")


def window(path):
    """[(index, va, text)] from function entry through the faulting instruction."""
    out = subprocess.run([OBJDUMP, "-d", f"--triple={TRIPLE}", path],
                         capture_output=True, text=True, timeout=600)
    if out.returncode != 0 or not out.stdout.strip():
        sys.exit(f"objdump produced nothing for {path}")
    started, rows = False, []
    for line in out.stdout.splitlines():
        if re.match(rf"^([0-9a-f]+)\s+<{FUNC}>:", line):
            started = True
            continue
        if not started:
            continue
        m = re.match(r"\s+([0-9a-f]+):\s+((?:[0-9a-f]{2} ){4})\s*(.*)", line)
        if not m:
            continue
        va = int(m.group(1), 16)
        rows.append((len(rows), va, m.group(3).strip()))
        if va == FAULT_VA:
            return rows
    sys.exit(f"fault site 0x{FAULT_VA:x} not reached in {FUNC}")


def whole_function(path):
    """Every instruction in the function, not just the window.

    Needed because a deletion inside the window is USED OUTSIDE it. Cluster D removed
    `[23] cincoffsetimm a2, s0, -0x98`, and `ld a2, 0x0(a2)` twenty-four instructions later then
    loaded through the incoming argument instead: mcause 4, load address misaligned,
    deterministically, on two board draws. The in-window closure check passed it. The window is
    not a closed unit and treating it as one costs boots.
    """
    out = subprocess.run([OBJDUMP, "-d", f"--triple={TRIPLE}", path],
                         capture_output=True, text=True, timeout=600)
    started, rows = False, []
    for line in out.stdout.splitlines():
        if re.match(rf"^([0-9a-f]+)\s+<{FUNC}>:", line):
            started = True
            continue
        # Stop only at a REAL function symbol. `.Lpcrel_hi8047` sits at 0x107450, INSIDE this
        # function -- breaking on it truncated the analysis at 2866 of the function's 4600
        # instructions, so a use in the last 38% was invisible to the closure gate. That is the
        # same defect as the window-vs-function one, one level out.
        m_sym = re.match(r"^[0-9a-f]+\s+<(\S+)>:", line)
        if started and m_sym and not m_sym.group(1).startswith(".L"):
            break
        if not started:
            continue
        m = re.match(r"\s+([0-9a-f]+):\s+((?:[0-9a-f]{2} ){4})\s*(.*)", line)
        if m:
            rows.append((len(rows), int(m.group(1), 16), m.group(3).strip()))
    return rows


def def_use(text):
    """(defined_register or None, [used registers]) for one disassembled instruction.

    Only cincoffsetimm / cincoffset / movc / ldc DEFINE. Stores do not: `stc a0, -0x5e0(s0)`
    STORES a0, it does not write it -- getting that backwards is easy and produces a dependency
    graph that is wrong in the direction that hides wild stores.
    """
    t = text.strip()
    if t.startswith("nop"):
        return None, []
    m = re.match(r"(cincoffsetimm|movc)\s+(\w+),\s*(\w+)", t)
    if m:
        return m.group(2), [m.group(3)]
    m = re.match(r"cincoffset\s+(\w+),\s*(\w+),\s*(\w+)", t)
    if m:
        return m.group(1), [m.group(2), m.group(3)]
    m = re.match(r"ldc?\s+(\w+),\s*-?0x[0-9a-f]+\((\w+)\)", t)
    if m:
        return m.group(1), [m.group(2)]
    m = re.match(r"(?:stc|sd|sw|sh|sb)\s+(\w+),\s*-?0x[0-9a-f]+\((\w+)\)", t)
    if m:
        return None, [m.group(1), m.group(2)]
    return None, []


# Callee-saved spills. NOPing these does not corrupt data, it corrupts the RETURN PATH: the
# function restores them on exit, and with the store gone the caller resumes with this function's
# values. [1] `sd ra` was already known to run away (exposure 41,700,295 -- it destroys the return
# address). [2] `stc s0, 0x7d0(sp)` is the same class and was NOT refused: it went to the board in
# the EJF cut and all three draws faulted deterministically outside the function.
CALLEE_SAVE = {1, 2}


def frame_slots(rows):
    """[(index, 'store'|'load', slot_offset)] for frame-relative accesses through s0.

    The register closure gate models REGISTERS. Every instruction in this window is the
    initialisation of a LOCAL, and a local is consumed through MEMORY -- so the gate could not see
    the dependency that actually matters. Removing `[27] stc a4(=null), -0x5a0(s0)` leaves that
    capability slot holding whatever the frame held; an `ldc` of it later in the function then
    reads an untagged word and faults with mcause 25 at a wrong address, which is indistinguishable
    from a result until you read the address.

    Offsets are tracked only where they are UNAMBIGUOUS: a base register defined by
    `cincoffsetimm rX, s0, OFF`, or s0 itself. Anything else drops the register from tracking. The
    scan is linear and ignores control flow, which over-approximates USES -- the safe direction,
    since an over-approximation only refuses more cuts.
    """
    # SEEDING s0 IS THE WHOLE GAME. The prologue is
    #     [3] movc s0, sp   [4] cincoffsetimm s0, s0, 0x7f0
    # so a naive scan starting from {"s0": 0} has s0 REDEFINED out of the tracked set by [3]
    # (`sp` is not tracked) and never recovers -- it then follows nothing and reports zero
    # accesses across a 4600-instruction function, which reads exactly like "this cut is safe".
    # It did, for E, F and G. So: assert the prologue shape, skip it, and start with s0 as the
    # frame base.
    if len(rows) < 5 or "movc" not in rows[3][2] or "cincoffsetimm" not in rows[4][2]:
        raise SystemExit("frame_slots: prologue is not the expected movc/cincoffsetimm pair; "
                         "the s0 seed would be wrong and every slot check silently vacuous")
    known = {"s0": 0}
    acc = []
    for i, _va, txt in rows[5:]:
        t = txt.strip()
        m = re.match(r"cincoffsetimm\s+(\w+),\s*(\w+),\s*(-?0x[0-9a-f]+|-?\d+)$", t)
        if m:
            d, b, off = m.group(1), m.group(2), int(m.group(3), 16 if "x" in m.group(3) else 10)
            if b in known:
                known[d] = known[b] + off
            else:
                known.pop(d, None)
            continue
        m = re.match(r"movc\s+(\w+),\s*(\w+)$", t)
        if m:
            d, b = m.group(1), m.group(2)
            if b in known:
                known[d] = known[b]
            else:
                known.pop(d, None)
            continue
        m = re.match(r"(stc|sd|sw|sh|sb|ldc|ld|lw|lwu|lh|lhu|lb|lbu)\s+(\w+),\s*(-?0x[0-9a-f]+|-?\d+)\((\w+)\)", t)
        if m:
            op, reg, imm, base = m.group(1), m.group(2), m.group(3), m.group(4)
            imm = int(imm, 16 if "x" in imm else 10)
            if base in known:
                acc.append((i, "store" if op[0] == "s" else "load", known[base] + imm))
            if op[0] == "l":              # a load DEFINES its destination
                known.pop(reg, None)
            continue
        # any other instruction: whatever it writes, we no longer know. Conservatively forget the
        # first operand if it looks like a destination register.
        m = re.match(r"\w[\w.]*\s+(\w+),", t)
        if m:
            known.pop(m.group(1), None)
    if not acc:
        raise SystemExit("frame_slots: tracked ZERO frame-relative accesses in a function that "
                         "is built entirely out of them. That is an instrument failure, not a "
                         "clean cut -- refusing to report it as one.")
    return acc


def slot_violations(rows, nopped):
    """NOPed stores whose frame slot is READ later with no surviving store in between."""
    acc = frame_slots(rows)
    bad = []
    for k, (i, kind, slot) in enumerate(acc):
        if kind != "store" or i not in nopped:
            continue
        for j, kind2, slot2 in acc[k + 1:]:
            if slot2 != slot:
                continue
            if kind2 == "store" and j not in nopped:
                break                      # re-initialised by surviving code; harmless
            if kind2 == "load":
                bad.append((i, slot, j, rows[j][2]))
                break
    return bad


def closure_violations(rows, nopped):
    """Surviving instructions whose operand's only in-window definition was NOPed.

    NOPing a def while keeping its use does not merely change data: the register keeps whatever
    it held on entry, and a store through it becomes a WILD STORE -- a trap INSIDE the window,
    which destroys the reachability argument that makes every verdict in this campaign readable.
    A cut must be def-use closed or it is not a cut, it is a different bug.
    """
    bad = []
    for i, va, txt in rows:
        if i in nopped:
            continue
        _, uses = def_use(txt)
        for u in uses:
            # the nearest in-window definition of `u` strictly before i
            src = None
            for j, _vaj, tj in rows[:i]:
                d, _ = def_use(tj)
                if d == u:
                    src = j
            if src is not None and src in nopped:
                bad.append((i, txt, u, src))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst", nargs="?")
    ap.add_argument("--nop", default="")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--allow-open", action="store_true",
                    help="permit a cut that is not def-use closed")
    ap.add_argument("--allow-any-base", action="store_true",
                    help="permit a base other than the pinned sha256")
    a = ap.parse_args()

    # BASE IDENTITY FIRST, before anything that depends on the window. Ordered this way because
    # the truncated builds have a 17-instruction window, so an index check running first reports
    # "index out of range 0..16" for what is actually the wrong base -- a confusing error that
    # hides the informative one. The most diagnostic check goes earliest.
    if not a.list:
        src_sha = hashlib.sha256(open(a.src, "rb").read()).hexdigest()
        if src_sha != BASE_SHA256 and not a.allow_any_base:
            print(f"REFUSING: {a.src} is sha256 {src_sha[:16]}..., not the pinned base "
                  f"{BASE_SHA256[:16]}...\n  Another build on this disk also contains the anchor; "
                  f"patching it would be internally consistent and wrong.\n"
                  f"  Durable base: ~/fpga-artifacts/s12-base.dom   "
                  f"(--allow-any-base to override)", file=sys.stderr)
            return 2

    rows = window(a.src)
    if a.list:
        for i, va, txt in rows:
            tag = ""
            if i in (34, 35):
                tag = "   <- the fault itself, REFUSED as a target"
            elif i <= 4:
                tag = "   <- frame setup"
            print(f"  [{i:2d}] 0x{va:x}  {txt}{tag}")
        return 0

    if not a.dst:
        sys.exit("need <out.dom> unless --list")
    idx = sorted({int(x) for x in a.nop.split(",") if x.strip() != ""})
    if not idx:
        sys.exit("--nop needs at least one index")
    bad = [i for i in idx if i in (34, 35)]
    if bad:
        sys.exit(f"REFUSING: {bad} is the faulting pair itself; removing it removes the bug, "
                 f"not the cause")
    cs = [i for i in idx if i in CALLEE_SAVE]
    if cs and not a.allow_open:
        sys.exit(f"REFUSING: {cs} is a callee-saved spill. Removing it corrupts the RETURN PATH, "
                 f"not the data -- the caller resumes with this function's register values and "
                 f"faults somewhere unrelated. Verified in the disassembly: [1] is restored by "
                 f"[4596] `ld ra, 0x7e0(sp)` and [2] by [4597] `ldc s0, 0x7d0(sp)`, immediately "
                 f"before the function's single `ret` at [4599].")
    if any(i < 0 or i >= len(rows) for i in idx):
        sys.exit(f"REFUSING: index out of range 0..{len(rows)-1}")

    # Closure over the WHOLE FUNCTION, not the window. The window is a prefix of the function so
    # indices mean the same instruction in both.
    fn = whole_function(a.src)
    sviol = slot_violations(fn, set(idx))
    if sviol and not a.allow_open:
        print("REFUSING: the cut is not FRAME-SLOT closed. A removed store leaves a local "
              "uninitialised and it is read later in the function:", file=sys.stderr)
        for i, slot, j, txt in sviol:
            print(f"    [{i:2d}] writes s0{slot:+#x}, read back by [{j}] {txt}", file=sys.stderr)
        print("  The reader then gets whatever the frame held. An `ldc` of a stale slot is "
              "untagged and traps with mcause 25 at an address that is NOT the fault site -- "
              "which reads like a result and is not one.", file=sys.stderr)
        return 2

    viol = closure_violations(fn, set(idx))
    if viol and not a.allow_open:
        print("REFUSING: the cut is not def-use closed. Surviving instructions read a register "
              "whose only in-window definition was removed:", file=sys.stderr)
        for i, txt, reg, src in viol:
            print(f"    [{i:2d}] {txt}   reads {reg}, defined only by NOPed [{src}]",
                  file=sys.stderr)
        print("  That leaves the register holding whatever it had on entry; a store through it "
              "is a WILD STORE that traps inside the window and makes the verdict unreadable.\n"
              "  Add the definer to the cut, or pass --allow-open if you mean it.", file=sys.stderr)
        return 2

    blob = bytearray(open(a.src, "rb").read())
    hits = []
    p = blob.find(ANCHOR)
    while p != -1:
        hits.append(p)
        p = blob.find(ANCHOR, p + 1)
    if len(hits) != 1:
        sys.exit(f"REFUSING: anchor occurs {len(hits)} times, need exactly 1")
    # The anchor starts at instruction [31]; derive the file offset of [0] from it rather than
    # trusting any recorded constant.
    base = hits[0] - 31 * 4

    for i in idx:
        off = base + i * 4
        blob[off:off + 4] = NOP
    changed = sum(x != y for x, y in zip(open(a.src, "rb").read(), blob))
    if changed > 4 * len(idx):
        sys.exit(f"REFUSING: {changed} bytes differ, expected at most {4*len(idx)}")

    open(a.dst, "wb").write(blob)
    print(f"NOPed {len(idx)} instruction(s) {idx} at file offset 0x{base:x}; "
          f"{changed} bytes changed")
    print(f"  in  {hashlib.sha256(open(a.src,'rb').read()).hexdigest()[:16]}")
    print(f"  out {hashlib.sha256(blob).hexdigest()[:16]}")

    after = window(a.dst)
    if after[35][2] != rows[35][2] or after[34][2] != rows[34][2]:
        sys.exit("REFUSING: the fault pair changed; the patch landed in the wrong place")
    print(f"  fault pair intact: [34] {after[34][2]} / [35] {after[35][2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
