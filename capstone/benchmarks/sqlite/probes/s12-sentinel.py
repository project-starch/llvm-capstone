#!/usr/bin/env python3
"""Build the SENTINEL arm: make the stale pre-`ldc` value of a4 a self-labelling constant.

THE HYPOTHESIS THIS TESTS. The RTL admits a path where the consumer of an in-flight `ldc` issues
WITHOUT the RAW hazard being detected, neither stalling nor forwarding, and therefore reads a4
straight out of the register file -- i.e. the value a4 held BEFORE the load. In the faulting window
that value is written by

    [32] movc a4, zero      ->  a4 = {cursor 0, NOT_CAP}

two instructions before the load, and `cincoffsetimm` raises UNEXPECTED_OPERAND on NOT_CAP with the
cursor reported in tval. That is mcause 25 with tval 0: the observed signature, bit for bit, and
`issue_read_operands.sv:1264-1266` says so in the tree in those words.

The problem with confirming it from the existing evidence is that tval 0 is ALSO what you see if
operand delivery simply fails and presents zero on both halves. The two are indistinguishable while
the stale value IS zero. So change the stale value.

    [26] movc a4, zero        ->  movc t0, zero
    [27] stc  a4, -0x5a0(s0)  ->  stc  t0, -0x5a0(s0)
    [28] sw   a4, 0x0(a5)     ->  sw   t0, 0x0(a5)
    [30] sw   a4, 0x0(a5)     ->  sw   t0, 0x0(a5)
    [32] movc a4, zero        ->  addi a4, zero, SENTINEL
    [33] stc  a4, 0x0(a5)     ->  stc  t0, 0x0(a5)

Every null the program stored is still stored, to the same slot, with the same value -- t0 now
carries it. The ONLY change is that a4, which nothing reads between [32] and the load, holds a
recognisable constant instead of zero. `addi` is used for the sentinel because
`cincoffset-stale-metadata.S` refuted metadata leakage for the lui/addi pair, so a4 ends up
genuinely {cursor SENTINEL, NOT_CAP} rather than a stale-tagged value.

t0 is the scratch register because it appears NOWHERE in this function -- verified over all 4600
instructions, not over the window -- so nothing observes the substitution.

PRE-REGISTERED READING. Write it down before the board, because three of the four outcomes
discriminate and one does not:

  mcause 25, tval == SENTINEL   the consumer read the stale register file. CONFIRMED, and decisive
                                at n=1: no other mechanism can put this constant in tval.
  mcause 25, tval == 0          REFUTES the stale-read account. Nothing writes 0 to a4 any more, so
                                a zero operand must be manufactured by the delivery path itself.
  mcause 25, tval == an address the operand came from somewhere else entirely -- e.g. writeback-port
                                displacement, which forwards the real cursor with the tag dropped.
  no fault                      weakest outcome. The perturbation cured it; uninformative alone,
                                and must NOT be read as support for anything.
"""
import argparse, hashlib, re, subprocess, sys

OBJDUMP = "/home/alexey/dev/llvm-capstone/llvm/cmake-build-debug/bin/llvm-objdump"
TRIPLE = "capstone64-unknown-elf"
FUNC = "sqlite3WhereCodeOneLoopStart"
BASE_SHA256 = "69fe70b767e76b2ba34cca2ec160f3567e1a641ded7baa639803220087f9ca20"
ANCHOR = bytes.fromhex("db2704ee" "5b170014" "5bc0e700" "5b370500" "5b27070b")
A4, T0 = 14, 5
SCRATCH_NAME = "t0"


def disas(path):
    out = subprocess.run([OBJDUMP, "-d", f"--triple={TRIPLE}", path],
                         capture_output=True, text=True, timeout=600).stdout
    started, rows = False, []
    for line in out.splitlines():
        if re.match(rf"^([0-9a-f]+) <{FUNC}>:", line):
            started = True
            continue
        m2 = re.match(r"^[0-9a-f]+ <(\S+)>:", line)
        if started and m2 and not m2.group(1).startswith(".L"):
            break
        if not started:
            continue
        m = re.match(r"\s+([0-9a-f]+):\s+((?:[0-9a-f]{2} ){4})\s*(.*)", line)
        if m:
            rows.append((len(rows), int(m.group(1), 16), m.group(3).strip()))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--sentinel", default="0x5a5")
    ap.add_argument("--value-only", action="store_true",
                    help="THE ARM THAT CAN ACTUALLY TEST THE VALUE ACCOUNT. Base with ONLY [32] "
                         "`movc a4, zero` -> `li a4, 0x5a5`; [33] `stc a4, 0x0(a5)` and all of "
                         "[26] [27] [28] [30] are left alone. One instruction from a baseline "
                         "attested at 5/5 wedges. The earlier sentinel arm carried the t0 "
                         "substitution as well, and that substitution by itself suppresses the "
                         "wedge -- so its clean result was predicted by the cure it inherited and "
                         "said nothing about a4's value. THIS arm keeps the wedging configuration "
                         "and changes only the value. WARNING: [33] will now store an untagged "
                         "0x5a5 where the program stored a null capability, into a slot read 14 "
                         "times later, so it may well break the program -- gate it under QEMU "
                         "before boarding, and if it cannot be gated the value account stays OPEN "
                         "rather than becoming refuted.")
    ap.add_argument("--only28", action="store_true",
                    help="ONE instruction: [28] `sw a4, 0x0(a5)` -> `movc t0, zero`, with [33] "
                         "`stc a4, 0x0(a5)` LEFT ALONE. Exists to finish the attribution: --tight "
                         "changes [28] AND [33], so a clean --tight cannot say which of the two "
                         "matters. This arm differs from --tight in [33] ALONE, so the pair is a "
                         "true one-instruction contrast against a baseline attested at 5/5 "
                         "wedges. If this wedges and --tight does not, the store at [33] reading "
                         "a4 is the cause; if this is clean too, [28] alone suffices and [33] is "
                         "not implicated.")
    ap.add_argument("--tight", action="store_true",
                    help="TIGHT CONTROL for the store-register question, TWO instructions instead "
                         "of five. [28] `sw a4, 0x0(a5)` becomes `movc t0, zero` and [33] "
                         "`stc a4` becomes `stc t0`; [26], [27] and [30] stay on a4. The full t0 "
                         "substitution changes five instructions, so a difference between it and "
                         "the base cannot be attributed to the store-register match rather than "
                         "to any of the other four. [28] is available as the scratch slot because "
                         "the QEMU functional gate already showed removing its zero-init is "
                         "behaviour-preserving -- its slot s0-0x10c is re-stored at [755].")
    ap.add_argument("--control", action="store_true",
                    help="MATCHED CONTROL: do everything except the sentinel. [32] stays "
                         "`movc a4, zero`, so a4 enters the load holding zero exactly as in the "
                         "base, while the layout, the instruction count, the register pressure "
                         "and the t0 substitution are all identical to the sentinel arm. The two "
                         "then differ by EXACTLY ONE THING -- the value a4 holds -- which is what "
                         "makes a difference between them attributable. Without it, a sentinel "
                         "arm that stops wedging is indistinguishable from plain perturbation, "
                         "and the register-patch arm already cured 0/4 that way.")
    a = ap.parse_args()

    sent = int(a.sentinel, 0)
    if not -2048 <= sent <= 2047:
        sys.exit(f"REFUSING: {sent} does not fit addi's 12-bit signed immediate")

    blob = bytearray(open(a.src, "rb").read())
    if hashlib.sha256(bytes(blob)).hexdigest() != BASE_SHA256:
        sys.exit(f"REFUSING: {a.src} is not the pinned base")

    rows = disas(a.src)
    # t0 must be invisible to this function, over the WHOLE function. Checking only the window is
    # how the frame-slot defect got in.
    live = [(i, t) for i, _v, t in rows if re.search(rf"\b{SCRATCH_NAME}\b", t)]
    if live:
        sys.exit(f"REFUSING: {SCRATCH_NAME} is referenced {len(live)} time(s) in {FUNC}, first at "
                 f"[{live[0][0]}] {live[0][1]}. Substituting into it would change the program.")
    print(f"{SCRATCH_NAME} is referenced 0 times in {FUNC}'s {len(rows)} instructions -- free")

    hits, p = [], blob.find(ANCHOR)
    while p != -1:
        hits.append(p)
        p = blob.find(ANCHOR, p + 1)
    if len(hits) != 1:
        sys.exit(f"REFUSING: anchor occurs {len(hits)} times, need exactly 1")
    base = hits[0] - 31 * 4

    def word(i):
        return int.from_bytes(blob[base + i * 4:base + i * 4 + 4], "little")

    def put(i, w):
        blob[base + i * 4:base + i * 4 + 4] = w.to_bytes(4, "little")

    # Verify each target is the instruction expected, by DECODED FIELDS, before touching it.
    for i, kind in ((26, "rd"), (27, "rs2"), (28, "rs2"), (30, "rs2"), (32, "rd"), (33, "rs2")):
        w = word(i)
        got = (w >> 7) & 0x1f if kind == "rd" else (w >> 20) & 0x1f
        if got != A4:
            sys.exit(f"REFUSING: [{i}] {rows[i][2]!r} has {kind}=x{got}, expected a4 (x{A4}). "
                     f"The window has drifted; patching it would hit the wrong operand.")

    if a.only28:
        w26 = word(26)
        put(28, (w26 & ~(0x1f << 7)) | (T0 << 7))       # [28] -> movc t0, zero; [33] untouched
        blob4 = bytes(blob)
        open(a.dst, "wb").write(blob4)
        after = disas(a.dst)
        print(f"  ONLY-28 arm, 1 instruction changed")
        print(f"  out sha256 {hashlib.sha256(blob4).hexdigest()[:16]}")
        for i in (26, 27, 28, 30, 32, 33, 34, 35):
            print(f"  [{i:2d}] {rows[i][2]:34s} ->  {after[i][2]}")
        if after[33][2] != rows[33][2]:
            sys.exit("REFUSING: [33] changed; this arm exists to leave it alone")
        if after[34][2] != rows[34][2] or after[35][2] != rows[35][2]:
            sys.exit("REFUSING: the fault pair changed")
        print("  [33] and the fault pair intact")
        return 0

    if a.value_only:
        put(32, ((sent & 0xfff) << 20) | (0 << 15) | (0 << 12) | (A4 << 7) | 0x13)
        blob3 = bytes(blob)
        open(a.dst, "wb").write(blob3)
        after = disas(a.dst)
        print(f"  VALUE-ONLY arm, 1 instruction changed")
        print(f"  out sha256 {hashlib.sha256(blob3).hexdigest()[:16]}")
        for i in (26, 27, 28, 30, 32, 33, 34, 35):
            print(f"  [{i:2d}] {rows[i][2]:34s} ->  {after[i][2]}")
        if after[34][2] != rows[34][2] or after[35][2] != rows[35][2]:
            sys.exit("REFUSING: the fault pair changed")
        print("  fault pair intact")
        return 0

    if a.tight:
        w26 = word(26)                       # `movc a4, zero`, borrowed for its encoding
        put(28, (w26 & ~(0x1f << 7)) | (T0 << 7))       # [28] -> movc t0, zero
        w33 = word(33)
        put(33, (w33 & ~(0x1f << 20)) | (T0 << 20))     # [33] -> stc t0, 0x0(a5)
        blob2 = bytes(blob)
        open(a.dst, "wb").write(blob2)
        after = disas(a.dst)
        print(f"  TIGHT control, 2 instructions changed")
        print(f"  out sha256 {hashlib.sha256(blob2).hexdigest()[:16]}")
        for i in (26, 27, 28, 30, 32, 33, 34, 35):
            print(f"  [{i:2d}] {rows[i][2]:34s} ->  {after[i][2]}")
        if after[34][2] != rows[34][2] or after[35][2] != rows[35][2]:
            sys.exit("REFUSING: the fault pair changed")
        print("  fault pair intact")
        return 0

    for i, kind in ((26, "rd"), (27, "rs2"), (28, "rs2"), (30, "rs2"), (33, "rs2")):
        w = word(i)
        if kind == "rd":
            put(i, (w & ~(0x1f << 7)) | (T0 << 7))
        else:
            put(i, (w & ~(0x1f << 20)) | (T0 << 20))
    if a.control:
        print("  MATCHED CONTROL: [32] left as `movc a4, zero`; a4 enters the load holding zero")
    else:
        # [32] -> `addi a4, zero, SENTINEL`: imm[11:0] | rs1=x0 | funct3=0 | rd=a4 | opcode 0x13
        put(32, ((sent & 0xfff) << 20) | (0 << 15) | (0 << 12) | (A4 << 7) | 0x13)

    open(a.dst, "wb").write(blob)
    after = disas(a.dst)
    print(f"  out sha256 {hashlib.sha256(bytes(blob)).hexdigest()[:16]}")
    for i in (26, 27, 28, 30, 32, 33, 34, 35):
        print(f"  [{i:2d}] {rows[i][2]:34s} ->  {after[i][2]}")
    if after[34][2] != rows[34][2] or after[35][2] != rows[35][2]:
        sys.exit("REFUSING: the fault pair changed")
    print("  fault pair intact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
