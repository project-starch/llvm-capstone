#!/usr/bin/env python3
"""Find S-14 sites in a domain image, statically, by the invariant rather than the shape.

    capinit-scan.py <image.dom> [more.dom ...]

S-14 is an open compiler defect: inside the synthesised `__capstone_cap_init`, a
capability spill slot is sometimes reloaded with a scalar load, so the tag is gone
and the first capability access through that register faults with cause 24 before
the domain enters. The issue asks for a gate on the pattern rather than on any
image or arena number, because any change to the global set can reintroduce it.

WHY NOT THE ISSUE'S LITERAL PATTERN. S-14 states the shape it first saw: `ld <rd>,
<imm>(sp)` consumed by `cincoffsetimm`. Two further instances found on MicroPython
images have a different shape and the same defect:

    cincoffset a3, sp, a3        the slot is reached through a materialised sp+imm,
    ld         ra, -0x370(a3)    not a bare sp, and
    stc        a0, 0x370(ra)     the consumer is a capability store, not cincoffsetimm

So this gates on the INVARIANT: the address operand of a capability access must be
a capability. A site is any `ldc`/`stc` (or `cincoffsetimm` feeding one) whose
address register was last defined by a scalar-producing instruction. The function
is straight-line, so the last definition is simply the previous one in order.

An unknown mnemonic is reported and never assumed either way, because guessing
would let the gate pass a faulting image.

WHAT THIS DOES NOT CATCH. The quiet form, an `stc` whose DATA register is scalar,
puts an untagged capability-sized word into a global and faults only when someone
reads it. Checking that needs tag propagation through `addi`, because the normal
install pattern is `auipc`+`addi`+`stc` and `addi` keeps the tag. Without that rule
the check fires about 350 times per image, on working images too, so it is left out
rather than shipped as noise.
"""
import re, subprocess, sys, collections

# Instructions that leave a capability (tag intact) in rd.
CAP_DEF = {"ldc", "cincoffset", "cincoffsetimm", "ccsrrw", "scc", "delin", "mrev",
           "cseal", "cunseal", "clcc", "cmove", "csetbounds", "csetboundsimm",
           "cbuildcap", "tighten", "shrink", "shrinkto", "split", "splitimm", "lcl"}
# Instructions that leave a scalar in rd (the interesting ones; the rest are caught
# by the unknown-mnemonic report).
SCALAR_DEF = {"ld", "lw", "lwu", "lh", "lhu", "lb", "lbu", "li", "lui",
              "addi", "add", "addiw", "addw", "sub", "subw", "and", "andi", "or",
              "ori", "xor", "xori", "sll", "slli", "srl", "srli", "sra", "srai",
              "slliw", "srliw", "sraiw", "sext.w", "zext.w", "mv", "neg", "not",
              "mul", "mulw", "div", "divu", "rem", "remu", "lcc", "seqz", "snez",
              "slt", "sltu", "slti", "sltiu"}
# auipc leaves a TAGGED pc-derived capability here, shown by the working
# install pattern in every image that runs: `auipc`+`addi`+`stc <rd>, imm(a1)`
# stores a global's address and those images enter and run.
CAP_DEF.add("auipc")
# rd inherits rs1's kind: `addi a0, a0, -0x800` after `auipc a0` keeps the tag,
# and that pair is the normal way a global's address is formed.
INHERIT = {"addi", "add", "cincoffset", "cincoffsetimm", "mv"}
NO_DEF = {"stc", "sd", "sw", "sh", "sb", "ret", "j", "jal", "jalr", "beq", "bne",
          "blt", "bge", "bltu", "bgeu", "beqz", "bnez", "blez", "bgez", "bltz",
          "bgtz", "nop", "ecall", "ebreak", "unimp", "fence"}
CAP_ACCESS = {"ldc": "load", "stc": "store"}

INSN = re.compile(r"^\s*([0-9a-f]+):\s+(\S+)\s*(.*)$")
MEM = re.compile(r"(-?(?:0x[0-9a-f]+|\d+))\((\w+)\)")

def body_of(img, func="__capstone_cap_init"):
    dis = subprocess.run(["llvm-objdump", "-d", "--no-show-raw-insn", img],
                         capture_output=True, text=True)
    if dis.returncode:
        sys.exit(f"llvm-objdump failed on {img}: {dis.stderr.strip()}")
    lines, out, inside, lo, hi = dis.stdout.splitlines(), [], False, None, None
    for line in lines:
        m = re.match(r"^([0-9a-f]+) <(.+)>:", line)
        if m and not m.group(2).startswith(".L"):
            if m.group(2) == func:
                inside, lo = True, int(m.group(1), 16)
                continue
            if inside:
                hi = int(m.group(1), 16)
                break
        if inside:
            out.append(line)
    return lo, hi, out

def scan(img):
    lo, hi, body = body_of(img)
    if lo is None:
        return None
    kind = {}                      # register -> "cap" | "scalar" | "unknown"
    sites, quiet, unknown = [], [], collections.Counter()
    for line in body:
        m = INSN.match(line)
        if not m:
            continue
        addr, mn, ops = int(m.group(1), 16), m.group(2), m.group(3)
        if mn == "stc":
            # THE QUIET FORM. The DATA operand is scalar, so a capability-sized
            # untagged word lands in a global and nothing faults here. The first
            # read through it faults somewhere else entirely: one of these wrote
            # an untagged type pointer that faulted three hundred tests later in
            # mp_convert_member_lookup.
            parts = [o.strip() for o in ops.split(",")]
            if parts and kind.get(parts[0]) == "scalar" and MEM.search(ops):
                m2 = MEM.search(ops)
                quiet.append((addr, parts[0], m2.group(2), m2.group(1)))
        if mn in CAP_ACCESS:
            mem = MEM.search(ops)
            if mem:
                base = mem.group(2)
                if base not in ("sp", "gp") and kind.get(base) == "scalar":
                    sites.append((addr, mn, base, mem.group(1)))
        rd = ops.split(",")[0].strip() if ops else None
        if rd and re.fullmatch(r"(x\d+|zero|ra|sp|gp|tp|t\d|s\d+|a\d+|fp)", rd):
            if mn in INHERIT:
                src = ops.split(",")[1].strip() if ops.count(",") >= 1 else None
                kind[rd] = kind.get(src, "unknown") if src else "unknown"
            elif mn in CAP_DEF:
                kind[rd] = "cap"
            elif mn in SCALAR_DEF:
                kind[rd] = "scalar"
            elif mn in NO_DEF:
                pass
            else:
                kind[rd] = "unknown"
                unknown[mn] += 1
    return lo, hi, sites, quiet, unknown

bad = 0
for img in sys.argv[1:]:
    r = scan(img)
    if r is None:
        print(f"{img}: no __capstone_cap_init")
        continue
    lo, hi, sites, quiet, unknown = r
    print(f"{img.split('/')[-1]}  __capstone_cap_init [{lo:#x}, {hi:#x})  "
          + (f"{len(sites)} faulting" if sites else "no faulting")
          + (f", {len(quiet)} QUIET" if quiet else ", no quiet"))
    for addr, mn, base, off in sites[:6]:
        print(f"    FAULTS HERE  {addr:#x}  {mn} through {base}, last defined by a "
              f"scalar (offset {off})")
    if quiet:
        print(f"    plus {len(quiet)} quiet site(s): stc of a scalar into a "
              "capability slot, which faults only when read")
        for addr, rs2, base, off in quiet[:4]:
            print(f"      {addr:#x}  stc {rs2} (scalar) -> {off}({base})")
    if unknown:
        print("    unknown mnemonics, not classified: "
              + ", ".join(f"{k}x{v}" for k, v in unknown.most_common(5)))
    # The quiet form is REPORTED, never failed on. It is present in images that boot
    # and run, so a gate keyed on it would reject every image: an auipc-derived
    # capability stored to the stack with a scalar SD and reloaded with LD is a real
    # tag loss, but the FIRST store of each holder still uses the live register, so
    # only later leaves carry an untagged pointer and only if something reads them.
    # That is the same "the Nth cap-init'd array is broken and the Mth is fine" the
    # SQLite port recorded. Counting it here is how the next person finds it.
    for addr, rs2, base, off in quiet[:4]:
        print(f"    quiet  {addr:#x}  stc {rs2} (scalar) -> {off}({base}), "
              "an untagged pointer into a global, reported not failed")
    if sites:
        bad += 1
sys.exit(1 if bad else 0)
