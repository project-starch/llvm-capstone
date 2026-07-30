#!/usr/bin/env python3
"""Report a domain's cap-table carve count and its highest referenced gp index.

WHY THIS EXISTS. Every `split` the entry glue performs to carve a global allocates a
revocation node, and the RTL's allocator head is 10 bits starting at 3
(capstone_rev_node.anvil:160,168) -- so allocation ~#1022 wraps to id 0 and reuses live
ids. Reuse can splice a node into the `next` chain twice; REVOKE_NODE (:13-32) has no
visit bound and no cycle detection, so it walks forever and never answers another query.
Since EVERY `stc` blocks on a revocation-node query with no timeout
(capstone_dyn_unit.anvil:395-404), the next capability store then hangs with no trap.

So "does this domain fit the pool?" is a build-time question with a numeric answer, and
answering it here costs seconds instead of a ~35 minute firmware rebuild plus a board
session.

TWO NUMBERS, BOTH NEEDED:

  count       -- carves the glue performs = capability slots it fills.
  max index   -- the highest gp slot the CODE actually references.

`max index >= count` is always a bug: those slots are carved-but-unfilled memory, and an
`ldc` from one yields an untagged capability that stalls the first `stc` through it. That
is exactly what INTERP_BUILD_LIMIT=900 produced on 2026-07-30 -- pc froze on the first
store through slot 904 -- and it is why a diagnostic clamp must never be shipped.

READING THE DESCRIPTOR. Layout is set by CapstoneAsmPrinter::emitGpCaptableInitDesc and
documented at start-gp-captable-interp.S:35-38:

    +0  u64 built_flag      (0 in the image)
    +8  u64 count
    +16 u64 gp_slot[2]      (16 B, where the glue parks gp)
    +32 per global, 24 B:   u64 size ; u64 align ; i64 blob_off

Do not derive `count` by dividing the section size: the section is 16-aligned and padded,
so the division is not exact and has already produced a wrong record count once.

READING THE CODE. A gp slot access is `cincoffset rd, gp, rs` (funct3=1, funct7=0xc,
rs1=x3) fed by an immediate, or the folded `ldc rd, imm(gp)` when index*16 fits in 12
bits. Both are scanned; the immediate divided by 16 is the index.
"""
import os
import struct
import subprocess
import sys


def sections(path):
    out = subprocess.run(["readelf", "-SW", path], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "capstone_gp" not in line:
            continue
        # `readelf -SW` renders the index as "[ 2]", which splits into TWO tokens, so the
        # name is field 2 and the file offset is field 5 -- NOT field 4. Getting this
        # wrong reads the section address as an offset and prints plausible garbage.
        f = line.replace("]", " ").split()
        yield f[2], int(f[5], 16), int(f[6], 16)


def carve_count(path):
    for name, off, size in sections(path):
        if name == ".capstone_gp_initdesc":
            with open(path, "rb") as fh:
                fh.seek(off)
                built, count = struct.unpack("<QQ", fh.read(16))
            return count, built
    return None, None


def build_limit(path, objdump):
    """The INTERP_BUILD_LIMIT baked into the glue, or None if unclamped.

    This is the check that matters, and it is exact. The clamp is NOT visible in the
    descriptor -- `count` still reads 1059 in a clamped image, because the clamp is
    applied to the glue's loop counter at run time (start-gp-captable-interp.S:361-366):

        li   t3, <N>
        bge  t3, s4, .+8
        mv   s4, t3

    right after the table split, so the table is carved full size and only the first N
    slots are filled. Slots N..count-1 are untagged memory; an `ldc` from one yields an
    untagged capability and the first `stc` through it stalls the pipeline with no trap.
    That is precisely how a diagnostic build got baked and produced a "real" hang on
    2026-07-30. Diagnostic clamps must never ship, so this greps for the sequence in the
    entry code rather than trusting the build to have passed the right flags.
    """
    # Scan the whole disassembly, not `--section=.text.entry`: that filter yields nothing
    # here (the entry code disassembles under the plain -d output) and the detector then
    # silently reported "unclamped" for a clamped image -- a false negative in a guard,
    # which is the worst kind. `bge t3, s4` is specific enough on its own.
    out = subprocess.run([objdump, "-d", "--triple=capstone64-unknown-elf", path],
                         capture_output=True, text=True).stdout
    lines = [l.split("\t") for l in out.splitlines()]
    for i, parts in enumerate(lines):
        if len(parts) < 3 or parts[-2].strip() != "li":
            continue
        ops = parts[-1].strip()
        if not ops.startswith("t3,"):
            continue
        nxt = lines[i + 1] if i + 1 < len(lines) else []
        if len(nxt) >= 3 and nxt[-2].strip() == "bge" and nxt[-1].strip().startswith("t3, s4"):
            try:
                return int(ops.split(",")[1].strip(), 0)
            except ValueError:
                return -1
    return None


def max_gp_index(path, objdump):
    """Highest gp-relative slot index referenced by the code."""
    out = subprocess.run([objdump, "-d", "--triple=capstone64-unknown-elf", path],
                         capture_output=True, text=True).stdout
    best, best_ctx = -1, ""
    pend = {}          # reg -> immediate, from `lui`/`addi` feeding a cincoffset
    for line in out.splitlines():
        # llvm-objdump separates MNEMONIC and OPERANDS with a tab:
        #     "   10044: 5b 99 01 18 \tcincoffset\ts2, gp, zero"
        # so parts[-1] is the operand list alone and never matches a mnemonic test.
        # Rejoin them; getting this wrong silently reports "no gp accesses found".
        parts = line.split("\t")
        if len(parts) < 3:
            # "0000000000010000 <sym>:" -- a new function. Drop tracked registers so a
            # value computed in one function is never attributed to a gp access in the
            # next; this is a flow-insensitive scan and stale registers are its only
            # real failure mode.
            if line.rstrip().endswith(":") and "<" in line:
                pend.clear()
            continue
        text = f"{parts[-2].strip()} {parts[-1].strip()}"
        # folded form: ldc rd, <imm>(gp)
        if text.startswith("ldc") and "(gp)" in text:
            try:
                imm = int(text.split(",")[-1].split("(")[0].strip(), 0)
            except ValueError:
                continue
            if imm // 16 > best:
                best, best_ctx = imm // 16, line.strip()
        # split form: lui rX, hi / addi rX, rX, lo / cincoffset rZ, gp, rX
        elif text.startswith("lui") and text.count(",") == 1:
            dst, imm = (p.strip() for p in text.split(",", 1))
            dst = dst.split()[-1]
            try:
                # RISC-V lui takes a 20-bit SIGNED immediate and objdump prints it as
                # unsigned hex, so `lui a0, 0xfffff` is -4096, not 4294963200. Treating
                # it as unsigned reported a max index of 268,435,200.
                v = (int(imm, 0) << 12) & 0xFFFFFFFF
                pend[dst] = v - (1 << 32) if v >= (1 << 31) else v
            except ValueError:
                pend.pop(dst, None)
        elif text.startswith(("addi", "add ")) and text.count(",") == 2:
            dst, src, imm = (p.strip() for p in text.split(",", 2))
            dst = dst.split()[-1]
            try:
                pend[dst] = pend.get(src, 0) + int(imm, 0)
            except ValueError:
                pend.pop(dst, None)
        elif text.startswith("cincoffset") and ", gp," in text:
            src = text.split(",")[-1].strip()
            imm = pend.get(src)
            # A negative or absurd offset is a stale register, not a slot reference.
            if imm is not None and 0 <= imm < (1 << 20) and imm // 16 > best:
                best, best_ctx = imm // 16, line.strip()
    return best, best_ctx


def main():
    if len(sys.argv) < 2:
        raise SystemExit(f"usage: {sys.argv[0]} <domain.dom> [budget]")
    path = sys.argv[1]
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    # Never hardcode an absolute build path: it embeds a username in a committed file
    # (which the pre-commit name scan blocks, correctly) and breaks on any other machine.
    # capstone-test-env.sh exports CAPSTONE_LLVM_BIN; fall back to the repo-relative build
    # directory, then to whatever llvm-objdump is on PATH.
    default_bin = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "..", "llvm", "cmake-build-debug", "bin")
    objdump = os.environ.get("CAPSTONE_LLVM_BIN")
    objdump = os.path.join(objdump, "llvm-objdump") if objdump else \
        os.path.normpath(os.path.join(default_bin, "llvm-objdump"))
    if not os.path.isfile(objdump):
        objdump = "llvm-objdump"

    count, built = carve_count(path)
    if count is None:
        raise SystemExit(f"{path}: no .capstone_gp_initdesc (not a gp-captable domain?)")
    clamp = build_limit(path, objdump)
    idx, ctx = max_gp_index(path, objdump)

    print(f"carve count      : {count}")
    print(f"INTERP_BUILD_LIMIT: {'none' if clamp is None else clamp}")
    # ADVISORY ONLY. This scan is flow-insensitive: it tracks lui/addi per function and
    # cannot follow branches, so a register live across a branch can be attributed to the
    # wrong gp access. It over-reported 1064 on a build whose true maximum is 1028. Use it
    # as a smell test, never as a gate.
    print(f"max gp index     : ~{idx} (approximate)   {ctx}")
    print(f"pool budget      : {budget}")

    bad = False
    if clamp is not None:
        # Carved-but-unfilled slots. An `ldc` from one is untagged and the first `stc`
        # through it stalls the pipeline with no trap (issue R-5).
        print(f"FAIL: a diagnostic INTERP_BUILD_LIMIT={clamp} is baked into this image; "
              f"slots {clamp}..{count - 1} are carved but never filled.")
        bad = True
    if count > budget:
        print(f"FAIL: {count} carves exceeds the {budget}-allocation budget; the rev-node "
              f"pool wraps at ~1021 and reuse can hang the next stc.")
        bad = True
    if built:
        print(f"WARN: built_flag is {built}, expected 0 in the image.")
    print("FAIL" if bad else "OK")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
