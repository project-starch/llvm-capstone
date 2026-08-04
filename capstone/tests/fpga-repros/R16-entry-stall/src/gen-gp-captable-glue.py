#!/usr/bin/env python3
"""Cap-table builder-glue generator (silicon ladder, Stage 0a).

Reads the `-capstone-gp-captable` descriptor `.capstone_gp_table` from a compiled
domain object and emits the app-specific `BUILD_GP_CAPTABLE` assembler macro that
the generic entry glue (`start-gp-captable-generic.S`) #includes. Replaces the
hand-baked N=1 macro in `start-gpfree-captable-app.S` with one derived automatically
from the compiler's descriptor, for arbitrary global counts.

Descriptor layout (CapstoneAsmPrinter.cpp:849), in getGpCaptableIndex order so slot
i matches `ldc gp[i]` in the code:
    u64 count
    per global: { u64 size, u64 align, i64 init_off }   (record i at byte 24*(i+1))
`init_off == 0` -> zero-init (.bss). `init_off != 0` -> the global has an image
initializer template (a `sym - .` diff, emitted as an ADD/SUB reloc pair against
`.capstone_gp_table`, ADD pointing at the global symbol).

Silicon-correct initialization (see plan `sqlite-on-silicon-scoping.md`): storage is
carved from `sp` (the DATA-authority cscratch region) via `split`, and initial
values are materialized as **instruction immediates** (`li reg,<word>; sd reg,off`) —
the generator reads the initializer bytes from the object at BUILD time, so there is
NO runtime data-read from the code image (the authority that wedges on silicon). This
unifies .bss (all-zero words -> `sd x0`) and .data in one path. Matches the capstone-c
reference's carve+init model while extending it with initialized globals.
"""
import json
import os
import re
import struct
import subprocess
import sys

LLVM_BIN = os.environ.get("CAPSTONE_LLVM_BIN", "")


def tool(name):
    return os.path.join(LLVM_BIN, name) if LLVM_BIN else name


def die(msg):
    sys.stderr.write("gen-gp-captable-glue: " + msg + "\n")
    sys.exit(1)


def align_up(n, a):
    return (n + a - 1) // a * a


def read_descriptor(main_o, out_inc):
    binp = out_inc + ".gptbl.bin"
    r = subprocess.run([tool("llvm-objcopy"), "--dump-section",
                        ".capstone_gp_table=" + binp, main_o], capture_output=True)
    if r.returncode != 0 or not os.path.exists(binp):
        return None
    data = open(binp, "rb").read()
    if len(data) < 8:
        die("descriptor too short")
    (count,) = struct.unpack_from("<Q", data, 0)
    recs = []
    off = 8
    for i in range(count):
        if off + 24 > len(data):
            die("descriptor truncated at record %d" % i)
        size, align, init_off = struct.unpack_from("<QQq", data, off)
        off += 24
        recs.append((size, align, init_off))
    return recs


def read_init_symbols(main_o):
    """record-index -> (symbol, addend) for initialized globals. Each initialized
    global's init_off is a `sym - .` PC-relative diff, emitted as an ADD/SUB reloc
    pair against .capstone_gp_table at descriptor offset 24*(i+1). The ADD half
    (R_RISCV_ADD8/16/32/64 = 33..36) points at the initializer symbol; the SUB half
    (37..40) points at the local `.L0` anchor. Select by RELOC TYPE, not by symbol
    name -- a function-local const is promoted to a `.L__const.*` symbol, so a
    "skip .L-prefixed symbols" heuristic would wrongly drop it and zero-fill it."""
    ADD_RELOCS = {33, 34, 35, 36}       # R_RISCV_ADD{8,16,32,64}
    out = subprocess.run([tool("llvm-readobj"), "-r", "--elf-output-style=GNU", main_o],
                         capture_output=True, text=True).stdout
    init_map = {}
    in_sec = False
    for line in out.splitlines():
        m = re.match(r"Relocation section '([^']+)'", line)
        if m:
            in_sec = (m.group(1) == ".rela.capstone_gp_table")
            continue
        if not in_sec:
            continue
        parts = line.split()
        if len(parts) < 5 or not re.fullmatch(r"[0-9a-fA-F]+", parts[0]):
            continue
        offset = int(parts[0], 16)
        try:
            reltype = int(parts[1], 16) & 0xffffffff   # Info = (symidx<<32)|type
        except ValueError:
            continue
        if reltype not in ADD_RELOCS:   # the SUB anchor half -- skip
            continue
        sym = parts[4]
        addend = 0
        if "+" in parts:
            try:
                addend = int(parts[parts.index("+") + 1], 0)
            except (ValueError, IndexError):
                addend = 0
        if offset % 24 == 0 and offset >= 24:
            init_map[offset // 24 - 1] = (sym, addend)
    return init_map


def read_symtab(main_o):
    out = subprocess.run([tool("llvm-readobj"), "--symbols", "--elf-output-style=JSON",
                          main_o], capture_output=True, text=True).stdout
    d = json.loads(out)
    table = {}
    for f in d:
        for k, v in f.items():
            if k == "Symbols":
                for s in v:
                    sym = s["Symbol"]
                    table[sym["Name"]["Name"]] = (sym["Section"]["Name"],
                                                  sym["Value"], sym["Size"])
    return table


_section_cache = {}


def section_bytes(main_o, secname):
    if secname not in _section_cache:
        binp = "/tmp/.gptbl_sec_%d.bin" % (abs(hash(secname)) % 100000)
        subprocess.run([tool("llvm-objcopy"), "--dump-section", secname + "=" + binp,
                        main_o], capture_output=True)
        _section_cache[secname] = open(binp, "rb").read() if os.path.exists(binp) else b""
    return _section_cache[secname]


def main():
    if len(sys.argv) != 3:
        die("usage: gen-gp-captable-glue.py <domain_main.o> <out.inc>")
    main_o, out_inc = sys.argv[1], sys.argv[2]

    recs = read_descriptor(main_o, out_inc)
    if recs is None:
        with open(out_inc, "w") as f:
            f.write("/* generated: no .capstone_gp_table (0 indexable globals) */\n"
                    ".macro BUILD_GP_CAPTABLE\n.endm\n")
        print("gen: 0 globals (no descriptor) -> empty BUILD_GP_CAPTABLE")
        return
    count = len(recs)
    init_map = read_init_symbols(main_o)
    symtab = read_symtab(main_o) if init_map else {}

    table_bytes = count * 16

    # Reserving the cap-table was `addi t1, t1, -table_bytes`, whose immediate is 12-bit
    # SIGNED, so it caps the table at 2047 B = 127 globals and used to die() above that.
    # That ceiling is not merely inconvenient: it is exactly why the silicon-ladder rungs
    # (<=96 carves) and the SQLite images (181 carves, 2896 B table) could never be compared
    # on the same glue, which left an untested band across the boundary while investigating
    # R-16. Fall back to li+sub past the immediate range.
    #
    # t2 is the correct scratch: the macro's documented contract is "Clobbers t1,t2"
    # (start-gp-captable-generic.S), and t2 is reassigned by the `split(t2, ...)` below
    # before any use. The <=2047 path is left byte-identical so every existing rung's
    # codegen is unchanged.
    if table_bytes <= 2047:
        reserve = ["  addi t1, t1, -%d               /* reserve N*16 cap-table */" % table_bytes]
    else:
        reserve = ["  li t2, %d                      /* table > 12-bit imm: li+sub */" % table_bytes,
                   "  sub t1, t1, t2                 /* reserve N*16 cap-table */"]

    prologue = ["  lcc(t1, sp, 4)                 /* t1 = sp.END */"] + reserve + [
                "  split(gp, sp, t1)              /* gp = [t1,END) table; sp = [base,t1) */",
                "  delin(gp)"]
    lines = []

    loop_id = 0
    for i, (size, align, init_off) in enumerate(recs):
        stor = align_up(size, 16)   # 16-align keeps split bounds + storage base aligned
        # Build the exact storage image: initializer bytes (if any) + zero pad to stor.
        init_bytes = b""
        tag = ".bss/zero"
        if i in init_map:
            sym, addend = init_map[i]
            if sym not in symtab:
                die("initialized global %d symbol %r not in symtab" % (i, sym))
            secname, val, ssize = symtab[sym]
            blob = section_bytes(main_o, secname)
            start = val + addend
            init_bytes = blob[start:start + size]
            if len(init_bytes) < size:
                die("global %d (%s): only %d of %d init bytes in %s"
                    % (i, sym, len(init_bytes), size, secname))
            tag = "init<-%s@%s" % (sym, secname)
        buf = init_bytes + b"\x00" * (stor - len(init_bytes))

        lines.append("  /* global %d: size=%d align=%d -> %d B (%s) */"
                     % (i, size, align, stor, tag))
        if stor <= 2047:
            lines.append("  addi t1, t1, -%d" % stor)
        else:
            lines.append("  li t3, %d" % stor)   # reserve overflows 12-bit imm
            lines.append("  sub t1, t1, t3")
        lines.append("  split(t2, sp, t1)             /* t2 = storage cap */")
        lines.append("  delin(t2)")

        if all(b == 0 for b in buf):
            # All-zero (.bss): a compact runtime loop, not an unrolled sd storm.
            # Unrolling would (a) overflow the 12-bit sd immediate past 2 KB and
            # (b) balloon .text (~1 insn / 8 B) past the 0x1000 PCC code window.
            # t3 = byte counter, t4 = moving store pointer (a copy of t2).
            loop_id += 1
            lbl = 90 + (loop_id % 9)      # numeric local label; nearest-match safe
            lines.append("  li t3, %d" % stor)
            lines.append("  cincoffset(t4, t2, x0)       /* t4 = t2 (moving ptr) */")
            lines.append("%d:" % lbl)
            lines.append("  sd x0, 0(t4)")
            lines.append("  cincoffsetimm(t4, t4, 8)")
            lines.append("  addi t3, t3, -8")
            lines.append("  bnez t3, %db" % lbl)
        else:
            sym, addend = init_map.get(i, (None, 0))
            # Large-RO delivery path: for a big initialized global, COPY its
            # initializer from the monitor-placed blob at the front of dom_data
            # (dom_data[k] == image[globals_base + k]) instead of an li/sd immediate
            # storm that overflows the 0x1000 code window. Post-link offset comes from
            # linker-resolved `lla <sym> - __gpfree_globals_base`. Requirements: a
            # file-scope (linkable, non-.L) symbol and an 8-multiple size.
            # The large-RO copy path is BROKEN (C-4b: it trips
            # `helper_cssplit: rs1_v->tag && !rs2_v->tag` at domain entry). Set
            # LADDER_NO_RO_COPY=1 to force the unrolled li/sd path instead, which
            # WORKS -- verified end to end: rv8_sha512 with its full 640 B table
            # returns its oracle under DOMAIN_WINDOW=32k. The unrolled path costs
            # ~8 KB of .text for 640 B of data, so it needs the 32 KiB window; at
            # 4 KiB the link fails outright. Use both knobs together.
            COPY_THRESHOLD = (1 << 30) if os.environ.get("LADDER_NO_RO_COPY") == "1" else 256
            if (stor > COPY_THRESHOLD and (size % 8) == 0
                    and sym is not None and not sym.startswith(".L")):
                loop_id += 1
                lbl = 90 + (loop_id % 9)
                lines.append("  /* large-RO: copy %d B from dom_data-front blob */" % size)
                lines.append("  lla t4, %s" % sym)
                if addend:
                    lines.append("  addi t4, t4, %d" % addend)
                lines.append("  lla t5, __gpfree_globals_base")
                lines.append("  sub t5, t4, t5               /* blob offset = sym-base */")
                lines.append("  cincoffset(t4, sp, t5)       /* t4 = dom_data + off (src) */")
                lines.append("  cincoffset(t3, t2, x0)       /* t3 = storage (dst) */")
                lines.append("  li t6, %d" % size)
                lines.append("%d:" % lbl)
                lines.append("  ld t5, 0(t4)")
                lines.append("  sd t5, 0(t3)")
                lines.append("  cincoffsetimm(t4, t4, 8)")
                lines.append("  cincoffsetimm(t3, t3, 8)")
                lines.append("  addi t6, t6, -8")
                lines.append("  bnez t6, %db" % lbl)
                if stor > size:                     # zero the 16-align pad tail
                    lines.append("  sd x0, 0(t3)")
            else:
                # Small initialized storage: unrolled li/sd immediates (no monitor dep).
                # Bounded by the 12-bit store offset; a big non-copyable global here is
                # the tracked large-RO limit (function-local .L template, or odd size).
                if stor > 2040:
                    die("global %d: %d B of *initialized* data overflows the 12-bit "
                        "store offset and is not copy-eligible (sym=%r, size%%8=%d) -- "
                        "needs the large-RO copy path (file-scope symbol, 8-mult size)"
                        % (i, stor, sym, size % 8))
                for k in range(0, stor, 8):
                    word = int.from_bytes(buf[k:k + 8], "little")
                    if word == 0:
                        lines.append("  sd x0, %d(t2)" % k)
                    else:
                        lines.append("  li t3, 0x%x" % word)
                        lines.append("  sd t3, %d(t2)" % k)
        # The cap-table store offset is ALSO a 12-bit signed immediate, so entries past
        # index 127 cannot be reached with `stc(t2, gp, i*16)`. Use the register form of
        # cincoffset to materialise the slot address instead -- no immediate involved.
        # t3/t4 are dead here: both are scratch for the init-copy above, which has already
        # finished for this entry. gp is unharmed, cincoffset passes rs1 through unchanged.
        if i * 16 <= 2047:
            lines.append("  stc(t2, gp, %d)               /* cap-table[%d] */" % (i * 16, i))
        else:
            lines.append("  li t3, %d" % (i * 16))
            lines.append("  cincoffset(t4, gp, t3)        /* t4 = &cap-table[%d] */" % i)
            lines.append("  stc(t2, t4, 0)                /* cap-table[%d] */" % i)

    lines += ["  scc(sp, sp, t1)               /* sp.cursor = top of remaining stack */",
              "  delin(sp)", ".endm"]

    # ---- C-4b fix: delinearize sp BEFORE any copy-path cincoffset of it. -------
    #
    # `cincoffset rd, rs1, rs2` with rd != rs1 CONSUMES rs1 unless rs1 is NONLIN:
    #   op_helper.c:635  *rd_v = *rs1_v;
    #                    if(!captype_is_copyable(rs1_v->val.cap.type)) *rs1_v = CAPREGVAL_NULL;
    #   cap.h:122        captype_is_copyable(ty) { return ty == CAP_TYPE_NONLIN; }
    # `sp` arrives from cscratch as CAP_TYPE_LIN and the builder's only delin(sp) is
    # its LAST line, so the copy path's `cincoffset(t4, sp, t5)` nulls sp outright.
    # Every following `split(t2, sp, t1)` then trips helper_cssplit's
    # `assert(rs1_v->tag && !rs2_v->tag)` with tag == 0. That is C-4b, and it is why
    # the failure appears only AFTER "Created domain ID = 0", only when the copy path
    # is selected, and never in the zero-init or unrolled paths.
    #
    # delin is the minimal correct fix rather than a workaround: helper_cssplit
    # asserts `type == LIN || type == NONLIN`, so every split still works, and split
    # (unlike cincoffset) never consumes rs1. sp is delinearized by the builder's last
    # line regardless, so this only moves that transition earlier -- no domain-visible
    # difference in the capability handed to compiled code.
    #
    # Emitted ONLY when a global actually took the copy path. This project has a
    # documented layout sensitivity (2026-07-26: four added instructions flipped a
    # passing rung), so every currently-measured rung must stay BYTE-IDENTICAL. The
    # condition is derived from the emitted body rather than by re-testing the
    # eligibility predicate, so the two cannot drift apart.
    uses_copy_path = any("large-RO: copy" in l for l in lines)
    if uses_copy_path:
        # ---- C-13: with sp pre-delin'd, EVERY LATER delin IS FATAL ON SILICON. ------
        #
        # The RTL's DELIN (capstone-ariane capstone_dyn_unit.anvil) accepts
        # CAP_TYPE_LINEAR only and raises UNEXPECTED_CAP_TYPE otherwise -- it is NOT
        # idempotent. Our QEMU helper_csdelin returns early for an already-NONLIN cap,
        # so a double delin is a silent no-op under emulation and a hard fault on the
        # board. SPLIT preserves cap_type (modify_cap_start/_end/_revnode copy it
        # through), so once the C-4b prologue delins sp, the gp and t2 caps split from
        # it are ALREADY NONLIN, as is sp itself at the tail.
        #
        # So the copy-path pre-delin above silently turned delin(gp), delin(t2) and the
        # trailing delin(sp) into faults. This is the likely root cause of R-9
        # (beebs_ns hangs) and of copy-path rungs failing on hardware while passing on
        # QEMU -- copy-path rungs are exactly the ones that get the pre-delin.
        #
        # Dropping them is a semantic no-op: those capabilities are non-linear already.
        # Gated on uses_copy_path so every non-copy-path rung stays BYTE-IDENTICAL.
        def _drop_redundant_delins(seq):
            out = []
            for l in seq:
                if l.strip().startswith("delin("):
                    out.append("  /* C-13: %s dropped -- already NONLIN via the "
                               "pre-delin'd sp; delin is LINEAR-only on silicon */"
                               % l.strip())
                    continue
                out.append(l)
            return out

        prologue = _drop_redundant_delins(prologue)
        lines = _drop_redundant_delins(lines)
        prologue = ["  delin(sp)                      /* C-4b: sp is LIN; the copy path's",
                    "                                    cincoffset would consume it */"] + prologue

    lines = ["/* GENERATED by gen-gp-captable-glue.py -- do not edit. */",
             "/* %d global(s); cap-table = %d bytes; %d initialized.%s */"
             % (count, table_bytes, len(init_map),
                " Copy path in use -> sp pre-delin." if uses_copy_path else ""),
             ".macro BUILD_GP_CAPTABLE"] + prologue + lines
    with open(out_inc, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("gen: %d global(s) (%d initialized), table=%d B, copy-path=%s -> %s"
          % (count, len(init_map), table_bytes, "yes" if uses_copy_path else "no", out_inc))


if __name__ == "__main__":
    main()
