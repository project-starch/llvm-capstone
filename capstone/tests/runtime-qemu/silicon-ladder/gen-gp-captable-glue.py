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
    if table_bytes > 2047:
        die("cap-table (%d B) exceeds a 12-bit immediate; needs li+sub (TODO)" % table_bytes)

    lines = ["/* GENERATED by gen-gp-captable-glue.py -- do not edit. */",
             "/* %d global(s); cap-table = %d bytes; %d initialized. */"
             % (count, table_bytes, len(init_map)),
             ".macro BUILD_GP_CAPTABLE",
             "  lcc(t1, sp, 4)                 /* t1 = sp.END */",
             "  addi t1, t1, -%d               /* reserve N*16 cap-table */" % table_bytes,
             "  split(gp, sp, t1)              /* gp = [t1,END) table; sp = [base,t1) */",
             "  delin(gp)"]

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
            # Initialized (non-zero) storage: unrolled li/sd immediates. Bounded by
            # the silicon big-table limit (offsets must stay in the 12-bit range and
            # the whole domain must fit the 0x1000 code window) -- large initialized
            # read-only tables need the tracked large-RO delivery mechanism instead.
            if stor > 2040:
                die("global %d: %d B of *initialized* data overflows the 12-bit store "
                    "offset -- large initialized tables need the large-RO silicon "
                    "delivery mechanism (tracked open item), not unrolled li/sd" % (i, stor))
            for k in range(0, stor, 8):
                word = int.from_bytes(buf[k:k + 8], "little")
                if word == 0:
                    lines.append("  sd x0, %d(t2)" % k)
                else:
                    lines.append("  li t3, 0x%x" % word)
                    lines.append("  sd t3, %d(t2)" % k)
        lines.append("  stc(t2, gp, %d)               /* cap-table[%d] */" % (i * 16, i))

    lines += ["  scc(sp, sp, t1)               /* sp.cursor = top of remaining stack */",
              "  delin(sp)", ".endm"]
    with open(out_inc, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("gen: %d global(s) (%d initialized), table=%d B -> %s"
          % (count, len(init_map), table_bytes, out_inc))


if __name__ == "__main__":
    main()
