#!/usr/bin/env python3
"""Check gp-captable domains against CVA6's ACTUAL bounds-compression behaviour.

THE RULE, from the RTL (capstone-ariane/core/include/ariane_pkg.sv).
compress_bounds has two branches, selected by whether the cursor sits on the base:

    if (bounds.start == cursor) begin   // "cursorless" -- ariane_pkg.sv:749-751

`split` sets cursor == base on BOTH of its outputs (capstone_dyn_unit.anvil:139-144),
so every capability the glue carves takes the cursorless branch. There:

  * the BASE is exact at any alignment -- decompress returns `start: cursor`
    verbatim (ariane_pkg.sv:662-665). There is no base rounding in this branch.
  * the TOP is truncated DOWN to a multiple of 2**E, where E is derived from the
    highest bit at which cursor and top DIFFER, floored at bit 20:

        lz = 63; while (lz > 20 && bit(cursor,lz) == bit(top,lz)) lz--;
        E  = lz - 20;                                  // ariane_pkg.sv:752-759

    So E is 0 -- and the capability exact -- whenever base and top lie in the same
    2 MiB (2**21) window. E only goes positive when [base, top) STRADDLES a 2 MiB
    boundary, and then the top silently loses its low E bits.

WHAT THIS MEANS FOR US. A domain that is <= 2 MiB and 2 MiB-aligned has every
interior capability inside one window (E = 0, exact), and any capability whose top
is exactly the window edge has a top that is a multiple of 2**21 and so survives
truncation too. Both hold today: the kernel module rounds the allocation up to a
power-of-two page count (capstone.c:83-84) and the page allocator returns it
2**order-page aligned. Domains are therefore exact BY CONSTRUCTION, not by luck --
which is why no ladder rung has ever hit this.

IT STOPS HOLDING THE MOMENT A DOMAIN EXCEEDS 2 MiB. Then interior splits straddle a
window boundary, tops truncate DOWNWARD, and a global silently gets a shorter
capability than it asked for. That is a real cliff sitting just past SQLite's
current size, so this script exists to fail the build when a domain reaches it.

AND QEMU CANNOT SEE ANY OF IT. helper_cssplit works on full 64-bit base/end/cursor
and never calls cap_compress (op_helper.c:848-870); tagged loads even restore exact
bounds from an out-of-band shadow map, bypassing the lossy decode entirely. RTL
round-trips EVERY capability write-back through compress_bounds (ex_stage.sv:
1080-1098) because the compressed form IS the architectural register state. So a
top-truncation bug passes under QEMU forever. Same shape as the DELIN divergence.

NOTE the OTHER branch is the one with the granule(L) = 1 << (max(0, hb(L)-12) + 3)
rule, base truncated down and top rounded up (ariane_pkg.sv:769-806). It applies
only once cursor != base -- e.g. after a cincoffset, or after the monitor's
C_SET_CURSOR. That is exactly what C-13 turned out to be, and it is why the granule
model belongs in the monitor's split geometry and NOT in the glue's carve.

THE PROCESS-ABI GEOMETRY (2026-10-05, ISSUES R-11's first hit). Everything above models
the old SDK region: a power-of-two block sized from the image alone, based at 0. A
managed application (capstone-exec, any image with a .capstone_domreq section) is laid
out differently, and this script reported memcached's image OK while it faulted on
silicon. So an image that declares .capstone_domreq is checked against the process
path instead, replayed end to end with a literal port of compress/decompress_bounds
(ariane_pkg.sv:672-728 and 793-851 at 776d9d859):
  * the module: tot = roundup_pow_of_two(PAGE_ALIGN(code_len + domreq_data + 9 KiB))
    (process.c, MONITOR_SPLIT_SLACK), code_len being the PT_LOAD span libcapstone
    passes; CMA places it 1 MiB-aligned (CONFIG_CMA_ALIGNMENT), so EVERY 1 MiB-aligned
    base in the CMA window is tried, not one;
  * the monitor's managed create_domain: repr_gran, split_size, data_off, data_top
    aligned down to repr_gran (M-14), the dom_data split, and its cursor parks at
    data_top - 16 and - 32 (each a lossy re-encode); and (monitor_splits, 2026-10-13) every
    capability the monitor itself splits off: code, seal, dom_data, the descriptor area, gp
    and the code capability at its entry, each from its parent's DECODED bounds;
  * the glue: CAPSTONE_GLUE_CONTEXTS' move to END - 32 and back and the arena split at
    END - A (A from __capstone_context_arena_bytes), then the table and every global
    carved downward, each split's upper half taking the lower half's DECODED top, with
    or without CAPSTONE_GLUE_CARVE_ALIGN, which is read from the image's own code (its
    `xor t3,t3,t1; srli t3,t3,21`); --carve-align on/off overrides that.
Reported per base: SHORT (a global's storage capability is shorter than the global:
the R-11 fault), INEXACT (a carve point lost bits: SHORT's cause), M-14 (dom_data's
top moved), and the stack left above the globals template. WIDEN (a storage
capability's bounds grow once its cursor moves off the base) is printed as a count
and does not fail the check.

THE FULL SCHEME DOES NOT RENORMALISE (found 2026-10-13, both formats). compress_bounds
rounds the top up with `T[11:3] += 1` and keeps E. When that pushes the span to 2**13
granules, the word needs E+1 and decodes about half the span SHORT. Exactly: a capability
[s, e) with len >= 4 KiB, E = hb(len) - 12, g = 2**(E+3) decodes NOT covering itself iff
roundup(e, g) - rounddown(s, g) >= 2**(E+13) (200,000 random cases, no exception). capstone-qemu's
cap_compress had the same flaw (0 mismatches against this port on 60,000 cases; its C mirrors the
RTL's identifiers, so that guards against a typo here, not a shared misreading -- a third port
written from the SV with explicit bit slicing agreed on 300,000). QEMU showed it only through STC:
its SPLIT and SHRINK keep exact bounds. Fixed in the encoder by taking E+1 when the rounded span
reaches 2**(E+13): capstone-qemu aad781e15a, and capstone-ariane 2a3a87dfc (fp-fixes, on 0568f93a9;
the capstone-sublet tree carries the same step). --renorm models that fix in either format, ported
from 2a3a87dfc: 0 mismatches against its bounds-model.py (300,000 decodes, fixed and unfixed) and
against aad781e15a (qemu-encoder-diff.py --renorm). Every encoder without it reaches the wrap once a
cursor leaves the base, so the replay also fails on:
  * SPLIT TRAP: a carve point outside the remainder's DECODED bounds, which SPLIT refuses
    with ILLEGAL_OPERAND_VALUE (capstone_dyn_unit.anvil, func SPLIT). The replay stops there.
  * SHORT ONCE MOVED: a global (the zero-fill walker), the stack (handed over with its cursor
    at t1, `scc sp, sp, t1`), dom_data or the context arena re-encoding to bounds that no longer
    cover it. The runtime's own later carves (context.c's) and heap splits are NOT replayed.

THE CLASSIC SDK PATH (an image without .capstone_domreq) is replayed by the same code under
--format full: the module sizes the block code_len + max(code_len, 64 KiB) (capstone.c's
dom_headroom), and the monitor keeps data_top = tot_size and parks only gp. Under
--format cursorless such an image keeps the older window check below (check()).

THE FULL FORMAT (--format full; capstone-sublet, 2026-10-13). The next bitstream's RTL removes
the cursorless branch: bounds_t IS the 27-bit full scheme for every capability, and the freed
28th bit widens perm to 4 (word {revnode 30 | perm 4 | type 3 | bounds 27}, capstone-qemu's
layout on isa/sublet-lifetimes). compress_bounds and decompress_bounds keep the old full
branch's bodies unchanged, so --format full is the same port with the start == cursor branch
skipped. --format full --renorm is the committed capstone-sublet tree, capstone-ariane d34c26002
(the format move on fp-fixes 4a4261bb4): its compress/decompress are 4a4261bb4's full branch with
the cursorless type and decoder branch removed (`git diff 4a4261bb4 d34c26002`, read 2026-10-13).
--format full alone is that tree before R-54's step, as read on 2026-10-13 before 13:04. Under it the base rounds DOWN and the top rounds UP,
to 2**(E+3) with E = max(0, hb(len) - 12) (exact below 4 KiB), so a carve no longer shortens a
global: it WIDENS one. A split at a point off the remainder's granule leaves the remainder's
top above the split, and the next global inherits that top. So under --format full:
  * OVERLAP: a global's storage capability reaches outside its own [t1, want_top), into its
    neighbour (or the stack remainder reaches into the lowest global). It fails the check unless
    --allow-overlap, so that accepting it is written into a build recipe, not defaulted.
  * the remainder's granule at the first carve is printed: the alignment a per-global carve
    would need, i.e. whether CAPSTONE_GLUE_CARVE_ALIGN can be adapted or the glue must first
    split off a globals sub-region.
  * WIDEN must read 0 (the encoding no longer depends on the cursor); a non-zero count is an
    ERROR in the port, not a finding.
The default stays --format cursorless, the resident bitstreams' encoding (0568f93a9 and
earlier, ariane_pkg.sv at 776d9d859). Every header line names the format it modelled.

Exit status 1 if any domain is at risk.
"""
import random
import struct
import subprocess
import sys

WINDOW_BITS = 21                      # ariane_pkg.sv floors the scan at bit 20
WINDOW = 1 << WINDOW_BITS
FORMAT = "cursorless"                 # --format: "cursorless" (resident silicon) or "full" (capstone-sublet)
RENORM = False                        # --renorm: R-54's fix, capstone-ariane 2a3a87dfc (fp-fixes) / aad781e15a (QEMU)


def full_granule(length):
    """The full scheme's exactness granule for a capability of this length: 1 below 4 KiB (iE = 0), else
    2**(E+3) with E = max(0, hb(length) - 12)."""
    if length < 4096:
        return 1
    return 1 << (max(0, length.bit_length() - 1 - 12) + 3)


def cursorless_top_exact(base, top):
    """Replay ariane_pkg.sv:752-759 for a split-produced (cursor == base) cap."""
    lz = 63
    while lz > 20 and ((base >> lz) & 1) == ((top >> lz) & 1):
        lz -= 1
    e = lz - 20
    return top % (1 << e) == 0, e


def find_initdesc(path):
    for tool in ("llvm-readelf", "readelf"):
        try:
            out = subprocess.run([tool, "-SW", path], capture_output=True,
                                 text=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            continue
        for line in out.splitlines():
            if ".capstone_gp_initdesc" in line:
                f = line.split()
                return int(f[f.index(".capstone_gp_initdesc") + 3], 16)
        return None
    return None


M64 = (1 << 64) - 1


def rtl_compress(start, end, cursor):
    """compress_bounds, ariane_pkg.sv:793-851 at 776d9d859 (a literal port). --format full skips the cursorless
    branch, as the capstone-sublet RTL does; the full branch is unchanged there."""
    if FORMAT == "cursorless" and start == cursor:
        lz = 63
        while lz > 20 and ((cursor >> lz) & 1) == ((end >> lz) & 1):
            lz -= 1
        e = lz - 20
        return ("cl", (end >> e) & 0x1FFFFF, e)
    ln = (end - start) & M64
    lz = 63
    while lz > 12 and ((ln >> lz) & 1) == 0:
        lz -= 1
    e = lz - 12
    if RENORM and not (e == 0 and ((ln >> 12) & 1) == 0):
        # R-54's fix, ported from capstone-ariane 2a3a87dfc (ariane_pkg.sv compress_bounds): when the span rounded out
        # to 2^(E+3) reaches 1024 granules, take E + 1 (64-bit arithmetic, as xlen_t)
        tg = ((end + ((1 << (e + 3)) - 1)) & M64) >> (e + 3)
        bg = start >> (e + 3)
        if ((tg - bg) & M64) >= 1024:
            e += 1
    if e == 0 and ((ln >> 12) & 1) == 0:
        return ("full", 0, start & 0x3FFF, end & 0xFFF, 0)
    b13_3 = ((start >> e) >> 3) & 0x7FF
    t11_3 = ((end >> e) >> 3) & 0x1FF
    if ((end >> (e + 3)) << (e + 3)) != end:
        t11_3 = (t11_3 + 1) & 0x1FF
    return ("full", 1, b13_3, t11_3, e)


def rtl_decompress(enc, cursor):
    """decompress_bounds, ariane_pkg.sv:672-728 at 776d9d859 (a literal port)."""
    if enc[0] == "cl":
        _, t, e = enc
        return cursor, ((((cursor >> (e + 21)) << 21) | t) << e) & M64
    _, ie, bf, tf, e = enc
    if ie == 0:
        b, t, carry, msb = bf, tf, (tf & 0xFFF) < (bf & 0xFFF), 0
    else:
        b, t, carry, msb = bf << 3, tf << 3, tf < ((bf << 3 >> 3) & 0x1FF), 1
    t = (t & 0xFFF) | ((((b >> 12) + carry + msb) & 3) << 12)
    lo = (((cursor >> (e + 14)) << 14) | b) << e
    hi = (((cursor >> (e + 14)) << 14) | t) << e
    a3, b3, t3 = (cursor >> (e + 11)) & 7, (b >> 11) & 7, (t >> 11) & 7
    r = (b3 - 1) & 7
    if a3 >= r and t3 < r:
        hi += 1 << (e + 14)
    elif a3 < r and t3 >= r:
        hi -= 1 << (e + 14)
    if a3 >= r and b3 < r:
        lo += 1 << (e + 14)
    elif a3 < r and b3 >= r:
        lo -= 1 << (e + 14)
    return lo & M64, hi & M64


def writeback(start, end, cursor):
    """What a register holds after the core writes [start, end) with this cursor."""
    return rtl_decompress(rtl_compress(start, end, cursor), cursor)


def read_elf(path):
    """PT_LOAD span, sections by name and symbols by name, from the file itself."""
    img = open(path, "rb").read()
    if img[:4] != b"\x7fELF" or img[4] != 2:
        raise ValueError("not a 64-bit ELF")
    phoff, shoff = struct.unpack_from("<QQ", img, 32)
    phentsize, phnum, shentsize, shnum, shstrndx = struct.unpack_from("<HHHHH", img, 54)
    lo = hi = None
    for i in range(phnum):
        ptype, _fl, _off, vaddr, _pa, _fsz, memsz, _al = struct.unpack_from("<IIQQQQQQ", img, phoff + i * phentsize)
        if ptype == 1:
            lo = vaddr if lo is None else lo        # libcapstone: the FIRST PT_LOAD's vaddr
            hi = max(hi or 0, vaddr + memsz)
    shdrs = [struct.unpack_from("<IIQQQQIIQQ", img, shoff + i * shentsize) for i in range(shnum)]
    strtab = shdrs[shstrndx]
    def name(off, tab):
        end = img.index(b"\0", tab[4] + off)
        return img[tab[4] + off:end].decode()
    secs = {name(h[0], strtab): h for h in shdrs}
    syms = {}
    for h in shdrs:
        if h[1] == 2:                                 # SHT_SYMTAB
            link = shdrs[h[6]]
            for j in range(h[5] // 24):
                st_name, _info, _other, st_shndx, st_value, _size = struct.unpack_from("<IBBHQQ", img, h[4] + 24 * j)
                if st_name:
                    syms[name(st_name, link)] = (st_value, st_shndx)
    return img, lo, hi, secs, syms


def glue_arena(img, secs):
    """CAPSTONE_CONTEXT_ARENA_BYTES as the glue's CONTEXTS_FIRST_ENTRY loads it: `addi t5, t3, -32` ... `li t4, A`
    then `sub t3, t3, t4`. B0 builds keep no symbol for it (the glue uses the define and the linker drops
    __capstone_context_arena_bytes), so it is read from the code. None when the sequence is absent."""
    text = secs.get(".text")
    if not text:
        return None
    off, size = text[4], text[5]
    words = [struct.unpack_from("<I", img, off + 4 * i)[0] for i in range(size // 4)]
    addi_t5_t3_m32 = (0xFE0 << 20) | (28 << 15) | (30 << 7) | 0x13
    sub_t3_t3_t4 = (0x20 << 25) | (29 << 20) | (28 << 15) | (28 << 7) | 0x33
    found = set()
    for i, w in enumerate(words):
        if w != sub_t3_t3_t4 or addi_t5_t3_m32 not in words[max(0, i - 12):i]:
            continue
        value, j = 0, i - 1
        seq = []
        while j >= 0 and ((words[j] & 0xF80) >> 7) == 29 and (words[j] & 0x7F) in (0x37, 0x13, 0x1B):
            seq.append(words[j])
            j -= 1
        for w2 in reversed(seq):                       # lui / addi / addiw into t4, in program order
            op = w2 & 0x7F
            if op == 0x37:
                value = ((w2 >> 12) << 12) & 0xFFFFFFFF
                value -= (value & 0x80000000) << 1
            else:
                imm = w2 >> 20
                imm -= (imm & 0x800) << 1
                rs1 = (w2 >> 15) & 31
                value = (value if rs1 == 29 else 0) + imm
        if seq:
            found.add(value)
    if len(found) > 1:
        raise ValueError("ambiguous CONTEXTS_FIRST_ENTRY arena sizes %s" % sorted(found))
    return found.pop() if found else None


def glue_carve_align(img, secs):
    """Whether the glue was built with CAPSTONE_GLUE_CARVE_ALIGN: its granule computation starts with
    `xor t3, t3, t1` immediately followed by `srli t3, t3, 21`, a pair nothing else in the glue emits."""
    text = secs.get(".text")
    if not text:
        return False
    off, size = text[4], text[5]
    xor_t3 = (6 << 20) | (28 << 15) | (4 << 12) | (28 << 7) | 0x33
    srli_t3_21 = (21 << 20) | (28 << 15) | (5 << 12) | (28 << 7) | 0x13
    prev = None
    for i in range(size // 4):
        w = struct.unpack_from("<I", img, off + 4 * i)[0]
        if prev == xor_t3 and w == srli_t3_21:
            return True
        prev = w
    return False


def process_layout(path, arena_override=None):
    img, lo, hi, secs, syms = read_elf(path)
    if lo is None:
        raise ValueError("no PT_LOAD")
    managed = ".capstone_domreq" in secs
    req_data = req_stack = 0
    if managed:
        d = secs[".capstone_domreq"]
        magic, req_data, req_stack = struct.unpack_from("<QQQ", img, d[4])
        if magic != 0x5145524d4f445043:
            raise ValueError("bad .capstone_domreq magic")
    ini = secs[".capstone_gp_initdesc"]
    _built, count = struct.unpack_from("<QQ", img, ini[4])
    recs = [struct.unpack_from("<QQq", img, ini[4] + 32 + 24 * i) for i in range(count)]
    arena = arena_override if arena_override is not None else glue_arena(img, secs)
    if arena is None and "__capstone_context_entry" in syms:
        raise ValueError("the image has minted contexts but no CONTEXTS_FIRST_ENTRY arena split was recognised in "
                         ".text; pass --arena BYTES")
    return dict(managed=managed, carve_align=glue_carve_align(img, secs),
                code_len=hi - lo, gpoff=ini[3] - lo, req_data=req_data, req_stack=req_stack,
                count=count, recs=recs, arena=arena or 0)


def monitor_splits(lay, base, tot, code_size, split_size, data_off, data_top, out):
    """The monitor's own create_domain splits (sbi_capstone.c, the FPGA line wt-mon-supcall 1f9aedd): the block, the
    seal split at base + split_size, the data split at ds, the managed descriptor split at data_top, the gp split at
    base + gpoff, and the code capability's cursor moved to its entry. Each split refuses a point outside its parent's
    DECODED bounds (TRAP); a capability that does not cover its region is SHORT ("monitor ..." in out["moved"]); one that
    covers more is OVERLAP. C-13 sizes repr_gran from the DATA length, so under a full-scheme encoding a code region
    whose own granule is larger rounds its top outward past split_size (2026-10-13; exact under cursorless)."""
    def split(name, parent, mid):
        if not parent[0] <= mid < parent[1]:
            out["trap"].append(("monitor " + name, mid - parent[0], parent[0], parent[1]))
            return None, None
        return writeback(parent[0], mid, parent[0]), writeback(mid, parent[1], mid)

    def judge(name, got, want):
        if got[0] > want[0] or got[1] < want[1]:
            out["moved"].append(("monitor " + name, want[1] - want[0], got[0], got[1]))
        elif got != want:
            out["overlap"].append(("monitor " + name, (want[0] - got[0]) + (got[1] - want[1]), got[0], got[1]))

    ds = base + split_size + data_off
    code, rest = split("seal split", writeback(base, base + tot, base), base + split_size)
    if rest is None:
        return
    judge("code [base, split_size)", code, (base, base + split_size))
    seal, data = split("data split", rest, ds)
    if data is None:
        return
    judge("seal", seal, (base + split_size, ds))
    if lay["managed"]:
        data, _desc = split("descriptor split", data, data_top)
        if data is None:
            return
    judge("dom_data", data, (ds, data_top))
    gpoff = lay["gpoff"]
    if gpoff and code_size > gpoff:
        c, gp = split("gp split", code, base + gpoff)
        if gp is None:
            return
        judge("code [base, gpoff)", c, (base, base + gpoff))
        judge("gp [gpoff, split_size)", gp, (base + gpoff, base + split_size))
        judge("code at its entry", writeback(c[0], c[1], base + min(0x40, gpoff - 16)), (base, base + gpoff))


def replay_process(lay, base, carve_align):
    """One base: the module's block, the monitor's split, the glue's carve. An image without .capstone_domreq takes
    the classic path: the module sizes the block as code_len + max(code_len, 64 KiB) (capstone.c's dom_headroom),
    the monitor keeps data_top = tot_size (no descriptor area, no top alignment) and parks only gp."""
    out = dict(short=[], inexact=0, widen=0, m14=None, overlap=[], gran=None, trap=[], moved=[])
    if lay["managed"]:
        total = lay["code_len"] + lay["req_data"] + 8 * 1024 + 1024
    else:
        total = lay["code_len"] + max(lay["code_len"], 64 * 1024)
    total = (total + 4095) & ~4095
    tot = 1 << (total - 1).bit_length()
    code_size = (lay["code_len"] + 15) & ~15
    repr_len = tot - code_size - 1536
    hb = repr_len.bit_length() - 1
    gran = 1 << ((hb - 12 if hb > 12 else 0) + 3)
    split_size = (code_size + gran - 1) & ~(gran - 1)
    data_off = (1536 + gran - 1) & ~(gran - 1)
    data_top = ((base + tot - 1024) & ~(gran - 1)) if lay["managed"] else base + tot
    ds = base + split_size + data_off
    monitor_splits(lay, base, tot, code_size, split_size, data_off, data_top, out)
    s, e = writeback(ds, data_top, ds)                 # the dom_data split (cursorless)
    for cur in ((data_top - 16, data_top - 32, ds) if lay["managed"] else (data_top - 16, ds)):  # parks, then back
        s, e = writeback(s, e, cur)
    if (s, e) != (ds, data_top):
        out["m14"] = (ds, data_top, s, e)
    end = e
    if lay["arena"]:
        s, e = writeback(s, e, end - 32)               # CONTEXTS_FIRST_ENTRY: END - 32 and back
        if s > ds or e < end:
            out["moved"].append(("dom_data", end - ds, s, e))
        s, e = writeback(s, e, s)
        if not s <= end - lay["arena"] < e:
            out["trap"].append(("arena split", end - lay["arena"], s, e))
        a = writeback(end - lay["arena"], e, end - lay["arena"])
        am = writeback(a[0], a[1], a[1] - 32)          # the context arena, its cursor moved
        if am[0] > end - lay["arena"] or am[1] < end:
            out["moved"].append(("arena", lay["arena"], am[0], am[1]))
        s, e = writeback(s, end - lay["arena"], s)     # sp = [base, END - A)
    s0 = s
    t1 = e
    g = 16
    if carve_align:
        x = (s ^ t1) >> 21
        g = 1
        while x:
            g <<= 1
            x >>= 1
        g = max(16, g)
        t1 &= ~(g - 1)
    table = lay["count"] * 16
    table = (table + g - 1) & ~(g - 1)
    sp_base, sp_top = s, e
    want_top = sp_top
    t1 -= table
    if not sp_base <= t1 < sp_top:                     # SPLIT raises ILLEGAL_OPERAND_VALUE (capstone_dyn_unit.anvil)
        out["trap"].append(("table", table, sp_base, sp_top))
    if out["trap"]:                                    # the core traps here; nothing after it happens
        out.update(stack=None, tot=tot, end=end, data_top=data_top)
        return out
    gs, ge = writeback(t1, sp_top, t1)
    sp_base, sp_top = writeback(sp_base, t1, sp_base)
    if sp_top != t1:
        out["inexact"] += 1
    if gs < t1 or ge > want_top:
        out["overlap"].append(("table", max(0, t1 - gs) + max(0, ge - want_top), gs, ge))
    out["gran"] = full_granule(t1 - sp_base)
    for i, (size, _al, _off) in enumerate(lay["recs"]):
        stor = max((size + 15) & ~15, 16)
        stor = (stor + g - 1) & ~(g - 1)
        want_top = t1
        t1 -= stor
        if not sp_base <= t1 < sp_top:
            out["trap"].append(("global[%d]" % i, size, sp_base, sp_top))
            out.update(stack=None, tot=tot, end=end, data_top=data_top)
            return out
        st = writeback(t1, sp_top, t1)
        sp_base, sp_top = writeback(sp_base, t1, sp_base)
        if sp_top != t1 or st[1] != want_top:
            out["inexact"] += 1
        if st[0] > t1 or st[1] < t1 + size:           # coverage; under cursorless the base is exact, so this is
            out["short"].append((i, size, st[0], st[1]))   # the old length check
        if st[0] < t1 or st[1] > want_top:
            out["overlap"].append(("global[%d]" % i, max(0, t1 - st[0]) + max(0, st[1] - want_top), st[0], st[1]))
        moved = writeback(st[0], st[1], st[0] + (8 if size > 8 else 1))
        if moved[0] > t1 or moved[1] < t1 + size:      # the zero-fill walker, or any pointer into the global
            out["moved"].append(("global[%d]" % i, size, moved[0], moved[1]))
        elif moved != st:
            out["widen"] += 1
            if FORMAT == "full":
                raise ValueError("--format full read WIDEN at base %#x global[%d]; the full scheme does not depend on "
                                 "the cursor, so the port is wrong" % (base, i))
    if sp_top > t1:
        out["overlap"].append(("stack", sp_top - t1, sp_base, sp_top))
    sm = writeback(sp_base, sp_top, t1)                # the glue hands sp over with `scc sp, sp, t1`: cursor AT t1
    if sm[0] > s0 or sm[1] < t1:
        out["moved"].append(("stack", t1 - s0, sm[0], sm[1]))
    out["stack"] = t1 - (ds + max(0, code_size - lay["gpoff"]))
    out.update(tot=tot, end=end, data_top=data_top)
    return out


def check_process(path, carve_align, cma_base, cma_size, show_base, arena=None, allow_overlap=False):
    name = path.split("/")[-1]
    lay = process_layout(path, arena)
    detected = lay["carve_align"]
    if carve_align is None:
        carve_align = detected
    how = "detected" if carve_align == detected else "FORCED, the code says %s" % ("on" if detected else "off")
    first = replay_process(lay, cma_base, carve_align)
    tot = first["tot"]
    step = min(tot, 1 << 20) if lay["managed"] else tot   # CMA aligns to 1 MiB; the buddy allocator to the block
    bases = list(range(cma_base, cma_base + cma_size - tot + 1, step))
    bad = []
    worst_stack = None
    widen = 0
    ov_n = ov_max = ov_bases = 0
    gran = 0
    for b in bases:
        r = replay_process(lay, b, carve_align)
        widen = max(widen, r["widen"])
        if r["stack"] is not None:                     # None: the replay stopped at a SPLIT TRAP
            worst_stack = r["stack"] if worst_stack is None else min(worst_stack, r["stack"])
        gran = max(gran, r["gran"])
        if r["overlap"]:
            ov_bases += 1
            ov_n = max(ov_n, len(r["overlap"]))
            ov_max = max(ov_max, max(o[1] for o in r["overlap"]))
        fault = (r["short"] or r["trap"] or r["moved"] or r["m14"] or (r["stack"] or 0) < lay["req_stack"]
                 or (FORMAT == "cursorless" and r["inexact"]))
        if fault or (r["overlap"] and FORMAT == "full" and not allow_overlap):
            bad.append((b, r))
    print("%-28s %s, format %s: code_len=%d domreq=%d arena=%d count=%d tot=%d carve-align=%s (%s); "
          "%d bases (%#x + k*%#x): %s" % (name, "process ABI" if lay["managed"] else "classic SDK",
                                          FORMAT + ("+renorm" if RENORM else ""),
                                          lay["code_len"], lay["req_data"], lay["arena"], lay["count"],
                                          tot, "on" if carve_align else "off", how, len(bases), cma_base, step,
                                          "OK" if not bad else "AT RISK at %d" % len(bad)))
    print("    worst stack above the globals template %d (declared %d); WIDEN up to %d storage capabilities"
          % (worst_stack, lay["req_stack"], widen))
    if FORMAT == "full":
        print("    OVERLAP at %d of %d bases: up to %d capabilities, up to %d bytes outside their own storage%s; the "
              "remainder's full-scheme granule at the first carve is up to %d bytes" % (
                  ov_bases, len(bases), ov_n, ov_max, " (allowed by --allow-overlap)" if ov_bases and allow_overlap
                  else "", gran))
    faults = [x for x in bad if x[1]["short"] or x[1]["trap"] or x[1]["moved"] or x[1]["m14"]]
    if faults and len(faults) < len(bad):
        print("    %d of the %d bases at risk FAULT; the rest only OVERLAP. The faulting ones first:" % (len(faults), len(bad)))
    for b, r in (faults + [x for x in bad if x not in faults])[:3]:
        report_base(lay, b, r, "")
    if show_base is not None:
        report_base(lay, show_base, replay_process(lay, show_base, carve_align), " (requested)")
    return bad


def report_base(lay, b, r, label):
    print("    base %#x%s: dom_data top %#x, %d inexact carve point(s), %d SHORT, %d SPLIT TRAP, %d SHORT ONCE MOVED%s%s"
          % (b, label, r["end"], r["inexact"], len(r["short"]), len(r["trap"]), len(r["moved"]),
             ", M-14: dom_data [%#x,%#x) became [%#x,%#x)" % r["m14"] if r["m14"] else "",
             ", stack %d below the declared %d" % (r["stack"], lay["req_stack"])
             if r["stack"] is not None and r["stack"] < lay["req_stack"] else ""))
    for i, size, lo, hi in r["short"][:3]:
        print("        global[%d] size %d got [%#x,%#x) = %d bytes" % (i, size, lo, hi, hi - lo))
    for what, size, lo, hi in r["trap"][:3]:
        print("        SPLIT TRAP carving %s (%d bytes): the remainder decodes as [%#x,%#x)" % (what, size, lo, hi))
    for what, size, lo, hi in r["moved"][:3]:
        print("        SHORT ONCE MOVED %s (%d bytes) re-encodes as [%#x,%#x)" % (what, size, lo, hi))
    if FORMAT == "full":
        for what, extra, lo, hi in sorted(r["overlap"], key=lambda o: -o[1])[:3]:
            print("        OVERLAP %s got [%#x,%#x), %d bytes outside its own storage" % (what, lo, hi, extra))


def check_legacy_full(name, count, recs, carve, tot, allow_overlap=False):
    """The old SDK region under --format full: carve downward from tot at base 0, each split's upper half taking the
    remainder's DECODED top, and classify by coverage and overlap as the process replay does."""
    sp_base, sp_top = 0, tot
    top = tot
    short, overlap = [], []
    gran = full_granule(tot - count * 16)
    for label, length in ([("cap-table", count * 16)] +
                          [("global[%d]" % i, max((s + 15) & ~15, 16)) for i, (s, _a, _b) in enumerate(recs)]):
        base = top - length
        st = writeback(base, sp_top, base)
        sp_base, sp_top = writeback(sp_base, base, sp_base)
        if st[0] > base or st[1] < top:
            short.append((label, length, st[0], st[1]))
        if st[0] < base or st[1] > top:
            overlap.append((label, max(0, base - st[0]) + max(0, st[1] - top), st[0], st[1]))
        top = base
    if sp_top > top:
        overlap.append(("stack", sp_top - top, sp_base, sp_top))
    bad = short + ([] if allow_overlap else overlap)
    print("%-28s format full" + ("+renorm" if RENORM else "") + ": count=%-5d carve=%-8d tot=%-9d %s; OVERLAP %d capabilities, up to %d bytes; the "
          "remainder's granule at the first carve %d" % (
              name, count, carve, tot, "OK" if not bad else "AT RISK", len(overlap),
              max([o[1] for o in overlap] or [0]), gran))
    for label, length, lo, hi in short[:3]:
        print("    SHORT %-12s len=%-8d got [%#x,%#x)" % (label, length, lo, hi))
    for label, extra, lo, hi in sorted(overlap, key=lambda o: -o[1])[:3]:
        print("    OVERLAP %-12s got [%#x,%#x), %d bytes outside its own storage" % (label, lo, hi, extra))
    return bad


def self_test():
    """The model's positive controls: each must FIRE, or the gate is unproven. Returns the number that did not."""
    global FORMAT, RENORM
    saved, saved_rn, bad = FORMAT, RENORM, 0
    try:
        RENORM = False
        FORMAT = "full"
        # the wrap: len 0x7ffe0, E = 6, rounded span 2**19 = 2**(E+13); decodes 256 KiB short (2026-10-13)
        got = writeback(0xadf2a000, 0xadfa9fe0, 0xadf2a000)
        if got != (0xadf2a000, 0xadf6a000):
            print("self-test FAIL: the full scheme's wrap did not fire: %s" % (tuple(map(hex, got)),)); bad += 1
        # the full scheme rounds an inexact top UP and does not depend on the cursor
        # [0x10010, 0x12018): len 0x2008, E = 1, granule 16, so the top must round up to 0x12020
        if writeback(0x10010, 0x12018, 0x10010) != (0x10010, 0x12020) or \
                writeback(0x10010, 0x12018, 0x11000) != (0x10010, 0x12020):
            print("self-test FAIL: the full scheme did not round the top up independently of the cursor"); bad += 1
        # the predicate, on a seeded sample
        rnd = random.Random(1)
        for _ in range(20000):
            k = rnd.randrange(13, 40)
            ln = (1 << k) - rnd.randrange(0, 1 << (k - 7)) if rnd.randrange(2) else rnd.randrange(4096, 1 << k)
            st = (rnd.randrange(0, 1 << 40) >> 4) << 4
            d = writeback(st, st + ln, st)
            g = full_granule(ln)
            span = -(-(st + ln) // g) * g - st // g * g
            if (d[0] > st or d[1] < st + ln) != (span >= g << 10):
                print("self-test FAIL: the wrap predicate disagrees at [%#x,%#x)" % (st, st + ln)); bad += 1
                break
        FORMAT = "cursorless"
        # the cursorless top truncation across a 2 MiB window (R-11's mechanism): E = 1 drops the top's bit 0
        if writeback(0x1f0000, 0x200011, 0x1f0000) != (0x1f0000, 0x200010):
            print("self-test FAIL: the cursorless truncation did not fire"); bad += 1
        # --renorm (2a3a87dfc): the wrap's example covers with E = 7 (its top rounded to 1 KiB), with the cursor moved,
        # and the predicate's sample all covers; the cursorless branch is untouched by it
        RENORM = True
        FORMAT = "full"
        if writeback(0xadf2a000, 0xadfa9fe0, 0xadf2a008) != (0xadf2a000, 0xadfaa000):
            print("self-test FAIL: --renorm did not cover the wrap's example"); bad += 1
        rnd = random.Random(2)
        for _ in range(20000):
            k = rnd.randrange(13, 40)
            ln = (1 << k) - rnd.randrange(0, 1 << (k - 7)) if rnd.randrange(2) else rnd.randrange(4096, 1 << k)
            st = (rnd.randrange(0, 1 << 40) >> 4) << 4
            d = writeback(st, st + ln, st + 8)
            if d[0] > st or d[1] < st + ln:
                print("self-test FAIL: --renorm left [%#x,%#x) short" % (st, st + ln)); bad += 1
                break
        FORMAT = "cursorless"
        if writeback(0x1f0000, 0x200011, 0x1f0000) != (0x1f0000, 0x200010):
            print("self-test FAIL: --renorm changed the cursorless branch"); bad += 1
        # the monitor's code region: 720,904 B of code, 40 KiB of data (a 1 MiB block) rounds the code capability out
        # past split_size under the full scheme, and stays exact under cursorless
        lay = dict(managed=True, code_len=720904, req_data=40 * 1024, req_stack=0, gpoff=0x60000, arena=0, count=0,
                   recs=[])
        for fmt, want in (("full", True), ("cursorless", False)):
            FORMAT = fmt
            o = dict(trap=[], moved=[], overlap=[])
            base, tot, code_size = 0xac000000, 1 << 20, (720904 + 15) & ~15
            hb = (tot - code_size - 1536).bit_length() - 1        # the replay's own geometry formulas
            gran = 1 << ((hb - 12 if hb > 12 else 0) + 3)
            monitor_splits(lay, base, tot, code_size, (code_size + gran - 1) & ~(gran - 1),
                           (1536 + gran - 1) & ~(gran - 1), (base + tot - 1024) & ~(gran - 1), o)
            got = any(x[0].startswith("monitor code") for x in o["overlap"])
            if got != want or o["trap"] or o["moved"]:
                print("self-test FAIL: the monitor code-region control under %s read %s" % (fmt, o)); bad += 1
    finally:
        FORMAT, RENORM = saved, saved_rn
    print("self-test: %s" % ("every control fires" if not bad else "%d control(s) did not fire" % bad))
    return bad


def check(path, allow_overlap=False):
    name = path.split("/")[-1]
    off = find_initdesc(path)
    if off is None:
        print("%-28s no .capstone_gp_initdesc (not a gp-captable domain)" % name)
        return []

    img = open(path, "rb").read()
    _built, count = struct.unpack_from("<QQ", img, off)
    recs = [struct.unpack_from("<QQq", img, off + 32 + 24 * i)
            for i in range(count)]

    carve = count * 16 + sum(max((s + 15) & ~15, 16) for s, _a, _b in recs)

    bad = []
    # The domain is allocated as a power-of-two page run, so model it as a region
    # based at 0 of that size; base 0 is the worst case for window alignment only
    # if the size exceeds one window, which is exactly what we are testing for.
    pages = (len(img) + 65536 - 1) // 4096 + 1
    order = max(0, (pages - 1).bit_length())
    tot = (1 << order) * 4096

    if FORMAT == "full":
        return check_legacy_full(name, count, recs, carve, tot, allow_overlap)

    if tot > WINDOW:
        # Interior splits now straddle a 2 MiB boundary. Replay the carve at a
        # WINDOW-aligned base and report any capability whose top truncates.
        top = tot
        for label, length in ([("cap-table", count * 16)] +
                              [("global[%d]" % i, max((s + 15) & ~15, 16))
                               for i, (s, _a, _b) in enumerate(recs)]):
            base = top - length
            ok, e = cursorless_top_exact(base, top)
            if not ok:
                bad.append((label, length, base, top, e,
                            top - (top >> e << e)))
            top = base

    status = ("OK" if not bad else "TOP-TRUNCATION x%d" % len(bad))
    print("%-28s count=%-5d carve=%-8d tot=%-9d %s" %
          (name, count, carve, tot, status))
    if tot > WINDOW and not bad:
        print("    note: domain is %d bytes, past the 2 MiB window -- exact only "
              "because every top happens to be aligned; treat as fragile" % tot)
    for label, length, base, top, e, lost in bad:
        print("    %-12s len=%-8d [%#x,%#x) straddles a 2 MiB window: E=%d, "
              "top loses %d bytes" % (label, length, base, top, e, lost))
    return bad


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="check gp-captable domains against CVA6's bounds compression")
    ap.add_argument("domains", nargs="*")
    ap.add_argument("--self-test", action="store_true", help="run the model's positive controls first; exit 1 if any "
                                                              "does not fire")
    ap.add_argument("--carve-align", choices=("auto", "on", "off"), default="auto",
                    help="process ABI: whether the glue has CAPSTONE_GLUE_CARVE_ALIGN; auto (default) reads it from "
                         "the image's code, on/off override that")
    ap.add_argument("--cma-base", type=lambda v: int(v, 0), default=0xac000000,
                    help="process ABI: the CMA window's base (default: the FPGA board's 0xac000000)")
    ap.add_argument("--cma-size", type=lambda v: int(v, 0), default=256 << 20)
    ap.add_argument("--arena", type=lambda v: int(v, 0), default=None,
                    help="process ABI: the context arena's size, when the glue's own sequence is not recognised")
    ap.add_argument("--base", type=lambda v: int(v, 0), default=None,
                    help="process ABI: also report this one base in detail (e.g. a board's DBAS)")
    ap.add_argument("--format", choices=("cursorless", "full"), default="cursorless",
                    help="the bitstream's bounds encoding: cursorless (default; 0568f93a9 and earlier) or full "
                         "(capstone-sublet: one scheme, no cursorless branch)")
    ap.add_argument("--renorm", action="store_true",
                    help="R-54's fix in the encoder (capstone-ariane 2a3a87dfc): the full branch takes E + 1 when the "
                         "rounded span reaches 1024 granules")
    ap.add_argument("--allow-overlap", action="store_true",
                    help="--format full: report OVERLAP without failing")
    a = ap.parse_args()
    FORMAT = a.format
    RENORM = a.renorm
    failed = 0
    if a.self_test and self_test():
        sys.exit(1)
    if not a.domains and not a.self_test:
        ap.error("no domains given")
    for p in a.domains:
        try:
            secs = read_elf(p)[3]
            if ".capstone_domreq" in secs or (FORMAT == "full" and ".capstone_gp_initdesc" in secs):
                bad = check_process(p, {"auto": None, "on": True, "off": False}[a.carve_align], a.cma_base, a.cma_size,
                                    a.base, a.arena, a.allow_overlap)
            else:
                bad = check(p, a.allow_overlap)
            if bad:
                failed += 1
        except Exception as exc:                       # noqa: BLE001
            print("%-28s ERROR %s" % (p.split("/")[-1], exc))
            failed += 1
    sys.exit(1 if failed else 0)
