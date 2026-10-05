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
    data_top - 16 and - 32 (each a lossy re-encode);
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

Exit status 1 if any domain is at risk.
"""
import struct
import subprocess
import sys

WINDOW_BITS = 21                      # ariane_pkg.sv floors the scan at bit 20
WINDOW = 1 << WINDOW_BITS


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
    """compress_bounds, ariane_pkg.sv:793-851 at 776d9d859 (a literal port)."""
    if start == cursor:
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
    return dict(carve_align=glue_carve_align(img, secs),
                code_len=hi - lo, gpoff=ini[3] - lo, req_data=req_data, req_stack=req_stack,
                count=count, recs=recs, arena=arena or 0)


def replay_process(lay, base, carve_align):
    """One base: the module's block, the monitor's managed split, the glue's carve."""
    out = dict(short=[], inexact=0, widen=0, m14=None)
    total = lay["code_len"] + lay["req_data"] + 8 * 1024 + 1024
    total = (total + 4095) & ~4095
    tot = 1 << (total - 1).bit_length()
    code_size = (lay["code_len"] + 15) & ~15
    repr_len = tot - code_size - 1536
    hb = repr_len.bit_length() - 1
    gran = 1 << ((hb - 12 if hb > 12 else 0) + 3)
    split_size = (code_size + gran - 1) & ~(gran - 1)
    data_off = (1536 + gran - 1) & ~(gran - 1)
    data_top = ((base + tot - 1024) & ~(gran - 1))
    ds = base + split_size + data_off
    s, e = writeback(ds, data_top, ds)                 # the dom_data split (cursorless)
    for cur in (data_top - 16, data_top - 32, ds):     # gp and code parks, cursor back
        s, e = writeback(s, e, cur)
    if (s, e) != (ds, data_top):
        out["m14"] = (ds, data_top, s, e)
    end = e
    if lay["arena"]:
        s, e = writeback(s, e, end - 32)               # CONTEXTS_FIRST_ENTRY: END - 32 and back
        s, e = writeback(s, e, s)
        s, e = writeback(s, end - lay["arena"], s)     # sp = [base, END - A)
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
    t1 -= table
    _gs, _ge = writeback(t1, sp_top, t1)
    sp_base, sp_top = writeback(sp_base, t1, sp_base)
    if sp_top != t1:
        out["inexact"] += 1
    for i, (size, _al, _off) in enumerate(lay["recs"]):
        stor = max((size + 15) & ~15, 16)
        stor = (stor + g - 1) & ~(g - 1)
        want_top = t1
        t1 -= stor
        st = writeback(t1, sp_top, t1)
        sp_base, sp_top = writeback(sp_base, t1, sp_base)
        if sp_top != t1 or st[1] != want_top:
            out["inexact"] += 1
        if st[1] - st[0] < size:
            out["short"].append((i, size, st[0], st[1]))
        moved = writeback(st[0], st[1], st[0] + (8 if size > 8 else 1))
        if moved != st:
            out["widen"] += 1
    out["stack"] = t1 - (ds + max(0, code_size - lay["gpoff"]))
    out.update(tot=tot, end=end, data_top=data_top)
    return out


def check_process(path, carve_align, cma_base, cma_size, show_base, arena=None):
    name = path.split("/")[-1]
    lay = process_layout(path, arena)
    detected = lay["carve_align"]
    if carve_align is None:
        carve_align = detected
    how = "detected" if carve_align == detected else "FORCED, the code says %s" % ("on" if detected else "off")
    first = replay_process(lay, cma_base, carve_align)
    tot = first["tot"]
    step = min(tot, 1 << 20)
    bases = list(range(cma_base, cma_base + cma_size - tot + 1, step))
    bad = []
    worst_stack = None
    widen = 0
    for b in bases:
        r = replay_process(lay, b, carve_align)
        widen = max(widen, r["widen"])
        worst_stack = r["stack"] if worst_stack is None else min(worst_stack, r["stack"])
        if r["short"] or r["inexact"] or r["m14"] or r["stack"] < lay["req_stack"]:
            bad.append((b, r))
    print("%-28s process ABI: code_len=%d domreq=%d arena=%d count=%d tot=%d carve-align=%s (%s); %d bases "
          "(%#x + k*%#x): %s" % (name, lay["code_len"], lay["req_data"], lay["arena"], lay["count"], tot,
                                 "on" if carve_align else "off", how, len(bases), cma_base, step,
                                 "OK" if not bad else "AT RISK at %d" % len(bad)))
    print("    worst stack above the globals template %d (declared %d); WIDEN up to %d storage capabilities"
          % (worst_stack, lay["req_stack"], widen))
    for b, r in bad[:3]:
        print("    base %#x: dom_data top %#x, %d inexact carve point(s), %d SHORT%s%s" % (
            b, r["end"], r["inexact"], len(r["short"]),
            ", M-14: dom_data [%#x,%#x) became [%#x,%#x)" % r["m14"] if r["m14"] else "",
            ", stack %d below the declared %d" % (r["stack"], lay["req_stack"]) if r["stack"] < lay["req_stack"] else ""))
        for i, size, lo, hi in r["short"][:3]:
            print("        global[%d] size %d got [%#x,%#x) = %d bytes" % (i, size, lo, hi, hi - lo))
    if show_base is not None:
        r = replay_process(lay, show_base, carve_align)
        print("    base %#x (requested): dom_data top %#x, %d inexact, %d SHORT" % (show_base, r["end"], r["inexact"],
                                                                       len(r["short"])))
        for i, size, lo, hi in r["short"][:3]:
            print("        global[%d] size %d got [%#x,%#x) = %d bytes" % (i, size, lo, hi, hi - lo))
    return bad


def check(path):
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
    ap.add_argument("domains", nargs="+")
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
    a = ap.parse_args()
    failed = 0
    for p in a.domains:
        try:
            secs = read_elf(p)[3]
            if ".capstone_domreq" in secs:
                bad = check_process(p, {"auto": None, "on": True, "off": False}[a.carve_align], a.cma_base, a.cma_size,
                                    a.base, a.arena)
            else:
                bad = check(p)
            if bad:
                failed += 1
        except Exception as exc:                       # noqa: BLE001
            print("%-28s ERROR %s" % (p.split("/")[-1], exc))
            failed += 1
    sys.exit(1 if failed else 0)
