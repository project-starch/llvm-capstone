"""The FPGA monitor's boot-time capability carve (sbi_capstone_init.S + cap_env_init/split_out_cap in
sbi_capstone_dom.c / sbi_capstone.c, opensbi-custom as built for fw_2f49d5b257f7), replayed through check-repr's
encoder. A capability is (start, end, cursor) as the register holds it: bounds DECODED, cursor exact. SPLIT per
capstone_dyn_unit.anvil func SPLIT: lower = [rs1.start, at) with cursor = rs1.start; upper = [at, rs1.end), cursor at.

A displaced cursor matters only where it is dereferenced (ISSUES R-55, claim-auditor 2026-10-15): mtime's is, by
handle_exception's rdtime emulation; dom_init resets the boot stack's (scc), cap_env_init the interrupt stack's
(__setcursor), and this silicon never reads the seal (cap_cih tied to 0). The region table printed last is the
other consequence: under the full scheme the region covering Linux begins at 0x80000000, the monitor's own text."""
import importlib.util, sys
spec = importlib.util.spec_from_file_location("cr", __import__("os").path.join(__import__("os").path.dirname(__file__), "check-repr.py"))
cr = importlib.util.module_from_spec(spec); spec.loader.exec_module(cr)

def cap(start, end, cursor):
    s, e = cr.writeback(start, end, cursor)
    return (s, e, cursor)

def split(c, at):
    assert c[0] <= at < c[1], f"SPLIT TRAP at {at:#x} outside [{c[0]:#x},{c[1]:#x})"
    return cap(c[0], at, c[0]), cap(at, c[1], at)

def setcursor(c, cur):
    return cap(c[0], c[1], cur)

def run(fmt, renorm, patched):
    cr.FORMAT, cr.RENORM = fmt, renorm
    out = []
    a0 = cap(0, 0x80014000, 0)                                  # CAPENTER rs1_o = [0, _cap_text_start)
    a1 = cap(0x80030000, 1 << 40, 0x80030000)                   # CAPENTER rd = [_cap_text_end, MEM_END)
    a1, a2 = split(a1, 0x80080380)                              # SPLIT(a2, a1, dom_stack)
    a2, a3 = split(a2, 0x80082380)                              # SPLIT(a3, a2, dom_stack_end)
    out.append(("boot stack (cscratch)", a2, (0x80080380, 0x80082380)))
    regions = [a0, a1, a3]
    def split_out(base, ln):
        for i, m in enumerate(regions):
            if base >= m[0] and base + ln <= m[1]:
                break
        else:
            return "SPLA (no region covers)"
        if base == m[0]:
            region = m
        else:
            m, region = split(m, base)
        if base + ln == m[1] and base == m[0]:
            return "EXACT FIT (hole)"
        region, mr = split(region, base + ln)
        if base == m[0]:
            regions[i] = mr
        else:
            regions[i] = m
            regions.append(mr)
        return setcursor(region, base) if patched else region
    for name, b, l in (("uart", 0x10000000, 0x100), ("mtime", 0x200bff8, 8), ("mtimecmp", 0x2004000, 8),
                       ("int stack", 0x80082380, 0x8000), ("int seal", 0x8008a380, 0x400),
                       ("int code", 0x80030000, 0x10000)):
        out.append((name, split_out(b, l), (b, b + l)))
    out.append(("REGIONS", regions, None))
    return out

for fmt, rn, p in (("cursorless", False, False), ("full", True, False), ("full", True, True)):
    print(f"== {fmt}{' +renorm' if rn else ''}{' PATCHED split_out_cap (cursor := base)' if p else ''}")
    for name, c, want in run(fmt, rn, p):
        if name == "REGIONS":
            for i, r in enumerate(c):
                print(f"   region {i}: [{r[0]:#x}, {r[1]:#x})" + ("   <- covers Linux 0x80200000" if r[0] <= 0x80200000 < r[1] else ""))
            continue
        if isinstance(c, str):
            print(f"   {name:22s} {c}"); continue
        flag = []
        if c[2] != want[0]: flag.append(f"CURSOR {c[2]:#x} != base {want[0]:#x}")
        if c[0] > want[0] or c[1] < want[1]: flag.append("SHORT")
        elif (c[0], c[1]) != want: flag.append(f"OVERLAP {(want[0]-c[0]) + (c[1]-want[1]):#x} B")
        print(f"   {name:22s} [{c[0]:#x}, {c[1]:#x}) cursor {c[2]:#x}   {'; '.join(flag) or 'exact'}")
