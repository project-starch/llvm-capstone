#!/usr/bin/env python3
"""Turn a capability fault in a run log into a source line.

    fault-locate.py <run-log> <image.dom> [options]

WHY THIS EXISTS. Every fault so far has been located by hand: read the anchor
rung's return value, subtract the symbol address to get the load base, subtract
that from the pc, run llvm-nm, read the disassembly, and guess which struct field
an offset names. That is twenty to forty minutes per fault and it has produced
wrong readings -- a stack_keep that was inferred and then measured as something
else, and a whole root cause assigned to the wrong subsystem.

THE PART THAT IS NOT ARITHMETIC. QEMU reports `env->pc`, which is only written at
translation-block boundaries, so the reported pc is usually the block entry and NOT
the faulting instruction. The monitor also reports rs1, the immediate and (for a
bounds fault) the access size, and those identify the instruction exactly. This
script does that disambiguation instead of leaving it to the reader, and it
REFUSES rather than guessing when nothing in the window matches.

Line numbers need an image built with line tables (MRUBY_DEBUG_INFO=1 for the mruby
port). Debug sections are non-alloc, so such an image can be a twin of the one that
actually ran -- CHECK that .text matches rather than assuming it, e.g.

    llvm-objcopy -O binary --only-section=.text <image> - | md5sum

Self-test:  fault-locate.py --self-test <image.dom>
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

MASK = 0xFFFFFFFF

# x0..x31 in ABI order; the monitor prints register NUMBERS, llvm-objdump prints names.
REG_NAMES = (
    ["zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2"]
    + ["s0", "s1"] + [f"a{i}" for i in range(8)]
    + [f"s{i}" for i in range(2, 12)] + [f"t{i}" for i in range(3, 7)]
)
# s0 is also fp in some printers; accept both for the same number.
REG_ALIASES = {"s0": {"s0", "fp"}}
assert len(REG_NAMES) == 32 and REG_NAMES[10] == "a0" and REG_NAMES[24] == "s8", \
    "REG_NAMES is not the RISC-V ABI order; a wrong name would match the wrong instruction"

# mnemonic -> (bytes touched, is_store). Capability accesses are 16 bytes.
ACCESS = {
    "lb": (1, False), "lbu": (1, False), "sb": (1, True),
    "lh": (2, False), "lhu": (2, False), "sh": (2, True),
    "lw": (4, False), "lwu": (4, False), "sw": (4, True),
    "ld": (8, False), "sd": (8, True),
    "ldc": (16, False), "stc": (16, True),
}
# Deliberately NO flw/fsw/fld/fsd: trans_rvf/rvd never reach gen_helper_*_with_cap,
# so a float access cannot raise the faults this script parses. Listing them would
# only add candidates that can never be the answer.

FAULTS = [
    ("oob", re.compile(
        r"Cap mem access OOB: pc = (?P<pc>[0-9a-fA-F]+), rs1 = x(?P<rs1>\d+), "
        r"cursor = (?P<cursor>[0-9a-fA-F]+), imm = (?P<imm>-?\d+), addr = (?P<addr>[0-9a-fA-F]+), "
        r"size = (?P<size>\d+), bounds = \((?P<lo>[0-9a-fA-F]+), (?P<hi>[0-9a-fA-F]+)\)")),
    # `value`/`value_hi` and the revoked line's bounds are OPTIONAL so this reads
    # logs from before the monitor was taught to print them.
    ("untagged", re.compile(
        r"Cap mem access requires capability: pc = (?P<pc>[0-9a-fA-F]+), rs1 = x(?P<rs1>\d+), "
        r"imm = (?P<imm>-?\d+)"
        r"(?:, value = (?P<value>[0-9a-fA-F]+), value_hi = (?P<value_hi>[0-9a-fA-F]+))?")),
    ("revoked", re.compile(
        r"Cap mem access on revoked capability: pc = (?P<pc>[0-9a-fA-F]+), rs1 = x(?P<rs1>\d+), "
        r"imm = (?P<imm>-?\d+)"
        r"(?:, cursor = (?P<cursor>[0-9a-fA-F]+), "
        r"bounds = \((?P<lo>[0-9a-fA-F]+), (?P<hi>[0-9a-fA-F]+)\))?")),
    ("halt", re.compile(
        r"domain halted by capability fault: cause = (?P<cause>\d+), pc = 0x(?P<pc>[0-9a-fA-F]+)")),
]

ANCHOR = re.compile(r"Called dom \(1-th time\) retval = (?P<v>\d+)")


def llvm_tool(name):
    """Resolve an llvm tool, preferring the build this project is configured against."""
    for base in (os.environ.get("CAPSTONE_LLVM_BIN"),
                 os.path.join(os.environ.get("CAPSTONE_LLVM_BUILD_DIR", ""), "bin")):
        if base:
            p = os.path.join(base, name)
            if os.path.isfile(p) and os.access(p, os.X_OK):
                return p
    found = shutil.which(name)
    if not found:
        sys.exit(f"{name} not found -- set CAPSTONE_LLVM_BIN or put it on PATH")
    return found


def die(msg, code=2):
    """No-data is an ERROR. A tool that prints an empty result reads like a finding."""
    print(f"fault-locate: REFUSED -- {msg}", file=sys.stderr)
    sys.exit(code)


def read_log(path):
    with open(path, "rb") as f:
        return f.read().decode("utf-8", "replace")


def find_fault(text):
    """The last fault, preferring the line that names the operand.

    The monitor's `domain halted by capability fault` line always FOLLOWS the
    `[CAPSTONE] Cap mem access ...` line for the same event, and carries no rs1 or
    immediate -- so taking the last match verbatim throws away the only fields that
    identify the instruction. Fall back to the halt line only when nothing richer
    precedes it.
    """
    hits = []
    for kind, rx in FAULTS:
        for m in rx.finditer(text):
            hits.append((m.start(), kind, m))
    if not hits:
        die("no capability-fault line in the log. Looked for: "
            + ", ".join(k for k, _ in FAULTS))
    hits.sort(key=lambda h: h[0])
    pos, kind, m = hits[-1]
    if kind == "halt":
        richer = [h for h in hits if h[1] != "halt" and pos - h[0] < 2000]
        if richer:
            return richer[-1][1], richer[-1][2], m
    return kind, m, (m if kind == "halt" else None)


def find_anchor(text, upto):
    ms = list(ANCHOR.finditer(text, 0, upto))
    if not ms:
        die("no anchor line before the fault. Expected a rung-0 "
            "'Called dom (1-th time) retval = N' carrying &domain_main. "
            "Pass --load-base if this domain has no anchor rung.")
    return int(ms[-1].group("v"))


def symbol_addr(nm, image, name):
    out = subprocess.run([nm, "-n", image], capture_output=True, text=True).stdout
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2] == name:
            return int(parts[0], 16)
    die(f"symbol {name!r} not found in {image}")


def text_bounds(readelf, image):
    """[start, end) of the only executable section, from the section table.

    The first version of this guard used the symbol range plus 64 KiB of slack, which
    in these images reaches into .capstone_gp_initdesc -- so a load base wrong by a
    page still produced a full, confident answer. The section table is exact.
    """
    out = subprocess.run([readelf, "-SW", image], capture_output=True, text=True).stdout
    for line in out.splitlines():
        m = re.search(r"\]\s+(\S+)\s+PROGBITS\s+([0-9a-f]+)\s+[0-9a-f]+\s+([0-9a-f]+).*\bAX\b", line)
        if m:
            base, size = int(m.group(2), 16), int(m.group(3), 16)
            return base, base + size
    return None


def text_symbols(nm, image):
    out = subprocess.run([nm, "-n", image], capture_output=True, text=True).stdout
    syms = []
    for line in out.splitlines():
        parts = line.split()
        # `.L` assembler labels are NOT functions. Offering one as the enclosing
        # symbol is how a fault in mrb_vm_run was once reported as living in a data
        # symbol, and the wrong reading survived a whole session.
        if len(parts) == 3 and parts[1] in ("T", "t") and not parts[2].startswith(".L"):
            syms.append((int(parts[0], 16), parts[2]))
    return syms


def enclosing(syms, addr):
    hit = None
    for a, n in syms:
        if a <= addr:
            hit = (a, n)
        else:
            break
    return hit


DISASM = re.compile(
    r"^\s*(?P<addr>[0-9a-f]+):\s+(?:[0-9a-f]{2} )+\s*"
    r"(?P<mn>[a-z][a-z0-9._]*)\s+(?P<ops>.*?)\s*$")
MEMOP = re.compile(r"^(?P<rd>\S+),\s*(?P<imm>-?0x[0-9a-f]+|-?\d+)\((?P<base>\w+)\)$")


def disassemble(objdump, image, start, stop):
    out = subprocess.run(
        [objdump, "-d", f"--start-address={hex(start)}", f"--stop-address={hex(stop)}", image],
        capture_output=True, text=True).stdout
    rows = []
    for line in out.splitlines():
        m = DISASM.match(line)
        if m:
            rows.append((int(m.group("addr"), 16), m.group("mn"), m.group("ops"), line.rstrip()))
    return rows


def match_instruction(rows, reg, imm, size, is_store):
    """First instruction addressing `reg` with `imm`, honouring size/direction if known."""
    hits = []
    want = REG_ALIASES.get(reg, {reg})
    for addr, mn, ops, raw in rows:
        m = MEMOP.match(ops)
        if not m or m.group("base") not in want:
            continue
        if int(m.group("imm"), 0) != imm:
            continue
        acc = ACCESS.get(mn)
        if acc and size is not None and acc[0] != size:
            continue
        if acc and is_store is not None and acc[1] != is_store:
            continue
        hits.append((addr, raw))
    return hits


def locate(args, text=None):
    nm = llvm_tool("llvm-nm")
    objdump = llvm_tool("llvm-objdump")
    symbolizer = llvm_tool("llvm-symbolizer")

    text = text if text is not None else read_log(args.log)
    kind, m, halt = find_fault(text)
    g = m.groupdict()
    pc = int(g["pc"], 16) & MASK

    if args.load_base is not None:
        base = args.load_base & MASK
    else:
        anchor = find_anchor(text, m.start())
        base = (anchor - symbol_addr(nm, args.image, args.anchor_symbol)) & MASK
    off = (pc - base) & MASK

    print(f"fault kind      {kind}"
          + (f"   (monitor cause {halt.group('cause')})" if halt is not None and kind != "halt" else ""))
    print(f"reported pc     {hex(pc)}")
    print(f"load base       {hex(base)}")
    print(f"image offset    {hex(off)}")

    syms = text_symbols(nm, args.image)
    if not syms:
        die(f"no text symbols in {args.image}; is it stripped?")
    enc = enclosing(syms, off)
    bounds = text_bounds(llvm_tool("llvm-readelf"), args.image)
    if bounds is None:
        die(f"{args.image} has no executable PROGBITS section; cannot bound the search")
    if not (bounds[0] <= off < bounds[1]) or enc is None:
        die(f"offset {hex(off)} lies outside the image's executable section "
            f"([{hex(bounds[0])}, {hex(bounds[1])})). The load base is wrong: check "
            f"that the anchor rung really returns &{args.anchor_symbol}, or pass "
            f"--load-base. Not disassembling whatever happens to be there.")
    print(f"reported pc in  {enc[1]} + {hex(off - enc[0])}")

    if kind == "halt":
        print("\nThis is the monitor's halt line, which carries no rs1/imm, so the "
              "instruction cannot be identified from it. Find the [CAPSTONE] line "
              "above it in the log for that.")
        show_source(symbolizer, args.image, off, "reported pc")
        return 0

    reg = REG_NAMES[int(g["rs1"])]
    imm = int(g["imm"])
    size = int(g["size"]) if g.get("size") else None
    # The monitor never reports load-vs-store, so direction is not a filter.
    is_store = None
    print(f"operand         rs1 = x{g['rs1']} ({reg}), imm = {imm}"
          + (f", size = {size}" if size else ""))
    if g.get("value"):
        v = int(g["value"], 16)
        hi = int(g["value_hi"], 16) if g.get("value_hi") else 0
        print(f"operand value   0x{v:x}" + (f" (high half 0x{hi:x})" if hi else "")
              + "   untagged word used as a base")
    elif kind == "untagged":
        print("operand value   NOT REPORTED -- this log predates the monitor printing it; "
              "re-run to get the word itself")
    if kind == "revoked" and g.get("lo"):
        lo, hi = int(g["lo"], 16), int(g["hi"], 16)
        print(f"bounds          [{hex(lo)}, {hex(hi)})  length {hi - lo}")
        print(f"cursor          {hex(int(g['cursor'], 16))}")
    if kind == "oob":
        lo, hi = int(g["lo"], 16), int(g["hi"], 16)
        print(f"bounds          [{hex(lo)}, {hex(hi)})  length {hi - lo}")
        print(f"cursor          {hex(int(g['cursor'], 16))}  "
              f"({int(g['cursor'], 16) - lo} past base)")

    rows = disassemble(objdump, args.image, off, off + args.window)
    if not rows:
        die(f"nothing disassembles at {hex(off)}; the load base is probably wrong")

    # THE REPORTED pc IS EXACT WHENEVER THE MONITOR RESTORED IT. Older monitors did
    # not: `_helper_access_with_cap` was static, so its GETPC() named QEMU's own text
    # rather than the TCG buffer, cpu_restore_state() failed silently, and env->pc kept
    # the translation-block entry. Check the instruction AT the pc first; only fall
    # back to scanning when it does not match, and say which happened.
    at_pc = match_instruction(rows[:1], reg, imm, size, is_store)
    if at_pc:
        faddr, raw = at_pc[0]
        print(f"\nFAULTING INSTRUCTION at {hex(faddr)}   (the reported pc is exact)")
    else:
        hits = match_instruction(rows, reg, imm, size, is_store)
        if not hits:
            die(f"no instruction in [{hex(off)}, {hex(off + args.window)}) addresses "
                f"{reg} with immediate {imm}"
                + (f" at size {size}" if size else "")
                + ". Either the load base is wrong or the window is too small "
                  "(--window). Not guessing.")
        if len(hits) > 1:
            # AMBIGUOUS IS NOT LOCATED. This backend emits adjacent identical loads
            # (-capstone-double-ldc makes every one of them a pair), and a third of all
            # memory instructions in a real image have a same-(base, imm, size) twin
            # within the window. Naming the first would be a confident wrong answer,
            # and a wrong function costs a session -- so list them and refuse.
            listing = "\n".join(f"    {r.strip()}" for _a, r in hits)
            die(f"AMBIGUOUS: {len(hits)} instructions in [{hex(off)}, "
                f"{hex(off + args.window)}) address {reg} with immediate {imm}"
                + (f" at size {size}" if size else "")
                + f", and the reported pc is not one of them:\n{listing}\n"
                  "  The monitor did not restore the exact pc for this run. Re-run on a\n"
                  "  QEMU that threads the caller's return address into\n"
                  "  _helper_access_with_cap, or narrow --window if you know the block.")
        faddr, raw = hits[0]
        print(f"\nFAULTING INSTRUCTION at {hex(faddr)}   "
              f"(reported pc was STALE by {faddr - off} bytes; scanned forward)")
    print(f"  {raw.strip()}")
    fenc = enclosing(syms, faddr)
    if fenc:
        print(f"  in {fenc[1]} + {hex(faddr - fenc[0])}")
    show_source(symbolizer, args.image, faddr, "faulting instruction")

    print("\ncontext")
    lo = max(0, faddr - 0x20)
    for a, _mn, _ops, r in disassemble(objdump, args.image, lo, faddr + 0x20):
        print(("> " if a == faddr else "  ") + r.strip())
    return 0


def show_source(symbolizer, image, addr, what):
    out = subprocess.run(
        [symbolizer, f"--obj={image}", "--functions=linkage", "--inlines", hex(addr)],
        capture_output=True, text=True).stdout.splitlines()
    out = [l for l in (x.strip() for x in out) if l]
    pairs = [(out[i], out[i + 1]) for i in range(0, len(out) - 1, 2)]
    useful = [(f, l) for f, l in pairs if not l.endswith(":0:0") and l != "??:0:0"]
    if not useful:
        print(f"\nno line table for the {what}. Rebuild the image with line tables "
              "(MRUBY_DEBUG_INFO=1) and check .text still matches.")
        return
    print(f"\nSOURCE ({what}, innermost first)")
    for f, l in useful:
        print(f"  {f}\n      {l}")


def self_test(image):
    """Four controls, built out of the image itself.

    The first version of this was REFUTED as a control and the way it failed is worth
    keeping. Its negative case built the impossible immediate out of the SAME window
    locate() was about to scan, so a refusal was guaranteed by construction: it proved
    the die() plumbing worked and nothing else. Its positive case explicitly REJECTED
    any candidate that had a nearer twin -- i.e. it selected away from the exact failure
    mode the scanner has -- and exercised 16 bytes of a 256-byte window.

    So: an exact pc, a stale pc at a real distance, an AMBIGUOUS pair that must be
    refused, and an immediate proven absent from the scanned window.
    """
    objdump = llvm_tool("llvm-objdump")
    nm = llvm_tool("llvm-nm")
    syms = text_symbols(nm, image)
    if not syms:
        die("self-test: no text symbols in the image")

    rows = []
    for a, _n in syms[len(syms) // 3:]:
        rows = disassemble(objdump, image, a, a + 0x4000)
        if len(rows) > 400:
            break
    if len(rows) < 400:
        die("self-test: could not disassemble a large enough region")

    base = 0x1000000

    class A:
        pass
    a = A()
    a.image, a.anchor_symbol, a.window, a.log, a.load_base = image, "domain_main", 0x100, None, base

    def synth(pc_off, regnum, imm, size):
        return (f"Called dom (1-th time) retval = {base + 0x1000}\n"
                f"[CAPSTONE] Cap mem access OOB: pc = {pc_off + base:x}, rs1 = x{regnum}, "
                f"cursor = 0, imm = {imm}, addr = 0, size = {size}, bounds = (0, 10)\n")

    import io
    results = []

    def run(label, text, want_exit, want_in_output=None):
        buf = io.StringIO()
        so, se = sys.stdout, sys.stderr
        code = 0
        try:
            sys.stdout = sys.stderr = buf
            locate(a, text=text)
        except SystemExit as e:
            code = e.code or 0
        except BaseException as e:                      # noqa: BLE001
            code = -1
            buf.write(f"unexpected {type(e).__name__}: {e}\n")
        finally:
            # Without this, ANY error escapes with both streams still redirected and
            # the failure is reported into a buffer nobody prints. That happened.
            sys.stdout, sys.stderr = so, se
        out = buf.getvalue()
        ok = code == want_exit and (want_in_output is None or want_in_output in out)
        results.append(ok)
        print(f"   {'PASS' if ok else 'FAIL'}: {label}")
        if not ok:
            print(f"      wanted exit {want_exit}"
                  + (f" and {want_in_output!r}" if want_in_output else "")
                  + f", got exit {code}")
            print("      " + "\n      ".join(out.strip().splitlines()[:12]))
        return out

    # --- pick a memory instruction with NO same-(base, imm, size) twin in the window,
    #     so "the scan found it" is unambiguous.
    uniq = None
    for i, (addr, mn, ops, _raw) in enumerate(rows):
        m = MEMOP.match(ops)
        if not (m and mn in ACCESS):
            continue
        reg, imm = m.group("base"), int(m.group("imm"), 0)
        if reg not in REG_NAMES:
            continue
        win = [r for r in rows if addr - 0x60 <= r[0] < addr + 0xa0]
        if len(match_instruction(win, reg, imm, ACCESS[mn][0], None)) == 1:
            # need >= 0x40 of preceding instructions for a stale-pc case worth testing
            earlier = [r for r in rows if addr - 0x60 <= r[0] < addr]
            if len(earlier) >= 8:
                uniq = (addr, mn, reg, imm, earlier[0][0])
                break
    if uniq is None:
        die("self-test: no unambiguous memory instruction found to build a control from")
    faddr, mn, reg, imm, far_pc = uniq
    size = ACCESS[mn][0]
    regnum = REG_NAMES.index(reg)

    print("== 1 positive: an EXACT pc must be taken as-is")
    run(f"exact pc {hex(faddr)} ({mn} {imm}({reg}))",
        synth(faddr, regnum, imm, size), 0, "the reported pc is exact")

    print(f"== 2 positive: a STALE pc {faddr - far_pc} bytes early must still resolve")
    run(f"stale pc {hex(far_pc)} -> {hex(faddr)}",
        synth(far_pc, regnum, imm, size), 0, f"FAULTING INSTRUCTION at {hex(faddr)}")

    print("== 3 negative: an AMBIGUOUS window must be REFUSED, not answered")
    twin = None
    for i, (addr, mn2, ops, _raw) in enumerate(rows[:-1]):
        m = MEMOP.match(ops)
        if not (m and mn2 in ACCESS and m.group("base") in REG_NAMES):
            continue
        win = [r for r in rows if addr <= r[0] < addr + 0x100]
        if len(match_instruction(win, m.group("base"), int(m.group("imm"), 0),
                                 ACCESS[mn2][0], None)) > 1:
            twin = (addr, m.group("base"), int(m.group("imm"), 0), ACCESS[mn2][0])
            break
    if twin is None:
        print("   SKIP: this image has no same-(base, imm, size) pair within a window")
    else:
        taddr, treg, timm, tsize = twin
        pre = [r[0] for r in rows if taddr - 0x20 <= r[0] < taddr]
        run(f"two candidates after {hex(pre[0] if pre else taddr)}",
            synth(pre[0] if pre else taddr, REG_NAMES.index(treg), timm, tsize),
            2, "AMBIGUOUS")

    print("== 4 negative: an immediate PROVEN absent from the scanned window")
    scan = [r for r in rows if far_pc <= r[0] < far_pc + a.window]
    bogus = None
    for delta in range(1, 2048):
        for cand in (imm + delta, imm - delta):
            if -2048 <= cand < 2048 and not match_instruction(scan, reg, cand, size, None):
                bogus = cand
                break
        if bogus is not None:
            break
    if bogus is None:
        die("self-test: every immediate in range matches; cannot build a negative control")
    # Proven against the window locate() will actually scan, and asserted here so the
    # control cannot quietly become tautological again.
    assert not match_instruction(scan, reg, bogus, size, None)
    run(f"immediate {bogus} absent from [{hex(far_pc)}, {hex(far_pc + a.window)})",
        synth(far_pc, regnum, bogus, size), 2, "no instruction in")

    ok = all(results)
    print("self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", nargs="?", help="serial/run log containing the fault line")
    ap.add_argument("image", nargs="?", help="the .dom that ran (line tables give source lines)")
    ap.add_argument("--anchor-symbol", default="domain_main",
                    help="symbol the rung-0 anchor returns (default: domain_main)")
    ap.add_argument("--load-base", type=lambda s: int(s, 0), default=None,
                    help="load base, if the domain has no anchor rung")
    ap.add_argument("--window", type=lambda s: int(s, 0), default=0x100,
                    help="bytes to search forward from the reported pc (default 0x100)")
    ap.add_argument("--self-test", metavar="IMAGE",
                    help="run the positive and negative controls against IMAGE and exit")
    args = ap.parse_args()

    if args.self_test:
        return self_test(args.self_test)
    if not args.log or not args.image:
        ap.error("need <run-log> and <image.dom>, or --self-test IMAGE")
    return locate(args)


if __name__ == "__main__":
    sys.exit(main())
