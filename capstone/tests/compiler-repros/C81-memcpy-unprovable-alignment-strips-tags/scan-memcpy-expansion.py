#!/usr/bin/env python3
"""Scan a Capstone image or object for byte/word-granular memcpy expansions (C-81).

WHY THIS EXISTS. `-fno-builtin` diverts only EXPLICIT `memcpy` calls, so an image built
with it can still carry the defect through `__builtin_memcpy` or a copy the compiler
synthesised. And `manifest.json` does not record the codegen flags, so there is no
retrospective flag test. The disassembly is the only artifact that sees all of it.

WHAT IT LOOKS FOR. A maximal run of straight-line scalar load/store instructions, no
branch or call inside it, moving >= THRESHOLD bytes in units smaller than a capability.
That is what the scalar path of findOptimalMemOpLowering emits
(CapstoneISelLowering.cpp:25859-25882) when it cannot prove 16-alignment.

A HIT IS A CANDIDATE, NOT A FINDING. An 8-byte-unit run can be ordinary struct code that
never held a capability; the finding is a hit over a region that holds one. Byte-granular
(unit 1) runs of this length are the strong signal -- nothing else emits them.

NO DATA IS AN ERROR. If the disassembler fails, or a file yields no instructions, this
exits non-zero and says where it looked, rather than printing zero hits like a clean bill.

The positive control is this folder's own repro.c: --self-test compiles it and requires
copy32_unknown and copy32_unknown_builtin to be FLAGGED and copy32_known to be CLEAN.
A scan whose control has not fired is not evidence.
"""
import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOADS = {"lb": 1, "lbu": 1, "lh": 2, "lhu": 2, "lw": 4, "lwu": 4, "ld": 8}
STORES = {"sb": 1, "sh": 2, "sw": 4, "sd": 8}
# What ends a straight-line run: control flow, and capability-WIDTH moves (a run containing
# ldc/stc is not the scalar path).
#
# `c[a-z]+` USED TO BE IN THIS LIST AND WAS A DEFECT. It matched cincoffset/cincoffsetimm, i.e.
# the pointer arithmetic a copy performs BETWEEN its own chunks, so one copy was split into
# sub-threshold pieces and dropped. Measured cost on two subobject images: 30 runs hidden in
# each (91 -> 121, 82 -> 112). The positive control never caught it because repro.c's copies
# have no intervening capability arithmetic -- a check keyed to one shape silently missing
# another. `check_no_split_on_pointer_arithmetic()` below is the test that would have.
BREAKERS = re.compile(r"^(j|jr|jal|jalr|cjal|cjalr|b[a-z]+|ret|ecall|ebreak|ldc|stc|movc|tail|call)\b")


def disassemble(path: Path, objdump: str):
    out = subprocess.run([objdump, "-d", "--no-show-raw-insn", str(path)],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"ERROR: {objdump} failed on {path} rc={out.returncode}: "
                         f"{out.stderr.strip()[:300]}")
    funcs, cur = {}, None
    for line in out.stdout.splitlines():
        m = re.match(r"^[0-9a-f]+\s+<(.+)>:$", line.strip())
        if m:
            cur = m.group(1)
            funcs.setdefault(cur, [])
            continue
        m = re.match(r"^\s+([0-9a-f]+):\s+(\S+)\s*(.*)$", line)
        if m and cur is not None:
            funcs[cur].append((int(m.group(1), 16), m.group(2), m.group(3).strip()))
    if not funcs:
        raise SystemExit(f"ERROR: no disassembly for {path} (looked with {objdump} -d). "
                         f"Not a scan result -- the instrument produced nothing.")
    return funcs


def runs(instructions, threshold):
    """Maximal straight-line scalar load/store runs, as (bytes_moved, unit, n_insns)."""
    found, cur = [], []
    def flush():
        if not cur:
            return
        moved = sum(w for _, w in cur if _ == "s")
        if moved >= threshold:
            unit = max(w for _, w in cur)
            if unit < 16:
                found.append((moved, min(w for _, w in cur), len(cur)))
    for _, mnem, _rest in instructions:
        if mnem in LOADS:
            cur.append(("l", LOADS[mnem]))
        elif mnem in STORES:
            cur.append(("s", STORES[mnem]))
        elif BREAKERS.match(mnem):
            flush(); cur = []
        # anything else (arithmetic, lui, addi) neither extends nor breaks the run
    flush()
    return found


def scan(path: Path, objdump: str, threshold: int):
    hits = []
    for fn, ins in disassemble(path, objdump).items():
        for moved, unit, n in runs(ins, threshold):
            hits.append((fn, moved, unit, n))
    return hits


def check_no_split_on_pointer_arithmetic() -> int:
    """A copy's own pointer arithmetic must not end its run. Synthetic, so it always runs.

    This is the regression test for the `c[a-z]+` breaker defect: 32 one-byte stores with a
    `cincoffsetimm` in the middle are ONE copy, and must be reported as one run of 32 bytes,
    not two of 16. Deterministic -- no compiler needed.
    """
    mid = [(0, "lbu", "a2, 0(a0)"), (0, "sb", "a2, 0(a1)")] * 16
    mid.insert(16, (0, "cincoffsetimm", "a0, a0, 0x10"))
    got = runs(mid, 16)
    ok = len(got) == 1 and got[0][0] == 16
    print(f"  self-test synthetic  pointer-arithmetic mid-copy "
          f"{'one run, not split (ok)' if ok else f'SPLIT into {len(got)} -- the breaker defect is back'}")
    # and the converse: control flow MUST still end a run
    br = [(0, "lbu", ""), (0, "sb", "")] * 16
    br.insert(16, (0, "beq", "a0, a1, .L1"))
    got2 = runs(br, 16)
    ok2 = len(got2) == 0 or all(r[0] < 16 for r in got2)
    print(f"  self-test synthetic  control flow mid-copy        "
          f"{'splits the run (ok)' if ok2 else 'DID NOT split -- breaker too narrow'}")
    return 0 if (ok and ok2) else 1


def self_test(objdump: str, threshold: int, clang: str) -> int:
    src = HERE / "repro.c"
    if not src.is_file():
        print(f"SELF-TEST ERROR: no {src}", file=sys.stderr)
        return 2
    flags = ["--no-default-config", "-target", "capstone64-unknown-elf", "-nostdinc",
             "-Xclang", "-target-feature", "-Xclang", "+m",
             "-Xclang", "-target-feature", "-Xclang", "+a",
             "-Xclang", "-target-feature", "-Xclang", "+f",
             "-Xclang", "-target-feature", "-Xclang", "+d",
             "-Xclang", "-target-abi", "-Xclang", "lp64", "-fno-jump-tables",
             "-mllvm", "-capstone-gp-free", "-mllvm", "-capstone-image-gp"]
    rc = check_no_split_on_pointer_arithmetic()
    with tempfile.TemporaryDirectory() as tmp:
        for arm, extra in (("plain", []), ("nobuiltin", ["-fno-builtin"])):
            obj = Path(tmp) / f"c81-{arm}.o"
            done = subprocess.run([clang, *flags, *extra, "-O1", "-c", str(src), "-o", str(obj)],
                                  capture_output=True, text=True)
            if done.returncode:
                print(f"SELF-TEST ERROR: control did not compile ({arm}): "
                      f"{done.stderr.strip().splitlines()[0] if done.stderr else '?'}", file=sys.stderr)
                return 2
            flagged = {fn for fn, *_ in scan(obj, objdump, threshold)}
            want_flagged = {"copy32_unknown", "copy32_unknown_builtin"} if arm == "plain" \
                else {"copy32_unknown_builtin"}
            want_clean = {"copy32_known"} | ({"copy32_unknown"} if arm == "nobuiltin" else set())
            for fn in sorted(want_flagged):
                ok = fn in flagged
                print(f"  self-test {arm:<10} {fn:<24} {'FLAGGED (ok)' if ok else 'NOT FLAGGED -- control failed'}")
                rc |= 0 if ok else 1
            for fn in sorted(want_clean):
                ok = fn not in flagged
                print(f"  self-test {arm:<10} {fn:<24} {'clean (ok)' if ok else 'FLAGGED -- false positive'}")
                rc |= 0 if ok else 1
    print("  self-test: PASS -- the scanner fires on the defect and not on the control"
          if rc == 0 else "  self-test: FAILED -- do not trust a scan from this build")
    return rc


def diff(a: Path, b: Path, objdump: str, threshold: int) -> int:
    """The validated use: two builds of ONE source, expansions present only in the first.

    A single image's hit count is dominated by ordinary struct code at unit 8. The
    informative quantity is the set difference. Runs are keyed on their SHAPE, not on the
    enclosing symbol, because these images are stripped and the two builds renumber
    `.Lpcrel_hiNN` differently.
    """
    import collections
    import hashlib
    ha = hashlib.sha256(a.read_bytes()).hexdigest()[:8]
    hb = hashlib.sha256(b.read_bytes()).hexdigest()[:8]
    if ha == hb:
        print(f"  {a.name}: BYTE-IDENTICAL ({ha}) -- no difference to measure, not a control")
        return 0
    def sig(p):
        return collections.Counter((m, u, n) for _f, m, u, n in scan(p, objdump, threshold))
    A, B = sig(a), sig(b)
    only_a, only_b = A - B, B - A
    by_unit = collections.Counter()
    for (_m, u, _n), k in only_a.items():
        by_unit[u] += k
    print(f"  A {ha} {a.name}: {sum(A.values())} runs")
    print(f"  B {hb} {b.name}: {sum(B.values())} runs")
    print(f"  only in A: {sum(only_a.values())}   only in B: {sum(only_b.values())}")
    print(f"  A-only by unit: {dict(sorted(by_unit.items()))}")
    for (m, u, n), k in sorted(only_a.items(), key=lambda kv: -kv[0][0]):
        note = "  <-- exactly one capability, byte-expanded" if (m == 16 and u == 1) else ""
        print(f"    {m:>5}B unit={u} ({n} insns) x{k}{note}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images", nargs="*", type=Path, help="images or objects to scan")
    p.add_argument("--objdump", default=None, help="llvm-objdump (default: found on PATH or the known build)")
    p.add_argument("--clang", default=None, help="clang for --self-test")
    p.add_argument("--threshold", type=int, default=16,
                   help="minimum bytes STORED in one run to report (default 16, one capability)")
    p.add_argument("--self-test", action="store_true", help="run the positive control and exit")
    p.add_argument("--diff", nargs=2, metavar=("A", "B"), type=Path,
                   help="two builds of ONE source: report expansions present only in A. "
                        "This is the validated use; a single-image scan under-determines.")
    p.add_argument("--skip-self-test", action="store_true",
                   help="scan without the control; the output then proves nothing about the scanner")
    a = p.parse_args()

    objdump = a.objdump or shutil.which("llvm-objdump") \
        or "/tmp/capstone/virt-stack/llvm-build/bin/llvm-objdump"
    if not Path(objdump).is_file():
        print(f"ERROR: no llvm-objdump at {objdump} (pass --objdump)", file=sys.stderr)
        return 2
    clang = a.clang or shutil.which("clang") or "/tmp/capstone/llvm-dev-current-20261017/bin/clang"

    if a.self_test:
        return self_test(objdump, a.threshold, clang)
    if a.diff:
        if not a.skip_self_test:
            print(f"positive control (repro.c, {clang}):")
            if self_test(objdump, a.threshold, clang):
                print("ERROR: refusing to report a diff whose control did not fire", file=sys.stderr)
                return 2
            print()
        return diff(a.diff[0], a.diff[1], objdump, a.threshold)
    if not a.images:
        print("ERROR: nothing to scan and --self-test not given", file=sys.stderr)
        return 2
    if not a.skip_self_test:
        print(f"positive control (repro.c, {clang}):")
        if self_test(objdump, a.threshold, clang):
            print("ERROR: refusing to report a scan whose control did not fire", file=sys.stderr)
            return 2
        print()

    total = 0
    print(f"{'image':<44} {'hits':>5}  worst run")
    for img in a.images:
        if not img.is_file():
            print(f"ERROR: {img} is not a file", file=sys.stderr)
            return 2
        hits = scan(img, objdump, a.threshold)
        total += len(hits)
        worst = max(hits, key=lambda h: h[1]) if hits else None
        desc = f"{worst[1]}B unit={worst[2]} in {worst[0]}" if worst else "-"
        print(f"{img.name:<44} {len(hits):>5}  {desc}")
        for fn, moved, unit, n in sorted(hits, key=lambda h: -h[1])[:6]:
            print(f"    {fn:<48} {moved:>5}B unit={unit} ({n} insns)")
    print(f"\n{total} candidate run(s). A hit is a CANDIDATE; the finding is a hit over a "
          f"region that holds a capability.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
