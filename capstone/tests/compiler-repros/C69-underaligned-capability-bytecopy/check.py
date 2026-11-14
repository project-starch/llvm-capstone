#!/usr/bin/env python3
"""C-69: an under-aligned capability access is lowered to a BYTE COPY, silently dropping the tag.

Two-sided by construction, because a one-sided check here would be worthless: the aligned control
in the same file must compile to a single `ldc`, and the under-aligned case must NOT. If the
control ever stops being a single ldc, this check is measuring the compiler's mood, not the defect.

Since the fix (CapstoneMisalignedCapAccess.cpp) the byte copy is still there, as the arm a MISALIGNED address takes;
an aligned address takes an ldc/stc after a run-time `andi ..., 15` test. So the static verdict is: byte copy with no
alignment test = PRESENT; byte copy behind the test = DISPATCHED (gone). This check reads the code shape only; the
run-time test c69-run.c is the evidence that the tag survives.

Usage:  check.py <clang> [-O<n>]  exit 0 = defect still present, 1 = gone (or the control broke)
"""
import re, subprocess, sys, tempfile, os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
clang = sys.argv[1] if len(sys.argv) > 1 else "clang"
opt = sys.argv[2] if len(sys.argv) > 2 else "-O2"   # -O0: clang marks every function optnone

out = subprocess.run([clang, "--target=capstone64-unknown-elf", opt, "-S", "-o", "-",
                      os.path.join(HERE, "repro.c")], capture_output=True, text=True)
if out.returncode != 0:
    print("ERROR: the reproducer did not compile; this says nothing about C-69", file=sys.stderr)
    print(out.stderr[-2000:], file=sys.stderr)
    sys.exit(2)
asm = out.stdout

def body(fn):
    m = re.search(rf'^{fn}:(.*?)^\t\.size\t{fn}', asm, re.S | re.M)
    if not m:
        print(f"ERROR: {fn} not found in the assembly -- not a clean result, a broken check", file=sys.stderr)
        sys.exit(2)
    return m.group(1)

def census(fn):
    return Counter(o for o in re.findall(r'^\s+([a-z][a-z0-9._]*)\s', body(fn), re.M))

def tested(fn):
    return re.search(r'^\s+andi\s+\w+, \w+, 15$', body(fn), re.M) is not None

at, direct, store = census("field_at"), census("field_direct"), census("field_store")

print(f"field_at      lbu={at['lbu']:2d}  ldc={at['ldc']:2d}  sd={at['sd']:2d}")
print(f"field_store    sb={store['sb']:2d}  stc={store['stc']:2d}  ld={store['ld']:2d}")
print(f"field_direct  ldc={direct['ldc']:2d}  lbu={direct['lbu']:2d}   <- the aligned control")

ok_control = direct['lbu'] == 0 and direct['ldc'] >= 1
if not ok_control:
    print("CONTROL BROKEN: the naturally-aligned access is no longer a plain ldc. "
          "Fix the control before reading anything into the other two.", file=sys.stderr)
    sys.exit(1)

load_bytecopy  = at['lbu'] >= 8 and not tested("field_at")
store_bytecopy = store['sb'] >= 8 and not tested("field_store")
if load_bytecopy or store_bytecopy:
    print(f"C-69 PRESENT: load byte-copy={load_bytecopy}, store byte-copy={store_bytecopy} "
          f"-- a capability is assembled from/scattered to bytes, so the tag cannot survive.")
    sys.exit(0)
print(f"C-69 ABSENT: no untested byte copy (alignment test: load={tested('field_at')}, store={tested('field_store')}), "
      "and the aligned control still lowers to ldc.")
sys.exit(1)
