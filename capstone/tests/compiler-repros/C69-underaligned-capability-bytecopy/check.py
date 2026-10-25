#!/usr/bin/env python3
"""C-69: an under-aligned capability access is lowered to a BYTE COPY, silently dropping the tag.

Two-sided by construction, because a one-sided check here would be worthless: the aligned control
in the same file must compile to a single `ldc`, and the under-aligned case must NOT. If the
control ever stops being a single ldc, this check is measuring the compiler's mood, not the defect.

Usage:  check.py <clang>        exit 0 = defect still present, 1 = gone (or the control broke)
"""
import re, subprocess, sys, tempfile, os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
clang = sys.argv[1] if len(sys.argv) > 1 else "clang"

out = subprocess.run([clang, "--target=capstone64-unknown-elf", "-O2", "-S", "-o", "-",
                      os.path.join(HERE, "repro.c")], capture_output=True, text=True)
if out.returncode != 0:
    print("ERROR: the reproducer did not compile; this says nothing about C-69", file=sys.stderr)
    print(out.stderr[-2000:], file=sys.stderr)
    sys.exit(2)
asm = out.stdout

def census(fn):
    m = re.search(rf'^{fn}:(.*?)^\t\.size\t{fn}', asm, re.S | re.M)
    if not m:
        print(f"ERROR: {fn} not found in the assembly -- not a clean result, a broken check", file=sys.stderr)
        sys.exit(2)
    return Counter(o for o in re.findall(r'^\s+([a-z][a-z0-9._]*)\s', m.group(1), re.M))

at, direct, store = census("field_at"), census("field_direct"), census("field_store")

print(f"field_at      lbu={at['lbu']:2d}  ldc={at['ldc']:2d}  sd={at['sd']:2d}")
print(f"field_store    sb={store['sb']:2d}  stc={store['stc']:2d}  ld={store['ld']:2d}")
print(f"field_direct  ldc={direct['ldc']:2d}  lbu={direct['lbu']:2d}   <- the aligned control")

ok_control = direct['lbu'] == 0 and direct['ldc'] >= 1
if not ok_control:
    print("CONTROL BROKEN: the naturally-aligned access is no longer a plain ldc. "
          "Fix the control before reading anything into the other two.", file=sys.stderr)
    sys.exit(1)

load_bytecopy  = at['lbu'] >= 8
store_bytecopy = store['sb'] >= 8
if load_bytecopy or store_bytecopy:
    print(f"C-69 PRESENT: load byte-copy={load_bytecopy}, store byte-copy={store_bytecopy} "
          f"-- a capability is assembled from/scattered to bytes, so the tag cannot survive.")
    sys.exit(0)
print("C-69 ABSENT: neither access is a byte copy, and the aligned control still lowers to ldc.")
sys.exit(1)
