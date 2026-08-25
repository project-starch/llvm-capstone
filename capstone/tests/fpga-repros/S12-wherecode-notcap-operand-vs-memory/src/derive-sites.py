#!/usr/bin/env python3
"""Derive the S-12 run constants FROM THE ARTIFACT, never from memory.

WHY THIS EXISTS. `expected mepc = DBAS + 0x41c` was committed, was correct for the build it was
written against, and became false the moment the self-arming code shifted the layout. It then
survived several re-reads BECAUSE it was already committed -- the commit made it look settled.
Used as a precondition it would have declared a good wedge at +0x484 VOID and thrown the real
result away: a precondition that rejects correct data, failing silently toward "nothing to see".

These addresses are properties of the BUILD, not of the bug. Re-derive them whenever the build
changes; that is what this is for.

TWO TRAPS THIS SCRIPT ITSELF FELL INTO, kept as comments because both are easy to repeat:

  1. Matching the mnemonic text `cincoffsetimm.*0xb0` picks the WRONG instruction. This function
     contains `cincoffsetimm a0, s0, -0xb0` at a LOWER address than the subject
     `cincoffsetimm a4, a4, 0xb0`, so "first match" gave 0x45c where the measured mepc was 0x484.
     A derivation that produces the wrong constant is worse than the stale constant it replaces.
     Match the ENCODING -- 5b 27 07 0b -- which is verified byte-identical to the SQLite fault.

  2. The shell version died with SIGPIPE (exit 141) because awk's early `exit` killed objdump
     under `set -o pipefail`, and printed nothing at all -- a silent failure that looked like
     "no sites found". Python reads the whole stream and cannot do that.

  usage: derive-sites.py <arm.dom> [...]
"""
import pathlib, re, subprocess, sys

SUBJECT_ENCODING = "5b 27 07 0b"       # cincoffsetimm a4, a4, 0xb0 -- the production consumer
ELF_VA_BASE = 0x10000
SLOT_OFF = 0x700 - 0x70                # s12_frame + 0x700 - 0x70, see s12_kernel.h

root = pathlib.Path(__file__).resolve().parents[5]
od = root / "llvm/cmake-build-debug/bin/llvm-objdump"
re_ = root / "llvm/cmake-build-debug/bin/llvm-readelf"
for t in (od, re_):
    if not t.exists():
        sys.exit(f"toolchain missing: {t}")

def sites(dom):
    dis = subprocess.run([str(od), "-d", "--triple=capstone64-unknown-elf", dom],
                         capture_output=True, text=True).stdout
    body, seen = [], False
    for l in dis.splitlines():
        if "<s12_compute>:" in l:
            seen = True; continue
        if seen and re.match(r"^0000", l):
            break
        if seen:
            body.append(l)
    if not body:
        return None, "no s12_compute in this artifact -- wrong file?"
    cons = arm = None
    for l in body:
        m = re.match(r"\s*([0-9a-f]+):\s+((?:[0-9a-f]{2} )+)\s*\t(.*)", l)
        if not m:
            continue
        a, by, txt = int(m.group(1), 16), m.group(2).strip(), m.group(3).strip()
        if cons is None and by == SUBJECT_ENCODING:
            cons = a
        if arm is None and re.search(r"\blcc\b", txt) and txt.rstrip().endswith("0x2"):
            arm = a
    syms = subprocess.run([str(re_), "--symbols", dom], capture_output=True, text=True).stdout
    frame = next((int(p[1], 16) for p in (l.split() for l in syms.splitlines())
                  if len(p) >= 8 and p[7] == "s12_frame"), None)
    return (cons, arm, frame), None

for dom in sys.argv[1:]:
    got, err = sites(dom)
    print(f"artifact: {dom}")
    if err:
        print(f"  ERROR: {err}"); continue
    cons, arm, frame = got
    if cons is None:
        print("  subject consumer  ABSENT -- this arm cannot produce the subject fault "
              "(expected for the lcc arm)")
    else:
        print(f"  subject consumer  VA 0x{cons:x}   -> expected mepc = DBAS + 0x{cons-ELF_VA_BASE:x}")
    if arm is None:
        print("  arming lcc        absent (built without -DS12_SELF_ARM_WP)")
    else:
        print(f"  arming lcc        VA 0x{arm:x}   -> a wedge at DBAS + 0x{arm-ELF_VA_BASE:x} is an "
              f"ARMING failure; that arm is VOID")
    if frame is None:
        print("  slot              s12_frame symbol NOT FOUND")
    else:
        slot = frame + SLOT_OFF
        print(f"  slot              VA 0x{slot:x}   -> watchpoint paddr = DBAS + 0x{slot-ELF_VA_BASE:x}")
        print(f"  slot is a granule base (16-byte aligned, required by the compare): {slot%16==0}")
        if slot % 16:
            print("  ERROR: slot is NOT a granule base -- the watchpoint can never match")
    print()
