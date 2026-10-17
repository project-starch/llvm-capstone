#!/usr/bin/env python3
"""Gate: no prediction runner may enable QEMU's test-only scalar-S LDC property.

Q-23's ruling keeps QEMU's experimental scalar-S LDC branch alive behind the CPU property
`x-capstone-test-s-ldc`, default off, because three Tier 0 suites' bootstrap needs it (BOOT_CAP
loads minted capabilities into registers from S before sret). Default QEMU then reads cause 24 as
the RTL does.

So a prediction is only valid if its QEMU invocation does NOT enable that property: with it on,
QEMU accepts an integer-based LDC that silicon refuses, and the prediction would be of the
emulator rather than of the hardware. That is exactly how the pre-ruling module passed run.py
41/41 and then wedged on vp2 at its first use.

WHY THE CONTROL HERE IS SYNTHETIC. The obvious positive control -- "the grep finds the property in
the three suites' run.sh" -- cannot fire: the property does not appear anywhere in this repo (the
suites live in the QEMU tree), so a gate relying on it would be unproven on the day it was written.
This instead injects the property into a copy of a real runner and requires the check to flag it,
which is deterministic and needs nothing from another lane. A second, weaker check confirms the
scan reads the right files at all, by finding a property that IS present (`x-capstone-u-mode`).

Exit 0 = clean, 1 = a runner enables it, 2 = the gate could not check (never a pass).
"""
import pathlib
import re
import sys
import tempfile

PROPERTY = "x-capstone-test-s-ldc"
KNOWN = "x-capstone-u-mode"          # present today, so the scan can be shown to read the files
# Every file that builds a QEMU command line for a prediction, plus the delegating runners.
CANDIDATES = (
    "capstone/runtime/virtual/run.py",
    "capstone/runtime/virtual/run-staged.py",
    "capstone/bug-corpora/tools/virtualvm.py",
    "capstone/bug-corpora/tools/appvm.py",
    "capstone/bug-corpora/tools/run-virtual-cases.py",
)


def enables(text: str) -> bool:
    """True if the text turns the property ON. Naming it with =false is not enabling it."""
    for m in re.finditer(re.escape(PROPERTY) + r"\s*=?\s*(\w+)?", text):
        value = (m.group(1) or "true").lower()
        if value not in ("false", "off", "0"):
            return True
    return False


def main() -> int:
    root = pathlib.Path(__file__).resolve().parents[4]
    files = [root / c for c in CANDIDATES]
    missing = [f for f in files if not f.is_file()]
    if missing:
        print("CANNOT CHECK: these runners are not where the gate expects them:", file=sys.stderr)
        for f in missing:
            print(f"  {f}", file=sys.stderr)
        return 2

    # control 1, synthetic: the check must flag an injected enable.
    with tempfile.TemporaryDirectory() as tmp:
        probe = pathlib.Path(tmp, "injected.py")
        probe.write_text(f"cmd = ['-cpu', 'rv64,{PROPERTY}=true']\n")
        if not enables(probe.read_text()):
            print("CANNOT CHECK: the injected positive control was not flagged", file=sys.stderr)
            return 2
        probe.write_text(f"cmd = ['-cpu', 'rv64,{PROPERTY}=false']\n")
        if enables(probe.read_text()):
            print("CANNOT CHECK: '=false' was read as enabling", file=sys.stderr)
            return 2
    print(f"  control: an injected {PROPERTY}=true is flagged, =false is not")

    # control 2: the scan reads the real files (a property that IS present is found).
    seen = [f for f in files if KNOWN in f.read_text()]
    if not seen:
        print(f"CANNOT CHECK: {KNOWN} found in none of the runners, so the scan may not be "
              f"reading them", file=sys.stderr)
        return 2
    print(f"  control: {KNOWN} found in {len(seen)} of {len(files)} runners, so the scan reads them")

    bad = [f for f in files if enables(f.read_text())]
    for f in files:
        print(f"  {'ENABLES' if f in bad else 'clean  '}  {f.relative_to(root)}")
    if bad:
        print(f"\nBLOCKED: {len(bad)} runner(s) enable {PROPERTY}; a prediction made with it is of "
              f"the emulator, not the hardware.", file=sys.stderr)
        return 1
    print(f"\nCLEAN: no runner enables {PROPERTY}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
