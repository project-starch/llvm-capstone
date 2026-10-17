#!/usr/bin/env python3
"""Two-sided test of the drivers' monitor-tag classifiers after the S-trap injection (2026-10-17).

capstone-sbi vp2/os-trap-inject makes the monitor hand a trap from S or U to the kernel: it reports EXCX or ILLX as
before, then INJS, and the system goes on. It also adds two stops of its own, NODM and INJL. Before this change both
drivers read an EXCX as the monitor's own stop (run_sqlite_stages_fpga.py's spin tag) or as an unhandled trap
(run_baked_rungs_fpga.py's no_result_kind), and knew neither NODM nor INJL. Each case asserts the old reading's error
and the new reading.

  python3 capstone/tests/rtl-smoke/fpga_driver/test_monitor_tags.py    exit 0 = all pass
"""
import pathlib, re, sys

HERE = pathlib.Path(__file__).parent
OLD_SPIN = re.compile(r"(SPL[AB]|ILLX|EXCX|RCPX|WCPX|SHAX|RGNO|DPIC|DPIX|DRET|RCSH|RCPR):([0-9A-Fa-f]{8})")


def load(fname, start, end):
    src = (HERE / fname).read_text()
    ns = {"re": re}
    exec(src[src.index(start):src.index(end)], ns)
    return ns


spin = load("run_sqlite_stages_fpga.py", "MONITOR_SPIN_TAGS = ", "def stall_shape(")["monitor_spin_tag"]
kind = load("run_baked_rungs_fpga.py", "def no_result_kind(", "def read_wedge_apertures(")["no_result_kind"]

EXCX_S = "EXCX:0000E002\nMCAU:00000018\nMEPC:FFFFFFFF0123ABCD\nMTVL:00000000DEAD0040\nMSTA:0000000A00000900\n"
INJECTED = EXCX_S + "INJS:00000018\n[  12.3] Oops - capability fault\n"

# (name, text, sqlite-driver spin tag, baked-rung kind, what the OLD sqlite regex read)
CASES = [
    ("EXCX without INJS is still a monitor spin", EXCX_S, "EXCX:0000E002", "monitor-trap", "EXCX:0000E002"),
    ("EXCX then INJS is not a spin", INJECTED, None, "injected-trap", "EXCX:0000E002"),
    ("ILLX then INJS is not a spin", "ILLX:0000E003\nINJS:00000002\n", None, "injected-trap", "ILLX:0000E003"),
    ("a later stop after an injection is found", INJECTED + "RGNO:0000E00C\n", "RGNO:0000E00C", "injected-trap",
     "EXCX:0000E002"),
    ("NODM is a monitor stop", EXCX_S.replace("MSTA:0000000A00000900", "MSTA:0000000A00001900") + "NODM:00000019\n",
     "NODM:00000019", "monitor-stop", "EXCX:0000E002"),
    ("INJL is a monitor stop", EXCX_S + "INJL:00000018\n", "INJL:00000018", "monitor-stop", "EXCX:0000E002"),
    ("an entry stall is unchanged", "SHA5:00000001\n", None, "entry-stall", None),
    ("a Linux-module rung with no tags", "insmod: ok\n", None, "no-share-markers", None),
]

fails = 0
for name, text, want_spin, want_kind, old in CASES:
    m = spin(text)
    got_spin = m.group(0) if m else None
    got_kind = kind(text)
    o = OLD_SPIN.search(text)
    got_old = o.group(0) if o else None
    ok = got_spin == want_spin and got_kind == want_kind and got_old == old
    fails += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {name}: spin {got_spin} (old {got_old}), kind {got_kind}")

# The old reading must be wrong where the injection changed the meaning, or this test proves nothing.
changed = [c for c in CASES if c[2] != c[4]]
print(f"{len(changed)} of {len(CASES)} cases read differently from the old regex")
if len(changed) < 3:
    fails += 1
    print("FAIL: the old regex no longer misreads the injected cases; revisit this test")
print("PASS" if fails == 0 else f"{fails} FAILED")
sys.exit(1 if fails else 0)
