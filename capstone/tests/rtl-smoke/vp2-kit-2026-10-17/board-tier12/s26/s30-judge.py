#!/usr/bin/env python3
"""s30-judge.py BOARD_TXT ARM (s30-old|s30-new): S-30's matched pair, read from the run's console. Exit 0 as predicted,
1 not, 2 no data. Predictions (pre-registered 2026-10-19 evening, before the boots):
- both: QNT0 001E8480; b0-stats 0; vm-contract 0 (identity branch, no STEP before it); b0-hello 0; no kernel Oops.
- s30-old (wrapper aab63d5): no S30H line. s30-read retval 0: the node record reads, and the dom_stack's tag words read
  with at least one nonzero (the positive control that the reads reach tag memory, since the monitor's region table
  holds capabilities). This is S-30 DEMONSTRATED for reads on the board.
- s30-new (wrapper 9bee361): exactly two S30H lines in this boot, the held piece's decoded base <= 0xBC2D2000 and end
  0xC0000000. s30-read retval 3: both groups refused, at their first read, and the boot goes on (RESULT present). The
  canary rungs before it pass, so Linux's RAM below 0xBC2D2000 is still reachable.
  A refusal is graded by its LINES, not by an absent value (amended 2026-10-19 18:40, before the boots): busybox devmem
  opens /dev/mem O_RDONLY and loads through mmap (devmem.c:73-88), so a refused read is a U-mode load access fault.
  The monitor's no-region path prints MTVL:<the address> and CPMX:00000002, and handle_exception injects it into S
  with INJS:00000002 (inject_into_s(CAPSTONE_NO_CPMP_REGION), 5b5e86b :3815; pb10's panics read the same pair). Linux kills devmem with a signal, so the shell's rc is 135 (SIGBUS) or 139 (SIGSEGV). Required, in this
  boot's console: MTVL:BFF00000 and MTVL:BC2DAD60, CPMX:00000002 and INJS:00000002, and both S30R refusals at rc 135 or 139."""
import os, re, sys
t = open(sys.argv[1], errors="replace").read().replace("\r", ""); arm = sys.argv[2]
rawp = os.path.join(os.path.dirname(os.path.abspath(sys.argv[1])), "board-raw.txt")
raw = open(rawp, errors="replace").read().replace("\r", "") if os.path.exists(rawp) else ""
b = raw.rfind("OpenSBI v"); seg = raw[b:] if b >= 0 else ""
out, bad = [], 0
def say(ok, msg):
    global bad; out.append(("OK   " if ok else "FAIL ") + msg); bad += not ok
def rung(r):
    m = re.findall(rf"^RESULT {r} retval=(-?\d+)", t, re.M); return m[-1] if m else None
if not seg: print("NO DATA: no OpenSBI banner in board-raw.txt"); sys.exit(2)
q0 = re.findall(r"QNT0:([0-9A-F]{8})", seg); h = re.findall(r"S30H:([0-9A-F]{8})", seg)
say(q0 == ["001E8480"], f"QNT0 {q0}")
for r in ("b0-stats", "vm-contract", "b0-hello"):
    say(rung(r) == "0", f"{r} retval {rung(r)} (want 0)")
say(not re.search(r"Oops|Kernel panic", t), "no kernel Oops or panic in the rungs")
sr = rung("s30-read")
if sr is None: print("\n".join(out)); print("NO DATA: no s30-read RESULT"); sys.exit(2)
words = re.findall(r"^S30R tag words read (\d+) nonzero (\d+) first (\S+)", t, re.M)
nodes = re.findall(r"^S30R node (0x[0-9A-F]+) rc=(\d+) (.*)$", t, re.M)
out.append(f"NOTE node reads {nodes}; tag {words}")
if arm == "s30-old":
    say(not h, f"no S30H line {h}")
    say(sr == "0", f"s30-read retval {sr} (want 0: every read returned)")
    say(bool(words) and int(words[-1][1]) >= 1, "at least one nonzero tag word (the reads reach tag memory)")
else:
    say(len(h) == 2 and int(h[0], 16) <= 0xBC2D2000 and int(h[1], 16) == 0xC0000000,
        f"S30H {h} (want 2 lines: base <= BC2D2000, end C0000000)")
    say(sr == "3", f"s30-read retval {sr} (want 3: both groups refused)")
    say("MTVL:BFF00000" in seg and "MTVL:BC2DAD60" in seg, "MTVL:BFF00000 and MTVL:BC2DAD60 in this boot (the monitor saw both refusals)")
    say("CPMX:00000002" in seg and "INJS:00000002" in seg, "CPMX:00000002 and INJS:00000002 in this boot (no region covers it; injected into S)")
    rcs = re.findall(r"^S30R (?:node|tag) 0x[0-9A-F]+ rc=(\d+)", t, re.M)
    bad_rc = [r for r in rcs if r != "0"]
    say(len(bad_rc) == 2 and all(r in ("135", "139") for r in bad_rc), f"refused reads' rc {bad_rc} (want two, each 135 or 139)")
print("\n".join(out)); print("JUDGE PASS" if not bad else f"JUDGE FAIL ({bad})"); sys.exit(1 if bad else 0)
