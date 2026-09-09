#!/usr/bin/env python3
"""Read one campaign boot's SLT result from ITS OWN segment of the raw transcript.

The console replays the previous boot's UART history on every connect, so a first-match grep
reads the WRONG run (2026-09-05: the S-04 boot's q_two summary was read as the negative
control's). Scope to the last `### TEST 2/2 START ... <test>` marker, then take the summary
after it.   usage: slt-read.py <results-dir> <name> <test-file> <native.txt>"""
import re, sys, pathlib
res, name, tf, native = sys.argv[1:5]
raw = pathlib.Path(res, f"{name}-raw.txt")
t = raw.read_text(errors="replace") if raw.exists() else ""
starts = [m.start() for m in re.finditer(r"### TEST 2/2 START[^\n]*" + re.escape(tf), t)]
seg = t[starts[-1]:] if starts else ""
m = re.search(r"SLT-SUMMARY[^\r\n]*", seg)
sil = m.group(0).strip() if m else ""
# the control runs as TEST 1/2, BEFORE this run's TEST 2/2 marker: take the last one before it
ctls = re.findall(r"RESULT k800 retval=(\d+)", t[:starts[-1]] if starts else t)
ctl = ctls[-1] if ctls else None
nat = dict(l.rstrip("\n").split("\t", 1) for l in open(native) if "\t" in l).get(name, "")
print(f"boot {name}: run-markers={len(starts)} control k800={ctl or 'NONE'} "
      f"G/enter={seg.count('G/enter')} H/return={seg.count('H/return')} halted={seg.count('halted')} NOMEM={seg.count('NOMEM')}")
print(f"  silicon: {sil or 'NO SUMMARY (VOID or wedge -- read the driver log)'}")
print(f"  native : {nat}")
ok = bool(sil and nat and sil.split(' completed')[0] == nat.split(' completed')[0])
print("  VERDICT:", "MATCHES NATIVE" if ok else "DIFFERS or NO RESULT")
sys.exit(0 if ok else 1)
