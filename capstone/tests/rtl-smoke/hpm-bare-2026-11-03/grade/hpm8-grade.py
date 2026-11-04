#!/usr/bin/env python3
"""hpm8-grade.py <run dir>: grade the hpm kit's L1I-miss positive control (hpm-8, added 2026-11-04) against
hpm-bare-2026-11-03/PREREG.md. Reads <run dir>/hpm-8/uart.txt (SR/SV) and run.log (image path, resident bitstream), hashes
the image the run loaded. Any bitstream may run it; the resident name is printed, not graded.
Vector: pass 1 L1I misses, pass 2 L1I misses, pass 1 cycles, pass 2 cycles, mhpmevent6 readback, traps, cause.
Simulated on bitstream 3's tokens (rtl-ids20-wt c3f5102f1, memory delay 12): 0x203 0x1 0x6636 0x834 0x1 0x0 0x0.
- EXACT: SR 7, readback 1, traps 0, cause 0.
- COUNTER: pass 1 misses in 480..560 (an 8 KiB cold block = 512 lines, plus the call and loop lines) and pass 2 <= 8.
- Cycles are timing: reported, with the per-miss cost (c1 - c2) / (m1 - m2).
Exit 0 = AS EXPECTED (the L1I-miss event counts on this bitstream); 1 = FINDING (it does not, or an exact field is off);
2 = NO DATA."""
import sys, re, hashlib, pathlib
D = pathlib.Path(sys.argv[1]) / 'hpm-8'
H = '0094235951c3e398cc7aad44b99fbe506ba3f0ef8b52e7566a383c9d883da4c5'
u, r = D / 'uart.txt', D / 'run.log'
if not u.exists() or not r.exists(): print(f'NO DATA: no uart.txt / run.log in {D}'); sys.exit(2)
t = u.read_text(errors='replace'); lg = r.read_text(errors='replace')
sr = [int(x, 16) for x in re.findall(r'^SR ([0-9A-Fa-f]{16})\s*$', t, re.M)]
sv = [int(x, 16) for x in re.findall(r'^SV ([0-9A-Fa-f]{16})\s*$', t, re.M)]
m = re.search(r'image (\S+\.bin) ->', lg); img = pathlib.Path(m.group(1)) if m else None
h = hashlib.sha256(img.read_bytes()).hexdigest() if img and img.exists() else None
bs = re.findall(r"resident NV bitstream = '([^']+)'", lg)
print(f'hpm-8: img {"OK" if h == H else "MISMATCH"} bs {bs} SR {sr} SV {" ".join(hex(x) for x in sv)}')
if h != H or sr != [7] or len(sv) != 7: print('FINDING: image or record shape off'); sys.exit(1)
m1, m2, c1, c2, rb, tr, ca = sv
exact = rb == 1 and tr == 0 and ca == 0
counts = 480 <= m1 <= 560 and m2 <= 8
print(f'exact (readback 1, traps 0, cause 0): {"OK" if exact else "OFF"}')
print(f'L1I misses pass 1 {m1} (480..560), pass 2 {m2} (<= 8): {"COUNTS" if counts else "DOES NOT COUNT AS PREDICTED"}')
if m1 > m2: print(f'cycles pass 1 {c1}, pass 2 {c2}: {(c1 - c2) / (m1 - m2):.1f} cycles per L1I miss (sim 46.8)')
v = 'AS EXPECTED' if exact and counts else 'FINDING'
print(f'HPM-8 on {bs[-1] if bs else "?"}: {v}'); sys.exit(0 if v == 'AS EXPECTED' else 1)
