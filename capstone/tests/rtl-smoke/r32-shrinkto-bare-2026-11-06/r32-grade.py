#!/usr/bin/env python3
"""Grade one bare boot of the R-32 kit (shrinkto.bin) from its UART capture.

usage: r32-grade.py <uart.txt> [--expect prefix|fixed]

Reads the LAST "SUPTEST BEGIN R" block (the console replays earlier boots on connect), then its "SR <n>" count and
"SV <v>" values. Exit 0 = the vector named by --expect (default prefix, the C.5a / stage A reading), 1 = the other
known vector or any other vector (a finding, printed value by value), 2 = VOID or no data (says where it looked).
"""
import re, sys

POS = ["parent start", "parent end", "SHRINK start", "SHRINK end",
       "SHRINKTO 64 start", "SHRINKTO 64 end", "SHRINKTO 1 start", "SHRINKTO 1 end",
       "SHRINKTO 17 start", "SHRINKTO 17 end", "SHRINKTO 2047 start", "SHRINKTO 2047 end",
       "fit@4032 start", "fit@4032 end", "cur16 SHRINKTO 32 start", "cur16 SHRINKTO 32 end"]
FIXED = [0, 4096, 0, 64, 0, 64, 0, 1, 0, 17, 0, 2047, 4032, 4096, 16, 48]
PREFIX = [0, 4096, 0, 64, 0, 63, 0, 0, 0, 16, 0, 2046, 4032, 4095, 16, 47]
CONTROLS = 4   # positions 0-3 (parent, SHRINK) read the same on both trees


def main(argv):
    if len(argv) < 2:
        print(__doc__); return 2
    path = argv[1]
    expect = "prefix"
    if "--expect" in argv:
        expect = argv[argv.index("--expect") + 1]
    if expect not in ("prefix", "fixed"):
        print(f"--expect must be prefix or fixed, not {expect}"); return 2
    text = open(path, errors="replace").read().replace("\r", "")
    begins = [m.start() for m in re.finditer(r"SUPTEST BEGIN R\b", text)]
    if not begins:
        print(f"VOID: no 'SUPTEST BEGIN R' in {path}"); return 2
    block = text[begins[-1]:]
    sr = re.search(r"^SR ([0-9A-F]{16})$", block, re.M)
    if not sr:
        print(f"VOID: no SR line after the last BEGIN in {path}"); return 2
    n = int(sr.group(1), 16)
    vals = [int(v, 16) for v in re.findall(r"^SV ([0-9A-F]{16})$", block, re.M)]
    end = re.search(r"^SUPTEST END$", block, re.M) is not None
    print(f"SR {n}  SV lines {len(vals)}  END {'yes' if end else 'no'}")
    if end and len(vals) == n and CONTROLS < n < len(FIXED) and vals[:CONTROLS] == FIXED[:CONTROLS]:
        # the trap handler records mcause and ends the run: a fault in an arm is a reading, not a void boot
        print(f"FINDING: a trap after position {n - 2} ({POS[n - 2]}), mcause {vals[-1]:#x}; values {vals[:-1]}")
        return 1
    if n != len(FIXED) or len(vals) != n or not end:
        print(f"VOID: SR {n}, {len(vals)} values, END {end}; expected SR {len(FIXED)} with END"
              + (f" (a short vector ends in a trap's mcause: last value {vals[-1]:#x})" if vals else ""))
        return 2
    if vals[:CONTROLS] != FIXED[:CONTROLS]:
        print("VOID: the controls (parent bounds, SHRINK) do not read 0 4096 0 64:", vals[:CONTROLS]); return 2
    want, other = (PREFIX, FIXED) if expect == "prefix" else (FIXED, PREFIX)
    for i, (v, a, b) in enumerate(zip(vals, PREFIX, FIXED)):
        tag = "both" if a == b else ("PRE-FIX" if v == a else "FIXED" if v == b else "NEITHER")
        print(f"  {i:2d} {POS[i]:24s} {v:6d}  {tag}")
    if vals == want:
        print(f"AS EXPECTED: the {expect} vector (" + ("SHRINKTO holds imm - 1 bytes" if expect == "prefix"
              else "SHRINKTO holds imm bytes") + ")"); return 0
    if vals == other:
        print(f"NOT AS EXPECTED: the {'fixed' if expect == 'prefix' else 'pre-fix'} vector exactly"); return 1
    print("FINDING: neither known vector; read the positions above"); return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
