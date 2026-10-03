#!/usr/bin/env python3
"""Verdict for one sup-capstl run (or the call-retpc control). Usage: compare.py <run dir with uart.txt> [control].
Exit 0 COMPLETED-AS-PREDICTED, 1 MISS (completed with a wrong value), 3 HANG (SUPTEST BEGIN/SB1 seen, no END: the
dots say how far it got), 2 NO-RESULT (no uart.txt or no SB1 -- the harness never started the test)."""
import re, sys, pathlib

def main(d, control=False):
    p = pathlib.Path(d) / "uart.txt"
    if not p.exists():
        print(f"NO-RESULT: no {p}"); return 2
    t = p.read_text(errors="replace")
    if "SB1" not in t:
        print("NO-RESULT: no SB1 (the test never started)"); return 2
    dots = t[t.find("SB1"):].count(".")
    if "SUPTEST END" not in t:
        print(f"HANG after {dots} dots, i.e. >= {256 * dots} escapes"); return 3
    sv = [int(x, 16) for x in re.findall(r"SV ([0-9A-F]{16})", t)]
    if control:
        ok = sv == [0, 0x12, 0x21, 0, 0x51, 0, 0, 0]
        print(("PASS exact" if ok else "MISS") + f" {[hex(x) for x in sv]}"); return 0 if ok else 1
    if len(sv) != 8:
        print(f"MISS: {len(sv)} readings {[hex(x) for x in sv]} (a trap ends with mcause, mepc)"); return 1
    nf0, st, it, ck, esc, nf1, mc, end = sv
    ok = st == 1 and it == 0x1000 and ck == 0x7F800 and esc > 0 and mc == 0 and end == 0x5E5E
    print(("COMPLETED" if ok else "MISS") + f": status {st:#x} iter {it:#x} checksum {ck:#x} escapes {esc} (dots {dots}) "
          f"csnodefree {nf0:#x} -> {nf1:#x} mcause {mc:#x} end {end:#x}")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main(sys.argv[1], len(sys.argv) > 2 and sys.argv[2] == "control"))
