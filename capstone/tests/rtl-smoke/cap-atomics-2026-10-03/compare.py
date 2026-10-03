#!/usr/bin/env python3
"""Verdict for one cap-atomics board run: compare the board's SV readings with the pre-registered vector.
Usage: compare.py <run dir holding uart.txt>. Exit 0 PASS, 1 MISS, 2 NO-RESULT (no uart.txt, no SUPTEST END,
or a count that disagrees with the SV lines) -- never a pass."""
import re, sys, pathlib

NZ = "non-zero"
EXPECTED = [0x1111, 0x1111, 0x1116, 0x1116, 0xABCD,                           # 1-5   plain control, amoadd.d, amoswap.d
            0x7FFFFFFF, 0xFFFFFFFF80000000, 0xFFFFFFFF80000000, 0x12345678,   # 6-9   amoadd.w (wrap), amoswap.w
            0x12345678, 0, 0x55, NZ, 0x55,                                    # 10-14 lr.w/sc.w ok, sc.w without reservation
            NZ, 0xB0B0,                                                       # 15-16 sc.w to another granule
            0xABCD, 0, 0x1234,                                                # 17-19 lr.d/sc.d
            1, 0x99, 0xA70D]                                                  # 20-22 a_cas loop, end marker


def main(d):
    p = pathlib.Path(d) / "uart.txt"
    if not p.exists():
        print(f"NO-RESULT: no {p}"); return 2
    t = p.read_text(errors="replace")
    if "SUPTEST END" not in t:
        print(f"NO-RESULT: no SUPTEST END (BEGIN={'SUPTEST BEGIN' in t}, SB1={'SB1' in t})"); return 2
    sr = re.findall(r"SR ([0-9A-F]{16})", t)
    sv = [int(x, 16) for x in re.findall(r"SV ([0-9A-F]{16})", t)]
    if not sr or int(sr[0], 16) != len(sv):
        print(f"NO-RESULT: SR {sr} != {len(sv)} SV lines"); return 2
    bad = []
    for i, e in enumerate(EXPECTED):
        b = sv[i] if i < len(sv) else None
        ok = b is not None and (b != 0 if e == NZ else b == e)
        if not ok:
            bad.append((i + 1, None if b is None else hex(b), e if e == NZ else hex(e)))
    if len(sv) != len(EXPECTED):
        bad.append(("count", len(sv), len(EXPECTED)))
    print(("PASS" if not bad else "MISS") + f" ({len(sv)} readings); board {[hex(x) for x in sv]}"
          + (f"; diffs (n, board, expected) {bad}" if bad else ""))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
