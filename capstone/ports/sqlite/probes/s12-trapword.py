#!/usr/bin/env python3
"""Decode the in-domain trap report word written by start-gp-captable-interp.S.

The handler packs a fault into the 32-bit `res` word the monitor passes back:

    bits 31..28  marker.  0xF = trapped.  0xE = trapped AND a4's cursor was NON-ZERO
                 (only built when INTERP_TRAP_A4 is defined; see below)
    bits 27..22  mcause & 0x3F
    bits 21..0   (mepc - _start) >> 2  -- the faulting instruction as an image offset

Any other top nibble is an ORDINARY RESULT, not a fault, and this tool says so rather than
decoding it into a plausible-looking fault.

THE a4 BIT.  With -DINTERP_TRAP_A4 the handler reads the architectural a4 at the fault and
folds the answer into the marker.  The two open accounts of S-12 predict opposite values:

    0xF  a4 cursor == 0     the ldc itself delivered the stored null   -> write-buffer forward
    0xE  a4 cursor != 0     the ldc landed; only the CONSUMER saw a stale value -> ordering escape

They do not overlap.  CVA6 retires in order and writes the GPR at commit, so the stc's
spuriously-aliased write (decoder.sv:1313 gives stc rd := rs2) commits BEFORE the younger ldc
and cannot be the last writer; and the consumer's exception is taken at commit, by which time
the ldc has certainly committed.  So the regfile value at the handler is the ldc's.

Usage:  s12-trapword.py <word> [<word> ...]      words as 0x... or decimal
        s12-trapword.py --selftest
"""
import sys

MARKERS = {
    0xF: ("TRAPPED", "a4 cursor ZERO (or the build has no INTERP_TRAP_A4)"),
    0xE: ("TRAPPED", "a4 cursor NON-ZERO -> ldc landed, consumer saw stale"),
}


def decode(word):
    w = word & 0xFFFFFFFF
    marker = (w >> 28) & 0xF
    if marker not in MARKERS:
        return None
    return {
        "word": w,
        "marker": marker,
        "mcause": (w >> 22) & 0x3F,
        "offset_insns": w & 0x3FFFFF,
        "offset_bytes": (w & 0x3FFFFF) * 4,
        "note": MARKERS[marker][1],
    }


def report(word):
    d = decode(word)
    if d is None:
        print(f"0x{word & 0xFFFFFFFF:08X}  NOT A TRAP REPORT -- ordinary result value "
              f"(top nibble 0x{(word >> 28) & 0xF:X}); do not decode as a fault")
        return d
    print(f"0x{d['word']:08X}  TRAPPED  mcause={d['mcause']}  "
          f"_start+0x{d['offset_bytes']:X} ({d['offset_insns']} insns)")
    print(f"            marker 0x{d['marker']:X}: {d['note']}")
    return d


def selftest():
    # Positive control: the word actually measured on the board, 3/3 draws, from the
    # -DINTERP_DOMAIN_MTVEC build WITHOUT the a4 bit.  mcause 25 at image offset 0xF4874.
    d = decode(0xF643D21D)
    assert d is not None and d["mcause"] == 25, d
    assert d["offset_bytes"] == 0xF4874, hex(d["offset_bytes"])
    # The a4 marker must decode as a trap too, with the SAME mcause and offset -- if the two
    # markers disagreed on anything but the note, the arm's results would be incomparable.
    e = decode(0xE643D21D)
    assert e is not None and (e["mcause"], e["offset_bytes"]) == (25, 0xF4874), e
    # Negative control: a real result word must NOT decode as a fault.  A tool that renders
    # every input as a fault would make the whole arm unreadable.
    assert decode(0x5122) is None
    assert decode(0x21178DE9) is None
    print("selftest OK: 0xF and 0xE decode alike (mcause 25 @ _start+0xF4874); "
          "non-marker words are refused")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(2)
    if args[0] == "--selftest":
        sys.exit(selftest())
    for a in args:
        report(int(a, 0))
