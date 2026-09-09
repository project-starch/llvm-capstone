#!/usr/bin/env python3
"""Extract the S-12 verdict fields from a board log, scoped to the TEST domain's own run.

TWO DEFECTS THIS EXISTS TO FIX, both found after they had already produced a wrong reading.

1. THE CONSOLE REPLAYS THE PREVIOUS BOOT. On connect it dumps a few hundred KB of the last
   session, so a whole-file grep matches the PREVIOUS run's markers. In ejf-2.log a single 3936-
   character replay line carries two DBAS values from a boot that is not the one under test.

2. UART OUTPUT IS CHUNK-SPLIT AT ARBITRARY BOUNDARIES. The test domain's DBAS line was emitted as
   `'DBA'` and then the rest, so `grep -ao "DBAS:[0-9A-Fa-f]*"` matched neither piece. It did not
   fail: it fell back to the LAST value it could see, which was the CONTROL domain's DBAS, and
   reported 0x82400000 for a run whose test domain was at 0x82800000. Under the wrong base the
   faulting mepc maps ~5 MB outside the image -- so a genuine in-function fault would have been
   classified NEG-other. This is the shape that silently converts a positive into a negative.

So: reassemble the UART payloads into one continuous stream, cut it at the TEST stage marker, and
match inside that. Every field is reported as UNKNOWN rather than defaulted, because a defaulted
base is worse than no base -- it yields an address that looks real.
"""
import argparse, re, sys

# `[fpga] [uart] '....'` -- the payload is a python-repr string, possibly split mid-token.
UART = re.compile(r"\[uart\]\s+'(.*)'\s*$")
STAGE = re.compile(r"\[stages\]\s+-->\s+TEST\s+(\d+)/(\d+)\s+(\S+)")


def uart_stream(path):
    """All UART payloads concatenated, plus the offset of each source line."""
    buf = []
    for line in open(path, errors="replace"):
        m = UART.search(line.rstrip("\n"))
        if m:
            buf.append(m.group(1).encode().decode("unicode_escape", errors="replace"))
    return "".join(buf)


def stage_regions(path):
    """[(name, uart_offset, line_no)] -- where each TEST stage begins.

    Two offsets because the fields come from two different places. DBAS/DENT/G-enter are printed
    by the guest over UART and must be located in the REASSEMBLED stream (chunk splitting). The
    trap latch and the GDB CSR reads are printed by the driver on its own lines, after the stage
    finishes, so they are located by LINE NUMBER. Scoping one by the other's coordinate silently
    finds nothing.
    """
    out, sofar = [], 0
    for ln, line in enumerate(open(path, errors="replace")):
        s = line.rstrip("\n")
        m = UART.search(s)
        if m:
            sofar += len(m.group(1).encode().decode("unicode_escape", errors="replace"))
            continue
        m = STAGE.search(s)
        if m:
            out.append((m.group(3), sofar, ln))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--stage", default=None,
                    help="substring of the TEST stage to scope to; default = the LAST stage")
    a = ap.parse_args()

    stream = uart_stream(a.log)
    if not stream:
        print("ERROR: no UART payloads in this log at all. That is a harness failure, not a "
              "verdict.", file=sys.stderr)
        return 2
    regions = stage_regions(a.log)
    if not regions:
        print("ERROR: no `[stages] --> TEST` marker. Cannot tell the test domain's output from "
              "the control's or from the replayed previous boot; refusing to guess.",
              file=sys.stderr)
        return 2

    if a.stage:
        picked = [r for r in regions if a.stage in r[0]]
        if not picked:
            print(f"ERROR: no stage matching {a.stage!r}. Stages present: "
                  f"{[n for n, _ in regions]}", file=sys.stderr)
            return 2
        name, start, startln = picked[-1]
    else:
        name, start, startln = regions[-1]
    scoped = stream[start:]
    tail = "".join(l for i, l in enumerate(open(a.log, errors="replace")) if i >= startln)

    print(f"stage: {name}")
    print(f"  scoped to {len(scoped)} of {len(stream)} UART chars "
          f"(everything before is the control or the replayed previous boot)")

    fields = {}
    for key, pat, where in (("DBAS", r"DBAS:([0-9A-Fa-f]{8})", scoped),
                            ("DENT", r"DENT:([0-9A-Fa-f]{8})", scoped),
                            ("enter", r"(SQ: G/enter)", scoped),
                            ("mepc", r"trap mepc = (0x[0-9a-f]+)", tail),
                            ("tval", r"trap tval = (0x[0-9a-f]+)", tail),
                            ("traplog", r"TRAP LOG[^\n]*?(0x[0-9a-f]+)", tail),
                            ("gdb", r"(gdb CSRs: mcause=\S+ mepc=\S+ mtval=\S+)", tail)):
        m = re.findall(pat, where)
        fields[key] = m[-1] if m else None
        print(f"  {key:8s} {fields[key] if fields[key] is not None else 'UNKNOWN'}")

    if fields["DBAS"] and fields["mepc"]:
        va = int(fields["mepc"], 16) - int(fields["DBAS"], 16) + 0x10000
        print(f"  VA       0x{va:x}   (mepc - DBAS + 0x10000)")
        if not (0x10000 <= va < 0x10000 + 0x200000):
            print("  WARNING: that VA is outside any plausible image extent. Either the DBAS "
                  "belongs to a different domain or the mepc does. Do NOT classify on it.")
    else:
        print("  VA       UNKNOWN -- missing DBAS or mepc; no classification is possible.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
