#!/usr/bin/env python3
"""Result lines of one board-r1e4.sh boot of the 2026-10-01 M1 campaigns, from its output directory.

Prints what a verdict needs and nothing that identifies the host: the k800 control lines, the domain's
R1 m1 lines (start/end/released/probe/stale), and, for a wedge, the trap cause, mepc - DBAS (the image
offset of the faulting instruction), tval, rev_node_head and the refusal record with its id split into
(generation = id[29:16], index = id[15:0]).

Exits 2 when the directory holds no driver.log, or a driver.log with no k800 result line: "no data" is an
error here, never an empty table.  Usage: extract.py <unify/board-<tag>-b1>
"""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # rtl-smoke/, for fpga_driver
from fpga_driver.transcript import read, scope_to_run, uart_text, TranscriptError

def wedge(log):
    v = {}
    for m in re.finditer(r"\[wedge\] sw=(\d+) [^\n]*?0x([0-9a-f]{2}) [01]{8}", log):
        v[int(m.group(1))] = int(m.group(2), 16)      # the LAST read of each aperture wins
    return v

def le(v, sws):
    if any(s not in v for s in sws): return None
    return sum(v[s] << (8 * i) for i, s in enumerate(sws))

def main(d):
    p = os.path.join(d, "driver.log")
    if not os.path.exists(p): sys.exit(f"extract: no driver.log in {d}")
    log = open(p, errors="replace").read()
    # THIS run's UART only: the console replays the previous boot on connect, so a whole-log count of
    # k800 lines includes the last boot's (m1v2-1 read 3 for a boot that ran two).
    try:
        run_uart = uart_text(scope_to_run(read(p)))
    except TranscriptError as e:
        print(f"extract: cannot scope {p} to this run: {e}", file=sys.stderr); sys.exit(2)
    k800 = re.findall(r"RESULT k800 retval=-?\d+ cycles=\d+ ran=\d+ instret=\d+", run_uart)
    if not k800: print(f"extract: no k800 RESULT line in {p} -- not a run, or the control never returned", file=sys.stderr); sys.exit(2)
    out = [f"# {os.path.basename(d.rstrip('/'))}: k800 lines {len(k800)}"]
    out += k800
    rl = os.path.join(d, "r1-lines.txt")
    r1 = [l.strip() for l in open(rl, errors="replace")] if os.path.exists(rl) else []
    keep = [l for l in r1 if re.match(r"R1 m1 (start|end|released|probe|stale|snap|lcc)", l)]
    out += keep
    out.append(f"# R1 m1 snapshot lines: {sum(l.startswith('R1 m1 snap') for l in r1)}")
    v = wedge(log)
    if not v:
        out.append("# no wedge dump: the run returned")
    else:
        dbas = re.findall(r"DBAS:([0-9A-F]{8})", run_uart)
        mepc = le(v, [196, 197, 198, 199, 200, 201, 202, 203]); tval = le(v, [210, 211, 213, 214, 215, 216, 217, 218])
        tl = v.get(255); head = le(v, [249, 250])
        base = int(dbas[-1], 16) if dbas else None
        out.append(f"# wedge: trap log 0x{tl:02x} -> seen={tl >> 7} mcause={tl & 0x7f}" if tl is not None else "# wedge: trap log UNREAD")
        out.append(f"# wedge: mepc 0x{mepc:x}, DBAS 0x{base:x} -> image+0x{mepc - base:x}" if (mepc is not None and base) else f"# wedge: mepc {mepc}, DBAS {base}: offset UNAVAILABLE")
        out.append(f"# wedge: tval 0x{tval:x}" if tval is not None else "# wedge: tval UNREAD")
        out.append(f"# wedge: rev_node_head[15:0] = {head}" if head is not None else "# wedge: rev_node_head UNREAD")
        serving = le(v, [251, 252, 253, 254])
        out.append(f"# wedge: rev_node_serving_idx[29:0] = {serving & 0x3FFFFFFF} (generation {(serving >> 16) & 0x3FFF}, index {serving & 0xFFFF})"
                   if serving is not None else "# wedge: rev_node_serving_idx UNREAD")
        if tval is not None:
            out.append(f"# wedge: tval[9:0] = 0x{tval & 0x3FF:03x} (the probed slot's offset in the pool: slot 15 of 64-byte leaves is 0x3c0)")
        rr = [l.strip() for l in log.split("\n") if l.strip().startswith("[refusal] wedge")]
        out += rr[-1:] or ["# refusal: no [refusal] wedge line"]
        idb = le(v, [205, 206, 207])
        if idb is not None and 208 in v and 204 in v:
            ident = idb | ((v[208] & 0x3F) << 24)
            out.append(f"# refusal id 0x{ident:08x}: generation {ident >> 16}, index {ident & 0xFFFF} (record byte 204 = 0x{v[204]:02x})")
    print("\n".join(out))

if __name__ == "__main__":
    if len(sys.argv) != 2: sys.exit(__doc__)
    main(sys.argv[1])
