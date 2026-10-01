#!/usr/bin/env python3
"""Verdict lines for one board-s1sql.sh boot, read against cells.tsv.

  extract.py <unify/board-s1sql-b<N><tag>> <BOOT>

Reads the run-scoped transcript (boot.txt: one `===== <label> =====` block per invocation, in run order)
and, for the wedge, driver.log's debug-mux dump. Cells are matched to blocks by ORDER and checked by
LABEL (image and `--s1-probe n`), so a block that ran something else is a MISMATCH, never a pass.

A returning cell PASSES only when its block has all four:
- `S1-PROBE <n>`, the harness's own line saying which probe ran (a dropped flag runs the benchmark and
  prints none);
- the cell's marker text (cells.tsv `want`, before any parenthesis);
- `SQ: H/return`;
- `SQ: obs=0`.
The host ALSO prints `speedtest1 did not run` for obs=0, because it expects the benchmark's marker.
That line means nothing here.

The fault cell is judged by the trap log (0x99 = cause 25, 0x9c = 28) and by mepc - DBAS equal to the
pre-registered offset. Its own text is lost in the wedge, so no S1-PROBE line is expected. A protected
cell that printed its NOTRAP marker is UNSAFE-SUCCESS. The refusal record is printed as read.
A record arm of `1000` (timeout) or an EMPTY record voids the lifetime reading for that boot.

Exits 2 with no boot.txt, no blocks, or no k800 line: "no data" is an error, never a clean table.
"""
import os, re, sys

def nodata(msg):
    print(f"extract: {msg}", file=sys.stderr); sys.exit(2)

def wedge(log):
    v = {}
    for m in re.finditer(r"\[wedge\] sw=(\d+) [^\n]*?0x([0-9a-f]{2}) [01]{8}", log):
        v[int(m.group(1))] = int(m.group(2), 16)
    return v

def le(v, sws):
    return None if any(s not in v for s in sws) else sum(v[s] << (8 * i) for i, s in enumerate(sws))

def main(out, boot):
    here = os.path.dirname(os.path.abspath(__file__))
    cells = [l.split("\t") for l in open(os.path.join(here, "cells.tsv")).read().splitlines() if l and not l.startswith("#")]
    order = [c for c in cells if c[6] == "all"] + [c for c in cells if c[6] == str(boot)]
    bt = os.path.join(out, "boot.txt")
    if not os.path.exists(bt): nodata(f"no {bt}")
    text = open(bt, errors="replace").read().replace("\r", "")
    parts = re.split(r"^===== (.+) =====$", text, flags=re.M)
    blocks = list(zip(parts[1::2], parts[2::2]))
    if not blocks: nodata(f"no ===== blocks in {bt}")
    lines, bad = [f"# {os.path.basename(out.rstrip('/'))}, boot {boot}: {len(blocks)} blocks, {len(order)} cells + k800"], 0
    k = re.findall(r"RESULT k800 retval=-?\d+ cycles=\d+ ran=\d+ instret=\d+", blocks[0][1])
    if not k or not blocks[0][0].startswith("k800"):
        print("\n".join(lines + ["k800: NO RESULT LINE in the first block -- VOID"])); sys.exit(2)
    lines.append(k[0] + ("  [control ok]" if "retval=4" in k[0] else "  [CONTROL FAILED: boot VOID]"))
    for i, c in enumerate(order):
        name, arm, probe, va, h, hostargs, cboot, want = c
        if i + 1 >= len(blocks):
            lines.append(f"{name}: NOT RUN (no block)"); bad += 1; continue
        label, body = blocks[i + 1]
        flat = body.replace("\n", "")
        if f"s1-{arm}.dom" not in label or not re.search(rf"--s1-probe {probe}(\s|$)", label):
            lines.append(f"{name}: MISMATCH -- block {i+1} ran [{label[:90]}]"); bad += 1; continue
        if want.startswith("FAULT"):
            unsafe = "NOTRAP" in body
            lines.append(f"{name}: {'UNSAFE-SUCCESS (NOTRAP printed)' if unsafe else 'no NOTRAP line'}; "
                         f"returned={'SQ: H/return' in body}; S1-PROBE={'S1-PROBE ' + probe in body}")
            continue
        mk = want.split(" (")[0].strip()
        got = {"S1-PROBE": re.search(rf"S1-PROBE {probe}\b", body) is not None,
               "marker": mk in body or mk in flat, "H/return": "SQ: H/return" in body, "obs=0": re.search(r"SQ: obs=0\b", body) is not None}
        val = next((m for m in re.finditer(r"__CAPSTONE_SPEEDTEST1_[A-Z_]+__[^\n]*", body) if mk in m.group(0)), None) or re.search(r"__CAPSTONE_SPEEDTEST1_[A-Z_]+__[^\n]*", body)
        ok = all(got.values())
        bad += 0 if ok else 1
        lines.append(f"{name}: {'PASS' if ok else 'FAIL ' + str({k2: v2 for k2, v2 in got.items() if not v2})} | {val.group(0)[:100] if val else 'no marker line'}")
    log = open(os.path.join(out, "driver.log"), errors="replace").read() if os.path.exists(os.path.join(out, "driver.log")) else ""
    v = wedge(log)
    fault = [c for c in order if c[7].startswith("FAULT")]
    if fault:
        c = fault[0]; m = re.match(r"FAULT (\d+) \+0x([0-9a-f]+)", c[7]); want_cause, want_off = int(m.group(1)), int(m.group(2), 16)
        if not v:
            lines.append(f"{c[0]}: NO WEDGE DUMP -- the fault cell did not wedge (see its block) or the dump was not read"); bad += 1
        else:
            tl = v.get(255); mepc = le(v, [196, 197, 198, 199, 200, 201, 202, 203])
            dbas = re.findall(r"DBAS:([0-9A-F]{8})", log); base = int(dbas[-1], 16) if dbas else None
            off = (mepc - base) if (mepc is not None and base) else None
            cause = (tl & 0x7f) if tl is not None else None
            ok = cause == want_cause and off == want_off
            bad += 0 if ok else 1
            lines.append(f"{c[0]}: {'PASS' if ok else 'FAIL'} -- trap log {('0x%02x' % tl) if tl is not None else 'UNREAD'} (cause {cause}), "
                         f"mepc {('0x%x' % mepc) if mepc is not None else 'UNREAD'} - DBAS {('0x%x' % base) if base else '?'} = "
                         f"{('+0x%x' % off) if off is not None else '?'}; pre-registered cause {want_cause} at +0x{want_off:x}")
            rr = [l.strip() for l in log.split("\n") if l.strip().startswith("[refusal] wedge")]
            lines += rr[-1:] or ["# refusal: no [refusal] wedge line"]
            if 204 in v:
                arm = (v[204] >> 2) & 0xF
                note = {1: "hit-dead", 2: "same-cycle invalidation", 4: "probe resolved dead", 8: "TIMEOUT: voids the lifetime reading"}.get(arm, "EMPTY or multi-hot: voids the lifetime reading" if want_cause == 25 else "not a revocation verdict (expected for a bounds fault)")
                lines.append(f"# refusal arm {arm:04b}: {note}")
    print("\n".join(lines)); print(f"# verdict: {'ALL AS PRE-REGISTERED' if bad == 0 else str(bad) + ' cell(s) NOT as pre-registered'}")
    sys.exit(0 if bad == 0 else 1)

if __name__ == "__main__":
    if len(sys.argv) != 3: sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
