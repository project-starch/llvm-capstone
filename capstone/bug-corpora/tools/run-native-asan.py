#!/usr/bin/env python3
"""Run a corpus's ASan-instrumented native binaries, fixed then buggy, and classify each by the
report AddressSanitizer printed -- for the corpora whose cases link a port's own allocator
(ffmpeg/pool-repros, memcached/allocator-repros, wireshark/wmem-repros), which the per-corpus
native runners build without ASan.

    run-native-asan.py --corpus <dir> --bin '<path template>' --out <fresh dir>
        [--buggy-args 'buggy {n}'] [--fixed-args 'fixed {n}']
        --control '<binary> <args>'=<expected report> [--control ...]

`{name}` is the case directory name, `{nn}` its two-digit number and `{n}` the number without
leading zeros. The binaries must already be built with -fsanitize=address; the build commands live
in each corpus's runners/run-asan.sh, which calls this.

WHAT IS SCORED, AND WHAT IS REFUSED.
  * The report TYPE is read from `ERROR: AddressSanitizer: <type>` -- heap-buffer-overflow,
    heap-use-after-free, use-after-poison, ... Matching the word "AddressSanitizer" alone is not
    enough: LeakSanitizer's summary contains it, which once made a leaking probe read as a
    detection on every arm. Leak detection is switched off (detect_leaks=0); it is not the question.
  * Nothing is scored unless every --control produced its expected report IN THIS RUN, with the
    same compiler and flags. A silent case is only a reading if ASan demonstrably reports on the
    heap this binary uses.
  * A fixed arm must exit 0, print `VERDICT FIXED`, and carry no ASan report; otherwise the row is
    FIXED-ARM-FAILED and its buggy reading is not scored.
  * A buggy arm that neither reports nor prints `VERDICT DEFECT-REPRODUCED` is OTHER: silence is
    only "ASan did not see it" when the defect demonstrably ran.
  * Every case directory gets a row.

Exit 0 when every row has a reading (REPORTED or SILENT), 75 on a control or coverage failure.
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPORT = re.compile(r"ERROR: AddressSanitizer: ([A-Za-z-]+)")
FRAME0 = re.compile(r"^\s*#0 0x[0-9a-f]+ in (\S+) (\S+)", re.M)
ENV = dict(os.environ, ASAN_OPTIONS="detect_leaks=0:abort_on_error=0:halt_on_error=1:color=never")


def run(argv, log):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=300, env=ENV)
        text, rc = p.stdout + p.stderr, p.returncode
    except subprocess.TimeoutExpired:
        text, rc = "[runner] TIMEOUT", -1
    log.write_text(f"$ {shlex.join(argv)}\nrc={rc}\n{text}")
    m = REPORT.search(text)
    f = FRAME0.search(text) if m else None
    return rc, text, (m.group(1) if m else None), (f"{f.group(1)} {f.group(2)}" if f else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--bin", required=True, help="binary path template, e.g. '/x/bin/defect-{nn}'")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--buggy-args", default="buggy {n}")
    ap.add_argument("--fixed-args", default="fixed {n}")
    ap.add_argument("--control", action="append", default=[], required=True,
                    help="'<binary> <args>=<expected report type>'; repeatable")
    ap.add_argument("--build-note", default="", help="compiler and flags, recorded in record.json")
    a = ap.parse_args()
    if a.out.exists():
        print(f"CONTROL-FAILED {a.out} exists: use a fresh directory", file=sys.stderr)
        return 75
    logs = a.out / "runs"
    logs.mkdir(parents=True)

    controls = []
    for i, spec in enumerate(a.control):
        cmd, want = spec.rsplit("=", 1)
        rc, text, kind, frame = run(shlex.split(cmd), logs / f"control-{i}.log")
        ok = kind == want
        controls.append(dict(command=cmd, required=want, report=kind, frame=frame, rc=rc, ok=ok))
        print(f"  control {i}: {kind or 'no report'} (required {want}) {'ok' if ok else 'FAILED'}  {frame or ''}", flush=True)
        if not ok:
            print("CONTROL-FAILED: ASan did not report where it must; no case is a reading", file=sys.stderr)
            (a.out / "record.json").write_text(json.dumps(dict(controls=controls), indent=2) + "\n")
            return 75

    rows = []
    cases = sorted(d for d in a.corpus.glob("[0-9][0-9]_*") if d.is_dir())
    for d in cases:
        nn = d.name[:2]
        fmt = dict(name=d.name, nn=nn, n=str(int(nn)))
        binary = a.bin.format(**fmt)
        if not Path(binary).is_file():
            rows.append(dict(case=d.name, outcome="NOT-BUILT", detail=binary))
            print(f"  {d.name:<62} NOT-BUILT", flush=True)
            continue
        frc, ftext, fkind, _ = run([binary, *a.fixed_args.format(**fmt).split()], logs / f"{d.name}-fixed.log")
        brc, btext, bkind, bframe = run([binary, *a.buggy_args.format(**fmt).split()], logs / f"{d.name}-buggy.log")
        fixed_ok = frc == 0 and fkind is None and "VERDICT FIXED" in ftext
        if not fixed_ok:
            outcome = "FIXED-ARM-FAILED"
            detail = f"fixed rc={frc} report={fkind} verdict-line={'VERDICT FIXED' in ftext}"
        elif bkind:
            outcome, detail = "REPORTED", f"{bkind} at {bframe}"
        elif "VERDICT DEFECT-REPRODUCED" in btext:
            outcome, detail = "SILENT", "the defect reproduced (VERDICT DEFECT-REPRODUCED) and ASan printed nothing"
        else:
            outcome, detail = "OTHER", f"buggy rc={brc}, no report, no DEFECT-REPRODUCED line"
        rows.append(dict(case=d.name, outcome=outcome, detail=detail, fixed_rc=frc, buggy_rc=brc))
        print(f"  {d.name:<62} {outcome:<16} {detail[:100]}", flush=True)

    bad = [r["case"] for r in rows if r["outcome"] not in ("REPORTED", "SILENT")]
    (a.out / "record.json").write_text(json.dumps(dict(
        corpus=str(a.corpus), started=time.strftime("%Y-%m-%dT%H:%M:%S%z"), build=a.build_note,
        asan_options=ENV["ASAN_OPTIONS"], controls=controls, rows=rows), indent=2) + "\n")
    tally = {}
    for r in rows:
        tally[r["outcome"]] = tally.get(r["outcome"], 0) + 1
    print(f"\n{a.corpus.name}: {len(rows)} cases; {tally}; not a reading: {bad or 'none'}")
    return 75 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
