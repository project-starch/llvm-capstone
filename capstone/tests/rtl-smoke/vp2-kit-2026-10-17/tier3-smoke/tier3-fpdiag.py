#!/usr/bin/env python3
"""tier3-fpdiag.py --board-kit <board kit> <out dir>: the runtime lane's FP discriminator (2026-10-26) in ONE boot. Its stage
(perf/fpdiag/board-stage: run-fpdiag.sh 1991d4a73faa93c8, the FINAL stage of 17:52; fp-escape and fp-divsqrt native and in a
context, ludcmp and st native and in a context)
is checked against its SHA256SUMS and staged whole, and run-fpdiag.sh is the one batch step.
Reads each `@@FP <where> <what>` section to its `@@FP rc=` line: the FPESC / FPDIV line's bad= (fp-escape, fp-divsqrt), the BENCH lines and
their VERIFY-FAILED count and BENCH-DONE failed= (ludcmp, st), any domain fault, rc, and a context's bench dmesg quote.
Exit 0 when the boot completed and every section of the runner's revision plus @@FP-DONE was read, 2 otherwise (no data is an error)."""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/tmp/capstone/wt-newarms/capstone/bug-corpora/tools")
import boardvm  # noqa: E402

D = Path("/mnt/hdd") / Path.home().name / "capstone/perf/fpdiag"
BASE = ("native n=100000", "native n=50000000", "context n=100000", "context n=50000000")
TAIL = ("native ludcmp", "native st", "context ludcmp", "context st")
DIV = ("native divsqrt n=100000", "native divsqrt n=20000000", "context divsqrt n=100000", "context divsqrt n=20000000")
# The sections each run-fpdiag.sh revision prints, by its sha256/16 (the runtime lane added fp-divsqrt at 17:52).
SECTIONS_BY_SHA = {"a19e3a68314d6809": BASE + TAIL, "1991d4a73faa93c8": BASE + DIV + TAIL}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board-kit", type=Path, required=True)
    ap.add_argument("--runner-sha", default="1991d4a73faa93c8", choices=sorted(SECTIONS_BY_SHA))
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    SECTIONS = SECTIONS_BY_SHA[a.runner_sha]
    stage = D / "board-stage"
    chk = subprocess.run(["sha256sum", "-c", "--quiet", "../SHA256SUMS"], cwd=stage, capture_output=True, text=True)
    if chk.returncode != 0:
        print(f"FPDIAG STOP: the stage does not match its SHA256SUMS: {chk.stdout[-300:]}{chk.stderr[-300:]}"); sys.exit(2)
    if hashlib.sha256((stage / "run-fpdiag.sh").read_bytes()).hexdigest()[:16] != a.runner_sha:
        print(f"FPDIAG STOP: run-fpdiag.sh is not {a.runner_sha}"); sys.exit(2)
    a.out.mkdir(parents=True, exist_ok=False)
    b = boardvm.BoardBatch(a.out / "stage")
    st = b.put("fpdiag", stage)
    b.step("fpdiag", f"sh {st}/run-fpdiag.sh")
    b.bounds = {"fpdiag": 2700 / 20}      # x BOARD_STEP_MULT 20 = 2700 s
    runs, transcript, completed = b.execute(a.board_kit, a.out / "guest", timeout=3000)
    text, result = runs.get("fpdiag", ("", {"kind": "none"}))
    (a.out / "fpdiag.txt").write_text(text)
    print(f"FPDIAG kit {a.board_kit} boot {'completed' if completed else 'DID NOT COMPLETE'}; transcript {transcript}; "
          f"step result {json.dumps(result, sort_keys=True)}")
    rc = 0 if completed and result.get("kind") != "none" else 2
    for name in SECTIONS:
        m = re.search(rf"^@@FP {re.escape(name)}$\n(.*?)^@@FP rc=(\d+) ?([^\n]*)$", text, re.M | re.S)
        if not m:
            print(f"READ {name}: NO DATA"); rc = 2; continue
        body, src, quote = m.group(1), m.group(2), m.group(3).strip()
        fe = re.search(r"^(?:FPESC|FPDIV) n=(\d+) bad=(\d+)(.*)$", body, re.M)
        bench = re.findall(r"^BENCH \S+ it=\S+ .*$", body, re.M)
        vf = len(re.findall(r"VERIFY-FAILED", body))
        dn = re.search(r"^BENCH-DONE \S+ failed=(\d+)$", body, re.M)
        flt = re.search(r"domain fault cause=(\d+)", body)
        parts = [f"rc {src}"]
        if fe:
            parts.append(f"n={fe.group(1)} bad={fe.group(2)} ::{fe.group(3)[:100]}")
        if bench or dn:
            parts.append(f"{len(bench)} BENCH, VERIFY-FAILED {vf}, BENCH-DONE failed={dn.group(1) if dn else 'ABSENT'}")
        if flt:
            parts.append(f"FAULT cause {flt.group(1)}")
        if quote:
            parts.append(f"quote {quote[:100]}")
        print(f"READ {name}: " + "; ".join(parts))
        for l in bench:
            print(f"READ {name} line: {l[:160]}")
    if not re.search(r"^@@FP-DONE$", text, re.M):
        print("READ: @@FP-DONE ABSENT"); rc = 2
    if re.search(r"Oops|Unable to handle|BUG:", text):
        print("READ: KERNEL OOPS in the step's text"); rc = 2
    sys.exit(rc)


if __name__ == "__main__":
    main()
