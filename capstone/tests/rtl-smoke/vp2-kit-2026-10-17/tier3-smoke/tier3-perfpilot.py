#!/usr/bin/env python3
"""tier3-perfpilot.py --board-kit <board kit> <out dir>: the perf evaluation's PILOT (the runtime lane's
PREREG-t1-t2-sampling.md rev 2, "The pilot"; 2026-10-26) in ONE boot through boardvm. The runtime lane's stage
(perf/pilot/kit/stage: capstone-vexec, run-pilot.sh, T1/ and T2/ with 16 images each) is checked against its SHA256SUMS
and staged whole; the one batch step runs run-pilot.sh, which runs 68 processes in steps.tsv's order.
Reads, per process: its @@PERF-BEGIN/@@PERF-END pair, rc, the number of BENCH lines between them, whether each BENCH line
carries check=, and the bench dmesg line the END line quotes. run-pilot.sh quotes dmesg's LAST bench line, so a process
that printed none would carry the previous process's: that is checked here (a repeated quote is reported). The
per-iteration analysis is the runtime lane's. Exit 0 when the boot completed and all 68 processes were read with
PILOT-DONE, 2 otherwise (no data is an error, not a zero)."""
import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/tmp/capstone/wt-newarms/capstone/bug-corpora/tools")
import boardvm  # noqa: E402

KIT = Path("/mnt/hdd") / Path.home().name / "capstone/perf/pilot/kit"
RUNNER_SHA = "7cccbc7b9d067f19"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board-kit", type=Path, required=True)
    ap.add_argument("--pilot-dir", type=Path, default=KIT, help="holds stage/, SHA256SUMS and the steps file (default: the first pilot)")
    ap.add_argument("--runner-sha", default=RUNNER_SHA, help="run-pilot.sh's sha256/16")
    ap.add_argument("--steps", default="steps.tsv", help="the steps file's name in --pilot-dir")
    ap.add_argument("--runner", default="run-pilot.sh", help="the stage's run script (a main-run kit names it run-k<n>.sh)")
    ap.add_argument("--done", default="PILOT", help="the run script's done marker is @@PERF-<this>-DONE")
    ap.add_argument("--step-s", type=int, default=2700, help="the step's bound in seconds (the boot's budget is this + 300)")
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    stage = a.pilot_dir / "stage"
    chk = subprocess.run(["sha256sum", "-c", "--quiet", str(a.pilot_dir / "SHA256SUMS")], cwd=stage, capture_output=True, text=True)
    if chk.returncode != 0:
        print(f"PILOT STOP: the stage does not match its SHA256SUMS: {chk.stdout[-300:]}{chk.stderr[-300:]}"); sys.exit(2)
    if hashlib.sha256((stage / a.runner).read_bytes()).hexdigest()[:16] != a.runner_sha:
        print(f"PILOT STOP: {a.runner} is not {a.runner_sha}"); sys.exit(2)
    plan = list(csv.DictReader(open(a.pilot_dir / a.steps), delimiter="\t"))
    a.out.mkdir(parents=True, exist_ok=False)
    b = boardvm.BoardBatch(a.out / "stage")
    pilot = b.put("pilot", stage)
    b.step("pilot", f"sh {pilot}/{a.runner}")
    b.bounds = {"pilot": a.step_s / 20}     # x the kit's BOARD_STEP_MULT (20) = --step-s (2700 s, 45 min, for a pilot)
    runs, transcript, completed = b.execute(a.board_kit, a.out / "guest", timeout=a.step_s + 300)   # boardvm: script budget = --step-s; board-run.sh: x BOARD_TIMEOUT_MULT 4 + 1800 s
    text, result = runs.get("pilot", ("", {"kind": "none"}))
    print(f"PILOT kit {a.board_kit} boot {'completed' if completed else 'DID NOT COMPLETE'}; transcript {transcript}; "
          f"step result {json.dumps(result, sort_keys=True)}")
    (a.out / "pilot.txt").write_text(text)
    rc = 0 if completed and result.get("kind") != "none" else 2
    bp = re.findall(r"^(bench_\w+)$", text, re.M)
    print(f"READ bench parameters: {len(bp)} {bp}")
    begins = {int(m.group(1)): m.end() for m in re.finditer(r"^@@PERF-BEGIN (\d+) .*$", text, re.M)}
    ends = {int(m.group(1)): m for m in re.finditer(r"^@@PERF-END (\d+) (\S+) (\S+) (\d+) (\S+) rc=(\d+) ?(?:stderr_lines=(\d+) ?)?(.*)$",
                                                     text, re.M)}
    prev_quote, nok = None, 0
    for s in plan:
        n = int(s["n"])
        e = ends.get(n)
        if n not in begins or e is None:
            print(f"READ {n} {s['program']} {s['arm']} {s['proc']} K{s['K']}: NO DATA (begin {n in begins}, end {e is not None})")
            rc = 2
            continue
        body = text[begins[n]:e.start()]
        bench = re.findall(r"^BENCH .*$", body, re.M)
        checks = sum(1 for l in bench if "check=" in l)
        same = (e.group(2), e.group(3), e.group(4), e.group(5)) == (s["program"], s["arm"], s["proc"], f"K{s['K']}")
        quote = e.group(8).strip()
        errl = e.group(7)
        repeat = quote and quote == prev_quote
        prev_quote = quote
        tails = sorted({m.group(1) for m in re.finditer(r"check=0x[0-9a-f]+(.*)$", body, re.M)})
        errs = len(re.findall(r"^BENCH-ERROR ", body, re.M))
        dn = re.search(r"^BENCH-DONE \S+ failed=(\d+)$", body, re.M)
        want = sum(int(x) for x in s.get("args", "1 10").split())     # W warm-up + I timed iterations
        vfail = len(re.findall(r"VERIFY-FAILED", body))
        ok = (e.group(6) == "0" and len(bench) == want and checks == want and same and not repeat and errs == 0
              and vfail == 0 and dn is not None and dn.group(1) == "0")
        nok += ok
        print(f"READ {n} {s['program']} {s['arm']} {s['proc']} K{s['K']}: rc {e.group(6)}, {len(bench)} BENCH (want {want}), "
              f"VERIFY-FAILED {vfail}, "
              f"{checks} with check= (suffixes {tails}), BENCH-ERROR {errs}, BENCH-DONE failed={dn.group(1) if dn else 'ABSENT'}, "
              f"plan {'matches' if same else 'DIFFERS'}, quote {'REPEATS the previous' if repeat else 'new'}"
              + (f", stderr_lines {errl}" if errl is not None else "") + f" :: {quote[:110]}")
    done = re.search(rf"^@@PERF-{re.escape(a.done)}-DONE$", text, re.M)
    print(f"READ summary: {nok} of {len(plan)} processes rc 0, W+I BENCH lines, no VERIFY-FAILED or BENCH-ERROR, BENCH-DONE failed=0 and a new quote; {a.done}-DONE "
          f"{'present' if done else 'ABSENT'}")
    if re.search(r"Oops|Unable to handle|BUG:", text):
        print("READ: KERNEL OOPS in the step's text"); rc = 2
    if not done or len(ends) != len(plan):
        rc = 2
    sys.exit(rc)


if __name__ == "__main__":
    main()
