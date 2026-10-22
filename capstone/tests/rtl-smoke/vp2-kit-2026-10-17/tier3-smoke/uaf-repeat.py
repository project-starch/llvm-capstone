#!/usr/bin/env python3
"""uaf-repeat.py --board-kit <kit> --tools <tools dir> --controls <controls.dom> --helper <juliet-timeout> --n <N> --out <fresh dir>

The Juliet runner's use-after-free control, repeated N times in ONE cold board boot (2026-10-22). On tree B it read
cause 13 (a load page fault) at control_read_probe+0x38 in 5 of 74 Juliet runs (temporal 2 of 24, spatial 3 of 48),
where tree A reads cause 25 (the revocation check) in 74 of 74; one-sided Fisher p ~ 0.03. Each run is the Juliet
gate's own command, byte for byte apart from its name: `./juliet-timeout <name> 10000 -- ./capstone-vexec
./images/controls.dom uaf-malloc`, with the same controls.dom, helper and kit launcher. Odd runs are untraced (as in
Juliet); even runs set CAPSTONE_VM_TRACE=1, so a cause-13 run has the launcher's service/boundary trace beside it.
A bounds-control run comes first, as in every Juliet batch. Every uaf step prints its fault line(s) (FL ...); the
whole output (TR ...) is printed for every cause-13 run and every tenth run, the tenth runs being the cause-25 traces
to compare against.
Judges nothing; writes <out>/steps.json and prints the tally: per arm (plain, traced) the count of runs with
cause 25, cause 13, other, and no fault line. Exits 2 unless every step is in the transcript."""
import argparse
import collections
import json
import re
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--board-kit", type=Path, required=True)
    p.add_argument("--tools", type=Path, required=True, help="bug-corpora/tools of a clean worktree")
    p.add_argument("--controls", type=Path, required=True)
    p.add_argument("--helper", type=Path, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.tools))
    import boardvm  # noqa: E402
    if a.out.exists():
        sys.exit(f"{a.out} exists; a run goes into a fresh directory")
    a.out.mkdir(parents=True)
    b = boardvm.BoardBatch(a.out / "stage")
    b.put("images/controls.dom", a.controls)
    b.put("juliet-timeout", a.helper)
    (a.out / "stage" / "juliet-timeout").chmod(0o755)
    b.bounds = {}
    # As in every Juliet batch, the bounds control runs first, so run 1 sits where ctl-uaf-0 sat (the second program
    # after the module loads), the position of the spatial set's three cause-13 readings.
    b.step("bounds-0", "./juliet-timeout bounds-0 10000 -- ./capstone-vexec ./images/controls.dom bounds-malloc")
    for k in range(1, a.n + 1):
        env = "CAPSTONE_VM_TRACE=1 " if k % 2 == 0 else ""
        name = f"uaf-{k}-{'traced' if env else 'plain'}"
        b.step(name, f"( {env}./juliet-timeout {name} 10000 -- ./capstone-vexec ./images/controls.dom uaf-malloc ) "
                     f"> /tmp/u.txt 2>&1; rc=$?; grep -a 'domain fault' /tmp/u.txt | sed 's/^/FL /'; "
                     f"if grep -q 'cause=13 ' /tmp/u.txt || [ {k % 10} = 0 ]; then tail -n 300 /tmp/u.txt | sed 's/^/TR /'; fi; "
                     f"exit $rc")
    steps, transcript, completed = b.execute(a.board_kit, a.out / "work", 3600 + 30 * a.n)
    rows = {n: {"result": r, "text": t} for n, (t, r) in steps.items()}
    (a.out / "steps.json").write_text(json.dumps({"completed": completed, "transcript": str(transcript),
                                                   "n": a.n, "steps": rows}, indent=1))
    tally = collections.Counter()
    for n, (t, r) in steps.items():
        if n == "bounds-0":
            print(f"BOUNDS-0 {r} {t.strip()[:200]}")
            continue
        causes = re.findall(r"^FL .*domain fault cause=(\d+) ", t, re.M)
        arm = n.rsplit("-", 1)[1]
        kind = "none" if not causes else causes[0] if causes[0] in ("13", "25") else "other"
        if len(causes) > 1:
            kind += "+more"
        tally[(arm, kind)] += 1
        if kind != "25":
            print(f"RUN {n} rc={r} causes={causes}")
    for (arm, kind), c in sorted(tally.items()):
        print(f"TALLY {arm} cause {kind}: {c}")
    print(f"completed={completed} steps={len(steps)} of {a.n + 1} transcript={transcript}")
    if len(steps) != a.n + 1:
        print(f"NO DATA: {len(steps)} of {a.n + 1} steps in the transcript")
        sys.exit(2)


if __name__ == "__main__":
    main()
