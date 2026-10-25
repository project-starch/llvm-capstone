#!/usr/bin/env python3
"""trace-smoke.py --board-kit <kit> --tools <tools dir> --image <mruby .dom> --smoke <smoke.rb> --out <fresh dir> [--runs 4]

mruby's stock smoke, N times in one cold board boot, with the launcher's service trace on (CAPSTONE_VM_TRACE=1: one
`CAPSTONE_VM_SERVICE tid= nr= result=` line per delegated system call, on stderr). It names the delegated call the
stall repeats. The 2026-10-21 capture read the stalled context alternating thread_self (service 0xb) and
delegate_transport (service 0xa) ~40 times a second, and could not say which call. Each run:
- smoke-<k>: the smoke, bounded at 600 s (30 s x BOARD_STEP_MULT 20), its stdout the step's text, its stderr (grants,
  the trace) to /tmp/tr<k>.txt in guest RAM. A bound fires the stall capture (boardvm STALL_FN) once per boot.
- sum-<k>: unbounded and fast. The trace's line count, the SERVICE line count, the 25 most frequent (nr, result)
  pairs, the other line kinds' counts, and the trace's last 40 lines. Its text is the reading, and the transcript keeps
  it whole.
The smoke's own text (M markers, SMOKE_DONE) is in smoke-<k>; the summary never re-prints it. Judges nothing; prints
the step table and writes <out>/steps.json."""
import argparse
import json
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--board-kit", type=Path, required=True)
    p.add_argument("--tools", type=Path, required=True, help="bug-corpora/tools of a clean worktree")
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--smoke", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--runs", type=int, default=4)
    a = p.parse_args()
    sys.path.insert(0, str(a.tools))
    import boardvm  # noqa: E402
    if a.out.exists():
        sys.exit(f"{a.out} exists; a run goes into a fresh directory")
    a.out.mkdir(parents=True)
    b = boardvm.BoardBatch(a.out / "stage")
    dom = b.put("images/mruby.dom", a.image)
    rb = b.put("smoke.rb", a.smoke)
    b.bounds = {}
    for k in range(1, a.runs + 1):
        b.step(f"smoke-{k}", f"rm -rf /tmp/s{k} && mkdir /tmp/s{k} && cd /tmp/s{k} && "
                             f"CAPSTONE_VM_TRACE=1 /test-domains/{boardvm.BATCH}/capstone-vexec {dom} {rb} /tmp/s{k} "
                             f"2> /tmp/tr{k}.txt")
        b.bounds[f"smoke-{k}"] = 30
        t = f"/tmp/tr{k}.txt"
        b.step(f"sum-{k}", f"echo \"TRACE lines $(wc -l < {t}) service $(grep -c CAPSTONE_VM_SERVICE {t})\"; "
                           f"grep CAPSTONE_VM_SERVICE {t} | sed 's/.* nr=\\([0-9]*\\) result=\\(-*[0-9]*\\).*/NR \\1 \\2/' "
                           f"| sort | uniq -c | sort -rn | head -n 25; "
                           f"grep -v CAPSTONE_VM_SERVICE {t} | cut -c1-30 | sort | uniq -c | sort -rn | head -n 12 "
                           f"| sed 's/^/KIND /'; echo TAIL; tail -n 40 {t}")
    steps, transcript, completed = b.execute(a.board_kit, a.out / "work", 3000)
    rows = {n: {"result": r, "text": t} for n, (t, r) in steps.items()}
    (a.out / "steps.json").write_text(json.dumps({"completed": completed, "transcript": str(transcript),
                                                   "steps": rows}, indent=1))
    for n, (t, r) in steps.items():
        print(f"STEP {n} {r}")
    print(f"completed={completed} steps={len(steps)} transcript={transcript}")
    if len(steps) != 2 * a.runs:
        print(f"NO DATA: {len(steps)} of {2 * a.runs} steps in the transcript")
        sys.exit(2)


if __name__ == "__main__":
    main()
