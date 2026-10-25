#!/usr/bin/env python3
"""trace-pair.py --board-kit <kit> --tools <tools dir> --smoke <smoke.rb> --arm <label>=<mruby .dom> ... --out <fresh dir>

mruby's stock smoke on two or more images in one cold board boot, each run with the launcher's service trace on
(CAPSTONE_VM_TRACE=1), in the order given. It is the matched pair for the SIGALRM-storm reading (2026-10-21): the r60
image f8594e11 (MRB_TICK_UNIT 4, a 4 ms ITIMER_REAL tick) against 1ab24a92 (MRB_TICK_UNIT 100). Of 124 objects only
task_hal.o and task.o differ, and the sources are identical. Each arm is given as label=path, and a label may repeat.
For each run k:
- run-<k>-<label>: the smoke, bounded at 600 s (30 s x BOARD_STEP_MULT 20), stdout the step's text, stderr (grants,
  the trace) to /tmp/tr<k>.txt. A bound fires the stall capture (boardvm STALL_FN) once per boot.
- sum-<k>: the trace's line count; its last 30 lines FIRST (cheap); then over its LAST 30,000 lines only (a stalled
  600 s trace runs to ~200,000 lines, and a full busybox sort of it took 30 minutes on 2026-10-21), the (nr, result)
  histogram and the line kinds; then the whole-trace counts of SIGDONE (3235774470) and rt_sigprocmask (135), by
  grep -c, which is linear.
Judges nothing; prints the step table and writes <out>/steps.json. Exits 2 unless every step is in the transcript."""
import argparse
import json
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--board-kit", type=Path, required=True)
    p.add_argument("--tools", type=Path, required=True, help="bug-corpora/tools of a clean worktree")
    p.add_argument("--smoke", type=Path, required=True)
    p.add_argument("--arm", action="append", required=True, help="label=path of a mruby .dom (repeatable, in order)")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.tools))
    import boardvm  # noqa: E402
    if a.out.exists():
        sys.exit(f"{a.out} exists; a run goes into a fresh directory")
    arms = [x.split("=", 1) for x in a.arm]
    if any(len(x) != 2 or not Path(x[1]).is_file() for x in arms):
        sys.exit(f"--arm must be label=existing-file: {a.arm}")
    a.out.mkdir(parents=True)
    b = boardvm.BoardBatch(a.out / "stage")
    rb = b.put("smoke.rb", a.smoke)
    doms = {}
    for label, path in arms:
        if label not in doms:
            doms[label] = b.put(f"images/{label}.dom", Path(path))
    b.bounds = {}
    for k, (label, _) in enumerate(arms, 1):
        name = f"run-{k}-{label}"
        b.step(name, f"rm -rf /tmp/s{k} && mkdir /tmp/s{k} && cd /tmp/s{k} && "
                     f"CAPSTONE_VM_TRACE=1 /test-domains/{boardvm.BATCH}/capstone-vexec {doms[label]} {rb} /tmp/s{k} "
                     f"2> /tmp/tr{k}.txt")
        b.bounds[name] = 30
        t = f"/tmp/tr{k}.txt"
        b.step(f"sum-{k}", f"echo \"TRACE lines $(wc -l < {t}) service $(grep -c CAPSTONE_VM_SERVICE {t}) "
                           f"sigdone $(grep -c 'nr=3235774470 ' {t}) sigprocmask $(grep -c 'nr=135 ' {t})\"; "
                           f"echo TAIL; tail -n 30 {t}; echo LAST30000; tail -n 30000 {t} | grep CAPSTONE_VM_SERVICE "
                           f"| sed 's/.* nr=\\([0-9]*\\) result=\\(-*[0-9]*\\).*/NR \\1 \\2/' | sort | uniq -c | sort -rn "
                           f"| head -n 15; tail -n 30000 {t} | grep -v CAPSTONE_VM_SERVICE | cut -c1-30 | sort | uniq -c "
                           f"| sort -rn | head -n 8 | sed 's/^/KIND /'")
    steps, transcript, completed = b.execute(a.board_kit, a.out / "work", 3000)
    rows = {n: {"result": r, "text": t} for n, (t, r) in steps.items()}
    (a.out / "steps.json").write_text(json.dumps({"completed": completed, "transcript": str(transcript),
                                                   "arms": arms, "steps": rows}, indent=1))
    for n, (t, r) in steps.items():
        print(f"STEP {n} {r} {'SMOKE_DONE' if 'SMOKE_DONE' in t else ''}")
    print(f"completed={completed} steps={len(steps)} transcript={transcript}")
    if len(steps) != 2 * len(arms):
        print(f"NO DATA: {len(steps)} of {2 * len(arms)} steps in the transcript")
        sys.exit(2)


if __name__ == "__main__":
    main()
