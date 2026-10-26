#!/usr/bin/env python3
"""perfmain3-check.py <kit dir> <runner sha256/16> <done tag> [arms, default T1,T2,T3]: a T3 perf kit's pre-launch check (2026-10-27), run before any
board is booked. Checks, and prints one line per failure:
- the stage matches its SHA256SUMS, and SHA256SUMS covers every stage file;
- the run script's sha256/16 is the one given, and it ends in @@PERF-<done tag>-DONE;
- every @@PERF-BEGIN line of the run script equals its steps.tsv row (n, program, arm, proc, K, args), and names an image
  under the arm's directory that exists and is a Capstone image (ELF e_machine 259);
- per program, the images of the given arms (default T1, T2 and T3) are distinct files, and each arm has the same number
  of processes;
- **each program's args equal the args the MAIN RUN gave that program** (perf/main1/k*/steps.tsv and perf/main2/*/steps.tsv).
  A program the main run never ran is a failure, so the check cannot pass by having nothing to compare. This is the check
  that the first T3 kits (2026-10-27, voided) failed: huffbench-libc ran "1 5" where the main run gave the non-re-entrant
  program "0 1".
Exit 0 when every check passes, 1 on any failure, 2 when the kit or the main-run plans cannot be read (no data is an error)."""
import collections
import glob
import hashlib
import re
import subprocess
import sys
from pathlib import Path

PERF = Path("/mnt/hdd") / Path.home().name / "capstone/perf"


def main():
    if len(sys.argv) not in (4, 5):
        print(__doc__); sys.exit(2)
    kit, sha, done = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    arms = set((sys.argv[4] if len(sys.argv) == 5 else "T1,T2,T3").split(","))
    stage = kit / "stage"
    runners = sorted(stage.glob("run-*.sh"))
    if not stage.is_dir() or len(runners) != 1 or not (kit / "steps.tsv").is_file():
        print(f"NO DATA: {kit} has no stage, not exactly one run-*.sh, or no steps.tsv"); sys.exit(2)
    main_args = collections.defaultdict(set)
    plans = glob.glob(str(PERF / "main1/k*/steps.tsv")) + glob.glob(str(PERF / "main2/*/steps.tsv"))
    for p in plans:
        for row in list(open(p))[1:]:
            f = row.rstrip("\n").split("\t")
            main_args[f[1]].add(f[5])
    if not main_args:
        print("NO DATA: no main-run steps.tsv was read"); sys.exit(2)
    bad = []
    chk = subprocess.run(["sha256sum", "-c", "--quiet", "../SHA256SUMS"], cwd=stage, capture_output=True, text=True)
    if chk.returncode:
        bad.append(f"stage does not match SHA256SUMS: {(chk.stdout + chk.stderr)[-200:]}")
    summed = {l.split(None, 1)[1].strip().lstrip("*./") for l in open(kit / "SHA256SUMS") if l.strip()}
    files = {str(p.relative_to(stage)) for p in stage.rglob("*") if p.is_file()}
    if files - summed:
        bad.append(f"stage files outside SHA256SUMS: {sorted(files - summed)[:5]}")
    run = runners[0]
    if hashlib.sha256(run.read_bytes()).hexdigest()[:16] != sha:
        bad.append(f"{run.name} is not {sha}")
    text = run.read_text()
    if not re.search(rf"@@PERF-{re.escape(done)}-DONE\"?\s*$", text.rstrip()):
        bad.append(f"{run.name} does not end in @@PERF-{done}-DONE")
    lines = re.findall(r'@@PERF-BEGIN (\d+) (\S+) (\S+) (\d+) K(\d+)"; \./capstone-vexec \./(\S+) (\d+ \d+) ', text)
    steps = [l.rstrip("\n").split("\t") for l in list(open(kit / "steps.tsv"))[1:]]
    if len(lines) != len(steps) or not lines:
        bad.append(f"{len(lines)} run lines vs {len(steps)} steps")
    img = collections.defaultdict(dict); per_arm = collections.defaultdict(collections.Counter)
    for (n, prog, arm, proc, K, path, args), s in zip(lines, steps):
        if [n, prog, arm, proc, K, args] != s[:6]:
            bad.append(f"run line {n} {prog} {arm} {proc} K{K} '{args}' != steps row {s[:6]}")
        f = stage / path
        if path.split("/")[0] != arm or not f.is_file() or int.from_bytes(f.read_bytes()[18:20], "little") != 259:
            bad.append(f"line {n}: image {path} is missing, outside {arm}/, or not a Capstone image")
        else:
            img[(prog, K)][arm] = hashlib.sha256(f.read_bytes()).hexdigest()
        per_arm[prog][arm] += 1
        if prog not in main_args:
            bad.append(f"line {n}: {prog} never ran in the main run, so its args have no reference")
        elif args not in main_args[prog]:
            bad.append(f"line {n}: {prog} {arm} args '{args}', the main run gave {sorted(main_args[prog])}")
    for (prog, K), d in img.items():
        if set(d) != arms or len(set(d.values())) != len(arms):
            bad.append(f"{prog} K{K}: the {sorted(arms)} images are not {len(arms)} distinct files ({sorted(d)})")
    for prog, c in per_arm.items():
        if len(set(c.values())) != 1 or set(c) != arms:
            bad.append(f"{prog}: processes per arm {dict(c)}")
    for b in bad:
        print(f"FAIL {kit.name}: {b}")
    print(f"CHECK {kit.name} runner {sha}: {len(lines)} processes, {len(per_arm)} programs, "
          f"main-run reference from {len(plans)} plans: {'PASS' if not bad else f'{len(bad)} FAILURE(S)'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
