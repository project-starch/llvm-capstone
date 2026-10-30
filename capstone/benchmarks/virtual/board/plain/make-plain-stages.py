#!/usr/bin/env python3
"""Wrap each Plain perf kit as a run-stage.py stage, so the SPEC lane's chain (chain-board.sh -> spec-board.sh ->
run-stage.py run --backend board) can boot it on board 1, bitstream 3. The Caps lane's make-caps-stages.py, adapted.

  make-plain-stages.py [--out $CAPSTONE_WORK/perf/T0b/board/stages]

Per kit K (the pilot pp1, then p1..p6) it writes <out>/perf-plain-K/:
  plan.json        -- what spec-board.sh requires and run-stage.py's read() parses: one arm ("perfkit"), one command
                      (the kit's run script), no validations, no outputs; its provenance keys name the kit and hashes.
  stage/run.sh     -- the wrapper (below), executable.
  stage/kit        -- a SYMLINK to the kit's stage directory (boardvm's put() copies with symlinks=False, so the kit's
                      files are staged by content, unchanged).
  stage/SHA256SUMS -- a symlink to the kit's SHA256SUMS (paths relative to the kit's stage, "./...").

THE PILOT GATE, part 1 (the stage side). The main kits p1..p6 are written DISARMED: their plan file is
`plan.json.disarmed`, and spec-board.sh refuses a stage without plan.json ("SB STOP: ... is not a make-stage.py
output"), which chain-board.sh turns into CB STOP. So a chain that reaches p1 before the pilot has been read VALID by
check-plain-boot.py --pilot stops there, whatever list it was given. plain-chain.sh arms them (`--arm`) only after the
gate passes. pp1 is always armed.

The wrapper runs the kit's run-K.sh UNCHANGED (the plain-tick-probe first, then every @@PERF step). Before any process
runs, every kit file is hashed ON THE BOARD and compared with SHA256SUMS; a mismatch ends the step with exit 3 before
the first process. The board-side hashes are printed as the SPEC-HASH block. Refuses (exit 1) if a kit's SHA256SUMS
does not verify on the host.

  make-plain-stages.py --arm [--out ...]   renames plan.json.disarmed -> plan.json in p1..p6 (plain-chain.sh does this)
"""
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path

WORK = Path(os.environ.get('CAPSTONE_WORK', '/mnt/hdd/' + os.environ.get('USER', '') + '/capstone'))
T0B = WORK / 'perf/T0b'
KITS = [('pp1', T0B / 'kits-pilot/pp1')] + [(f'p{i}', T0B / f'kits/p{i}') for i in range(1, 7)]
PILOT = 'pp1'

RUN_SH = r'''#!/bin/sh
# Plain perf kit @K@ as a run-stage.py stage (make-plain-stages.py). Runs the kit's run-@K@.sh UNCHANGED, so its stdout
# reaches the step's text exactly as the kit prints it. The kit's files are checked on the board against its SHA256SUMS
# before the first process; a mismatch ends the step (exit 3) with nothing run.
S=$(cd "$(dirname "$0")" && pwd)
up() { cut -d' ' -f1 /proc/uptime; }
sysb() { for f in cycles instret entries; do cat /sys/module/capstone_vm/parameters/bench_$f 2>/dev/null || echo 0; done | tr '\n' ' '; }
bench() { dmesg | grep 'capstone_vm: bench' | tail -1 | sed 's/.*capstone_vm: //'; }
nbench() { dmesg | grep -c 'capstone_vm: bench'; }
echo "@@SPEC-STAGE plain-@K@ board arms=perfkit commands=1 validations=0"
cd "$S/kit" || { echo "@@PLAIN-NOKIT $S/kit"; exit 3; }
sha256sum $(cut -c67- "$S/SHA256SUMS") > /tmp/plain-sums.txt 2>&1
if [ "$(cat /tmp/plain-sums.txt)" != "$(cat "$S/SHA256SUMS")" ]; then
  echo "@@PLAIN-SUMS MISMATCH: the staged kit is not the kit (nothing was run)"
  sed 's/^/@@PLAIN-SUMS-GOT /' /tmp/plain-sums.txt
  exit 3
fi
echo "@@PLAIN-SUMS OK $(wc -l < "$S/SHA256SUMS") files verified on the board; run-@K@.sh $(sha256sum ./run-@K@.sh | cut -c1-16)"
echo "@@SPEC-ARM-BEGIN perfkit"
echo "@@SPEC-BEGIN 0 run perfkit 0"
s0=$(sysb); u0=$(up)
sh ./run-@K@.sh
rc=$?
u1=$(up); s1=$(sysb)
sys=$(echo $s0 $s1 | awk '{printf "%.0f,%.0f,%.0f", $4-$1, $5-$2, $6-$3}')
echo "@@SPEC-END 0 run perfkit 0 rc=$rc up0=$u0 up1=$u1 sys=$sys nbench=$(nbench) :: $(bench)"
echo "@@SPEC-HASH-BEGIN perfkit"
cat /tmp/plain-sums.txt
echo "@@SPEC-HASH-END perfkit"
echo "@@SPEC-ARM-END perfkit"
rm -f /tmp/plain-sums.txt
echo "@@SPEC-DONE plain-@K@ board"
'''


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=T0B / 'board/stages')
    ap.add_argument('--arm', action='store_true', help='arm p1..p6 (after the pilot gate passed)')
    a = ap.parse_args()
    if a.arm:
        n = 0
        for k, _ in KITS:
            d = a.out / f'perf-plain-{k}'
            if k != PILOT and (d / 'plan.json.disarmed').exists():
                (d / 'plan.json.disarmed').rename(d / 'plan.json')
                n += 1
        print(f'armed {n} main-kit stages under {a.out}')
        return 0
    for k, kit in KITS:
        r = subprocess.run(['sha256sum', '-c', '--quiet', str(kit / 'SHA256SUMS')], cwd=kit / 'stage',
                           capture_output=True, text=True)
        if r.returncode:
            sys.exit(f'{kit}: SHA256SUMS does not verify on the host: {r.stdout[-400:]}{r.stderr[-400:]}')
        steps = (kit / 'steps.tsv').read_text().splitlines()[1:]
        d = a.out / f'perf-plain-{k}'
        st = d / 'stage'
        st.mkdir(parents=True, exist_ok=True)
        for link, target in (('kit', kit / 'stage'), ('SHA256SUMS', kit / 'SHA256SUMS')):
            p = st / link
            if p.is_symlink() or p.exists():
                p.unlink()
            p.symlink_to(target)
        (st / 'run.sh').write_text(RUN_SH.replace('@K@', k))
        (st / 'run.sh').chmod(0o755)
        plan = dict(bench=f'plain-{k}', size='board', arms=['perfkit'],
                    commands=[dict(input=None, output=None, error=None, keep=False, argv=['sh', f'run-{k}.sh'])],
                    generation=[], target_validation=[], host_validation=[], outputs=[], inputs={},
                    directed='hand-written run.sh wrapping a perf kit (make-plain-stages.py); data are the kit\'s '
                             '@@PERF/BENCH/BENCH-KERN and PLAINPROBE lines, read by parse-board.py and check-plain-boot.py',
                    kit=str(kit), kit_sha256sums=sha(kit / 'SHA256SUMS'),
                    run_script=f'run-{k}.sh', run_script_sha256=sha(kit / 'stage' / f'run-{k}.sh'),
                    launcher_sha256=sha(kit / 'stage' / 'capstone-vexec'),
                    probe_sha256=sha(kit / 'stage' / 'plain-tick-probe'), processes=len(steps),
                    arms_in_kit=['Plain', 'T1 (MapLife)', 'T3 (ObjLife)'], pilot=k == PILOT)
        for f in ('plan.json', 'plan.json.disarmed'):
            if (d / f).exists():
                (d / f).unlink()
        name = 'plan.json' if k == PILOT else 'plan.json.disarmed'
        (d / name).write_text(json.dumps(plan, indent=1) + '\n')
        size = sum(f.stat().st_size for f in (kit / 'stage').rglob('*') if f.is_file())
        print(f'{d} {name} processes={len(steps)} kit_bytes={size} run-{k}.sh={plan["run_script_sha256"][:16]} '
              f'SHA256SUMS={plan["kit_sha256sums"][:16]} run.sh={sha(st / "run.sh")[:16]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
