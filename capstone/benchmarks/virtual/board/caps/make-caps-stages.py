#!/usr/bin/env python3
"""Wrap each Caps perf kit as a run-stage.py stage, so the SPEC lane's chain (chain-board.sh -> spec-board.sh ->
run-stage.py run --backend board) can boot it on board 1, bitstream 3.

  make-caps-stages.py [--out $CAPSTONE_WORK/perf/T0a/board/stages]

Per kit K (capsh1..3, capsa1..4) it writes <out>/perf-caps-K/:
  plan.json        -- what spec-board.sh requires and run-stage.py's read() parses: one arm ("perfkit"), one command
                      (the kit's run script), no validations, no outputs. Its provenance keys name the kit and hashes.
  stage/run.sh     -- the wrapper (below). Executable, as spec-board.sh checks.
  stage/kit        -- a SYMLINK to the kit's own stage directory (images, launcher, run-K.sh). boardvm's put() copies a
                      tree with symlinks=False, so the kit's files are staged by content, unchanged and not duplicated.
  stage/SHA256SUMS -- a symlink to the kit's SHA256SUMS.

The wrapper runs the kit's run-K.sh UNCHANGED, so every @@PERF-BEGIN / BENCH-INFO / BENCH / BENCH-DONE / @@PERF-END
line reaches the step's text exactly as in the campaign's pilot boots (tier3-perfpilot.py ran `sh .../run-t3k1.sh` as
its one step). The @@SPEC-* lines around it are only what read() needs to see one command that ended. Before any
process runs, every kit file is hashed ON THE BOARD and compared with SHA256SUMS. On a mismatch the step ends with
exit 3 before the first process. The board-side hashes are printed as the SPEC-HASH block.
Refuses (exit 1) if a kit's SHA256SUMS does not verify on the host.
"""
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path
WORK = __import__('os').environ.get('CAPSTONE_WORK', '/mnt/hdd/' + __import__('os').environ.get('USER', '') + '/capstone')

KITS = Path(WORK + '/perf/T0a/kits')
NAMES = [('heap15', f'capsh{i}') for i in (1, 2, 3)] + [('other88', f'capsa{i}') for i in (1, 2, 3, 4)]

RUN_SH = r'''#!/bin/sh
# Caps perf kit @K@ as a run-stage.py stage (hand-written wrapper; make-caps-stages.py). Runs the kit's run-@K@.sh
# UNCHANGED, so its stdout reaches the step's text exactly as in the campaign. The kit's files are checked on the board
# against its SHA256SUMS before the first process; a mismatch ends the step (exit 3) with nothing run.
S=$(cd "$(dirname "$0")" && pwd)
up() { cut -d' ' -f1 /proc/uptime; }
sysb() { for f in cycles instret entries; do cat /sys/module/capstone_vm/parameters/bench_$f 2>/dev/null || echo 0; done | tr '\n' ' '; }
bench() { dmesg | grep 'capstone_vm: bench' | tail -1 | sed 's/.*capstone_vm: //'; }
nbench() { dmesg | grep -c 'capstone_vm: bench'; }
echo "@@SPEC-STAGE caps-@K@ board arms=perfkit commands=1 validations=0"
cd "$S/kit" || { echo "@@CAPS-NOKIT $S/kit"; exit 3; }
sha256sum $(cut -c67- "$S/SHA256SUMS") > /tmp/caps-sums.txt 2>&1
if [ "$(cat /tmp/caps-sums.txt)" != "$(cat "$S/SHA256SUMS")" ]; then
  echo "@@CAPS-SUMS MISMATCH: the staged kit is not the kit (nothing was run)"
  cat /tmp/caps-sums.txt | sed 's/^/@@CAPS-SUMS-GOT /'
  exit 3
fi
echo "@@CAPS-SUMS OK $(wc -l < "$S/SHA256SUMS") files verified on the board; run-@K@.sh $(sha256sum ./run-@K@.sh | cut -c1-16)"
echo "@@SPEC-ARM-BEGIN perfkit"
echo "@@SPEC-BEGIN 0 run perfkit 0"
s0=$(sysb); u0=$(up)
sh ./run-@K@.sh
rc=$?
u1=$(up); s1=$(sysb)
sys=$(echo $s0 $s1 | awk '{printf "%.0f,%.0f,%.0f", $4-$1, $5-$2, $6-$3}')
echo "@@SPEC-END 0 run perfkit 0 rc=$rc up0=$u0 up1=$u1 sys=$sys nbench=$(nbench) :: $(bench)"
echo "@@SPEC-HASH-BEGIN perfkit"
cat /tmp/caps-sums.txt
echo "@@SPEC-HASH-END perfkit"
echo "@@SPEC-ARM-END perfkit"
rm -f /tmp/caps-sums.txt
echo "@@SPEC-DONE caps-@K@ board"
'''


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=Path(WORK + '/perf/T0a/board/stages'))
    a = ap.parse_args()
    for group, k in NAMES:
        kit = KITS / group / k
        r = subprocess.run(['sha256sum', '-c', '--quiet', str(kit / 'SHA256SUMS')], cwd=kit / 'stage',
                           capture_output=True, text=True)
        if r.returncode:
            sys.exit(f'{kit}: SHA256SUMS does not verify on the host: {r.stdout[-400:]}{r.stderr[-400:]}')
        steps = (kit / 'steps.tsv').read_text().splitlines()[1:]
        d = a.out / f'perf-caps-{k}'
        st = d / 'stage'
        st.mkdir(parents=True, exist_ok=True)
        for link, target in (('kit', kit / 'stage'), ('SHA256SUMS', kit / 'SHA256SUMS')):
            p = st / link
            if p.is_symlink() or p.exists():
                p.unlink()
            p.symlink_to(target)
        (st / 'run.sh').write_text(RUN_SH.replace('@K@', k))
        (st / 'run.sh').chmod(0o755)
        plan = dict(bench=f'caps-{k}', size='board', arms=['perfkit'],
                    commands=[dict(input=None, output=None, error=None, keep=False, argv=['sh', f'run-{k}.sh'])],
                    generation=[], target_validation=[], host_validation=[], outputs=[], inputs={},
                    directed='hand-written run.sh wrapping a perf kit (make-caps-stages.py); data are the kit\'s '
                             '@@PERF/BENCH lines, read by capstone/benchmarks/virtual/parse-board.py',
                    kit=str(kit), kit_sha256sums=sha(kit / 'SHA256SUMS'),
                    run_script=f'run-{k}.sh', run_script_sha256=sha(kit / 'stage' / f'run-{k}.sh'),
                    launcher_sha256=sha(kit / 'stage' / 'capstone-vexec'), processes=len(steps),
                    arms_in_kit=['Caps', 'MapLife', 'ObjLife'])
        (d / 'plan.json').write_text(json.dumps(plan, indent=1) + '\n')
        size = sum(f.stat().st_size for f in (kit / 'stage').rglob('*') if f.is_file())
        print(f'{d} processes={len(steps)} kit_bytes={size} run-{k}.sh={plan["run_script_sha256"][:16]} '
              f'run.sh={sha(st / "run.sh")[:16]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
