#!/usr/bin/env python3
"""Wrap each ObjBounds perf kit as a run-stage.py stage, so the SPEC lane's chain (chain-board.sh -> spec-board.sh ->
run-stage.py run --backend board) can boot it on board 2, on the ObjBounds bitstream (capstone-ariane objbounds-b
603528b8b: bitstream 3 without the per-access lifetime check).

  make-objb-stages.py [--out $CAPSTONE_WORK/perf/objb/board/stages] [--arm]

Per kit K (objbh1..3, objba1..4) it writes <out>/perf-objb-K/:
  plan.json (objbh1, the pilot) or plan.json.disarmed (every other kit) -- what spec-board.sh requires and
                      run-stage.py's read() parses: one arm ("perfkit"), one command (the kit's run script).
                      spec-board.sh refuses a stage without plan.json, so the main kits cannot boot before the pilot
                      gate (objb-first.sh) passes. --arm renames plan.json.disarmed to plan.json in every main kit.
  stage/run.sh     -- the wrapper (below). Executable, as spec-board.sh checks.
  stage/kit        -- a SYMLINK to the kit's stage directory (images, launcher, run-K.sh); boardvm's put() copies by
                      content.
  stage/SHA256SUMS -- a symlink to the kit's SHA256SUMS.

The wrapper runs the kit's run-K.sh UNCHANGED, so its @@PERF/BENCH lines reach the step text as in the campaign.
Before any process runs, every kit file is hashed ON THE BOARD and compared with SHA256SUMS; on a mismatch the step
ends with exit 3 before the first process. Refuses (exit 1) if a kit's SHA256SUMS does not verify on the host.
"""
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path

W = Path(os.environ.get('CAPSTONE_WORK', f'/mnt/hdd/{os.environ.get("USER", "")}/capstone'))
KITS = W / 'perf/objb/kits'
PILOT = 'objbh1'
NAMES = [('heap15', f'objbh{i}') for i in (1, 2, 3)] + [('other88', f'objba{i}') for i in (1, 2, 3, 4)]

RUN_SH = r'''#!/bin/sh
# ObjBounds perf kit @K@ as a run-stage.py stage (make-objb-stages.py). Runs the kit's run-@K@.sh UNCHANGED. The kit's
# files are checked on the board against its SHA256SUMS before the first process; a mismatch ends the step (exit 3).
S=$(cd "$(dirname "$0")" && pwd)
up() { cut -d' ' -f1 /proc/uptime; }
sysb() { for f in cycles instret entries; do cat /sys/module/capstone_vm/parameters/bench_$f 2>/dev/null || echo 0; done | tr '\n' ' '; }
bench() { dmesg | grep 'capstone_vm: bench' | tail -1 | sed 's/.*capstone_vm: //'; }
nbench() { dmesg | grep -c 'capstone_vm: bench'; }
echo "@@SPEC-STAGE objb-@K@ board arms=perfkit commands=1 validations=0"
cd "$S/kit" || { echo "@@OBJB-NOKIT $S/kit"; exit 3; }
sha256sum $(cut -c67- "$S/SHA256SUMS") > /tmp/objb-sums.txt 2>&1
if [ "$(cat /tmp/objb-sums.txt)" != "$(cat "$S/SHA256SUMS")" ]; then
  echo "@@OBJB-SUMS MISMATCH: the staged kit is not the kit (nothing was run)"
  sed 's/^/@@OBJB-SUMS-GOT /' /tmp/objb-sums.txt
  exit 3
fi
echo "@@OBJB-SUMS OK $(wc -l < "$S/SHA256SUMS") files verified on the board; run-@K@.sh $(sha256sum ./run-@K@.sh | cut -c1-16)"
echo "@@SPEC-ARM-BEGIN perfkit"
echo "@@SPEC-BEGIN 0 run perfkit 0"
s0=$(sysb); u0=$(up)
sh ./run-@K@.sh
rc=$?
u1=$(up); s1=$(sysb)
sys=$(echo $s0 $s1 | awk '{printf "%.0f,%.0f,%.0f", $4-$1, $5-$2, $6-$3}')
echo "@@SPEC-END 0 run perfkit 0 rc=$rc up0=$u0 up1=$u1 sys=$sys nbench=$(nbench) :: $(bench)"
echo "@@SPEC-HASH-BEGIN perfkit"
cat /tmp/objb-sums.txt
echo "@@SPEC-HASH-END perfkit"
echo "@@SPEC-ARM-END perfkit"
rm -f /tmp/objb-sums.txt
echo "@@SPEC-DONE objb-@K@ board"
'''


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=W / 'perf/objb/board/stages')
    ap.add_argument('--arm', action='store_true', help='arm the main kits (after the pilot gate passed)')
    a = ap.parse_args()
    if a.arm:
        n = 0
        for _, k in NAMES:
            if k == PILOT:
                continue
            d = a.out / f'perf-objb-{k}'
            if (d / 'plan.json.disarmed').exists():
                (d / 'plan.json.disarmed').rename(d / 'plan.json'); n += 1
            elif not (d / 'plan.json').exists():
                sys.exit(f'{d}: neither plan.json nor plan.json.disarmed')
        print(f'armed {n} main-kit stages under {a.out}')
        return 0
    for group, k in NAMES:
        kit = KITS / group / k
        r = subprocess.run(['sha256sum', '-c', '--quiet', str(kit / 'SHA256SUMS')], cwd=kit / 'stage',
                           capture_output=True, text=True)
        if r.returncode:
            sys.exit(f'{kit}: SHA256SUMS does not verify on the host: {r.stdout[-400:]}{r.stderr[-400:]}')
        steps = (kit / 'steps.tsv').read_text().splitlines()[1:]
        d = a.out / f'perf-objb-{k}'
        st = d / 'stage'
        st.mkdir(parents=True, exist_ok=True)
        for link, target in (('kit', kit / 'stage'), ('SHA256SUMS', kit / 'SHA256SUMS')):
            p = st / link
            if p.is_symlink() or p.exists():
                p.unlink()
            p.symlink_to(target)
        (st / 'run.sh').write_text(RUN_SH.replace('@K@', k))
        (st / 'run.sh').chmod(0o755)
        plan = dict(bench=f'objb-{k}', size='board', arms=['perfkit'],
                    commands=[dict(input=None, output=None, error=None, keep=False, argv=['sh', f'run-{k}.sh'])],
                    generation=[], target_validation=[], host_validation=[], outputs=[], inputs={},
                    directed='hand-written run.sh wrapping a perf kit (make-objb-stages.py); data are the kit\'s '
                             '@@PERF/BENCH lines, read by capstone/benchmarks/virtual/parse-board.py',
                    kit=str(kit), kit_sha256sums=sha(kit / 'SHA256SUMS'),
                    run_script=f'run-{k}.sh', run_script_sha256=sha(kit / 'stage' / f'run-{k}.sh'),
                    launcher_sha256=sha(kit / 'stage' / 'capstone-vexec'), processes=len(steps),
                    arms_in_kit=['T1 (MapLife images: ObjBounds on this bitstream)', 'T3 (ObjLife images)'])
        for f in ('plan.json', 'plan.json.disarmed'):
            if (d / f).exists():
                (d / f).unlink()
        name = 'plan.json' if k == PILOT else 'plan.json.disarmed'
        (d / name).write_text(json.dumps(plan, indent=1) + '\n')
        size = sum(f.stat().st_size for f in (kit / 'stage').rglob('*') if f.is_file())
        print(f'{d} {name} processes={len(steps)} kit_bytes={size} run-{k}.sh={plan["run_script_sha256"][:16]} '
              f'run.sh={sha(st / "run.sh")[:16]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
