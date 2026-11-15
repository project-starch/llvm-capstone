#!/usr/bin/env python3
"""One boot of the Perl reuse-distance measurement on 500.perlbench_r's train script suns.pl.

    run-perlbench-suns-reuse.py --arm stock|protected --rep N --image PERL_DOM --spec SPEC17 \
        --kit KIT --work NEW_DIR [--step-timeout S]

SPEC17 is a SPEC CPU2017 installation. Stages, under spec/, what SPEC's run directory for suns
holds: data/train/input/suns.pl and its dictionary WORDS, and data/all/input/lib (the modules SPEC
ships); and SPEC's expected files data/train/output/suns.out and validate under expected/. The guest
copies spec/ to /tmp/run, since suns.pl writes its validate file into the working directory, and
runs the image the way SPEC's object.pm invokes perlbench for a script without an .in file,
`perl -I. -I./lib suns.pl`, with PERL_HASH_SEED=0 (perlbench fixes its hash seed at build time). The
standard output goes to suns.out; both files are compared with SPEC's expected ones in the guest.
NEW_DIR then holds trap_platform.py's results.json, boot-1/ and stage-1/ and rep.json, the record
collect-reuse.py reads (the REUSE-GAP line parsed with run-cpython-reuse.py's parse_reuse). Exits
nonzero when the run is to be rejected: incomplete boot, nonzero exit, fault, observer error, or an
output file that differs from SPEC's.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
TRAP_PLATFORM = REPO / 'capstone/runtime/virtual/trap_platform.py'
BENCH = 'benchspec/CPU/500.perlbench_r/data'
STEP = ("rm -rf /tmp/run; mkdir -p /tmp/run; cp -a /mnt/vm/spec/. /tmp/run/; cd /tmp/run; t0=$(date +%s); "
        "PERL_HASH_SEED=0 /mnt/vm/capstone-vexec /mnt/vm/perl-{arm}.dom -I. -I./lib suns.pl > suns.out; rc=$?; "
        "echo \"PERL_TIME:$(( $(date +%s) - t0 ))\"; "
        "for f in suns.out validate; do cmp $f /mnt/vm/expected/$f && echo \"SPEC_MATCH:$f\"; done; exit $rc")


def load_parse_reuse():
    spec = importlib.util.spec_from_file_location('cpython_reuse', HERE.parent / 'cpython/run-cpython-reuse.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.parse_reuse


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def tree_sha(root):
    h = hashlib.sha256()
    for f in sorted(Path(root).rglob('*')):
        if f.is_file():
            h.update(str(f.relative_to(root)).encode() + b'\0' + bytes.fromhex(sha(f)))
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--arm', choices=('stock', 'protected'), required=True)
    p.add_argument('--rep', type=int, required=True)
    for name in ('image', 'spec', 'kit', 'work'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--step-timeout', type=int, default=3600)
    a = p.parse_args()

    data = a.spec / BENCH
    name = f'perl-{a.arm}'
    command = STEP.format(arm=a.arm)
    files = {'spec/suns.pl': data / 'train/input/suns.pl', 'spec/WORDS': data / 'train/input/WORDS',
             'spec/lib': data / 'all/input/lib', 'expected/suns.out': data / 'train/output/suns.out',
             'expected/validate': data / 'train/output/validate', f'perl-{a.arm}.dom': a.image}
    argv = [sys.executable, str(TRAP_PLATFORM), 'run', '--kit', str(a.kit), '--work', str(a.work),
            '--step-timeout', str(a.step_timeout)]
    for dest, src in files.items():
        argv += ['--file', f'{dest}={src}']
    argv += ['--step', f'{name}={command}']
    t0 = time.time()
    run = subprocess.run(argv)
    wall = time.time() - t0

    results = json.loads((a.work / 'results.json').read_text())
    r = next(x for x in results if x['name'] == name)
    out = r['output'].replace('\r', '')
    reuse_gap = load_parse_reuse()(out)
    matches = re.findall(r'^SPEC_MATCH:(\S+)$', out, re.M)
    stage = a.work / 'stage-1'
    rec = {
        'program': 'perl', 'arm': a.arm, 'rep': a.rep,
        'completed': run.returncode == 0 and r['kind'] == 'exit',
        'rc': r['value'] if r['kind'] == 'exit' else None,
        'kind': r['kind'], 'fault': r['fault'], 'kernel_faults': r['kernel_faults'],
        'boot_wall_seconds': round(wall, 1),
        'guest_seconds': int(m.group(1)) if (m := re.search(r'^PERL_TIME:(\d+)$', out, re.M)) else None,
        'reuse_gap': reuse_gap,
        'spec_match': matches, 'outputs_equal_spec': sorted(matches) == ['suns.out', 'validate'],
        'step_command': command,
        'staged_sha256': {'perl.dom': sha(stage / f'perl-{a.arm}.dom'),
                          'capstone-vexec': sha(stage / 'capstone-vexec'), 'gate.sh': sha(stage / 'gate.sh'),
                          'suns.pl': sha(stage / 'spec/suns.pl'), 'WORDS': sha(stage / 'spec/WORDS'),
                          'lib_tree': tree_sha(stage / 'spec/lib'),
                          'expected/suns.out': sha(stage / 'expected/suns.out'),
                          'expected/validate': sha(stage / 'expected/validate')},
        'spec_version': (a.spec / 'version.txt').read_text().strip(),
        'platform_sha256': {n: sha(a.kit / 'images' / n) for n in ('Image', 'fw_jump.elf', 'initramfs.cpio.gz')}
                           | {'qemu-system-riscv64': sha(a.kit / 'qemu-system-riscv64'),
                              'capstone-vexec': sha(a.kit / 'adapter/capstone-vexec')},
        'run_staged_sha256': sha(TRAP_PLATFORM),
    }
    (a.work / 'rep.json').write_text(json.dumps(rec, indent=1, sort_keys=True) + '\n')
    rg = reuse_gap.get('sv-heads')
    print(f'perl {a.arm} rep{a.rep}: kind={r["kind"]} rc={rec["rc"]} spec_match={matches} '
          f'error={rg and rg["error"]} issues={rg and rg["issues"]} distinct={rg and rg["distinct_starts"]} '
          f'capacity={rg and rg["capacity"]} guest={rec["guest_seconds"]}s boot={wall:.0f}s')
    ok = (rec['completed'] and rec['rc'] == 0 and not r['fault'] and not r['kernel_faults'] and rg
          and rg['error'] == 0 and rec['outputs_equal_spec'])
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
