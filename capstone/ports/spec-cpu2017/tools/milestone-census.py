#!/usr/bin/env python3
"""Build an immutable milestone platform and measure matched SPEC arms."""
import argparse
import concurrent.futures
import copy
import hashlib
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
BENCHES = ['502.gcc_r', '538.imagick_r', '519.lbm_r', '544.nab_r',
           '500.perlbench_r', '505.mcf_r', '557.xz_r']
# Prefixes, not estimates of whole-program performance. See profile.md.
SLICES = dict(zip(BENCHES, ['final_start_function:1', 'GetVirtualPixelsFromNexus:32',
                          'LBM_performStreamCollideTRT:3', 'mme34:5',
                          'Perl_pp_nextstate:10000', 'primal_bea_mpp:3551',
                          'sha_compress:13392']))
COMPILER = ['llvm', 'clang', 'lld', 'cmake', 'third-party']
SDK = ['compiler-rt', 'capstone/ports/common', 'capstone/ports/musl-capstone',
       'capstone/sublet', 'capstone/runtime/CMakeLists.txt',
       'capstone/runtime/application', 'capstone/runtime/cmake',
       'capstone/runtime/common', 'capstone/runtime/domain',
       'capstone/runtime/include', 'capstone/runtime/host',
       'capstone/tests/capstone-test-env.sh']
VIRTUAL = 'capstone/runtime/virtual'
# Application.cmake selects these implementation files. Its own hash above
# invalidates reuse if that source list changes; all local headers are tracked.
VIRTUAL_SDK = {'start.S', 'thread.c', 'pthread.c', 'fenv.c', 'mapping.c',
               'heap.c', 'link.ld', 'build-sdk.sh'}
RECIPE = 'release-assertions-capstone-clang-lld-sdk-virtual-default-v1'


def run(args, **kwargs):
    return subprocess.run([str(x) for x in args], check=True, **kwargs)


def capture(args):
    return run(args, stdout=subprocess.PIPE, text=True).stdout.strip()


def envpath(name, default):
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


def dump(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def sdk_identity(cap):
    config = json.loads((cap / 'sdk/sdk.json').read_text())
    if (config.get('profile'), config.get('heap_lifetimes'),
        config.get('heap_object_bounds'), config.get('heap_scrub_free')) != ('virtual', True, True, False):
        raise ValueError('SDK does not use the default virtual protection policy')
    paths = [cap / 'sdk/sdk.json', cap / 'sdk/capstone-cc']
    paths += [Path(config[k]) for k in ('cc', 'ld', 'libc', 'runtime', 'builtins', 'script')]
    paths += sorted(p for p in (cap / 'build/lib/clang').rglob('*') if p.is_file())
    return {str(p): sha(p) for p in paths}


def disk(root, compiler=False):
    free = min(shutil.disk_usage(root).free, shutil.disk_usage(Path.home()).free)
    minimum = 20 if compiler else 15
    print(f'disk free {free} bytes; minimum {minimum * 1024**3}', flush=True)
    if free < minimum * 1024**3:
        raise ValueError(f'UNRESOLVED: less than {minimum} GiB free; no copy/build/run started')


def git(repo, *args):
    return capture(['git', '-C', repo, *args])


def identities(repo, commit):
    result = {p: git(repo, 'rev-parse', f'{commit}:{p}') for p in COMPILER + SDK}
    # Launcher, adapter and standalone gate changes do not rebuild the SDK.
    # Hash entry names/modes/object IDs, including any newly added headers.
    entries = git(repo, 'ls-tree', commit, VIRTUAL + '/').splitlines()
    entries = [s for s in entries if s.split()[1] == 'blob' and
               (Path(s.split('\t', 1)[1]).name in VIRTUAL_SDK or s.endswith('.h'))]
    result[VIRTUAL + '/SDK sources and headers'] = hashlib.sha256(
        '\n'.join(entries).encode()).hexdigest()
    result['recipe'] = RECIPE
    return result


def export(repo, commit, paths, dest):
    dest.mkdir(parents=True)
    producer = subprocess.Popen(['git', '-C', str(repo), 'archive', commit, *paths],
                                stdout=subprocess.PIPE)
    try:
        run(['tar', '-x', '-C', dest], stdin=producer.stdout)
    finally:
        producer.stdout.close()
        rc = producer.wait()
    if rc:
        raise ValueError('git archive failed')
    (dest / 'llvm').mkdir(exist_ok=True)


def shell(script, log, env):
    with log.open('w') as stream:
        run(['bash', '-c', 'set -euo pipefail\n' + script], env=env,
            stdout=stream, stderr=subprocess.STDOUT)


def verify_platform(platform):
    manifest = json.loads((platform / 'platform.json').read_text())
    for name, digest in manifest['files'].items():
        if sha(platform / name) != digest:
            raise ValueError(f'platform hash mismatch: {name}')
    return manifest


def copy_spec(source, dest):
    def ignore(directory, names):
        parent = Path(directory)
        if parent == source:
            return [n for n in names if n in ('result', 'tmp')]
        if parent.parent == source / 'benchspec/CPU':
            return [n for n in names if n in ('run', 'build')]
        return []
    shutil.copytree(source, dest, symlinks=False, ignore=ignore)


def check_plain_config(kit, specroot):
    installed = (specroot / 'config/riscv-plain.cfg').read_text()
    body, marker, hashes = installed.partition('\n__HASH__')
    if not marker or not hashes.strip():
        raise ValueError('plain SPEC config is missing executable verification hashes')
    if body.strip() != (kit / 'config/riscv-plain.cfg').read_text().strip():
        raise ValueError('plain config changed; supply a template built with that config')


def benchmark_arms(arms, bench):
    settings = copy.deepcopy(arms)
    if bench == '500.perlbench_r':
        # The guest boots the module in both arms; keep its slot capacity and
        # the CPU capacity consistent. Nested perl domains require two slots.
        settings['platform']['cpu_props'] = [p for p in settings['platform'].get('cpu_props', [])
            if not p.startswith('x-capstone-supervisor-slots=')] + ['x-capstone-supervisor-slots=2']
        for arm in settings['arms'].values():
            # Perl's file-owner tests need guest ownership, and its children
            # execute a relative path to the image staged on local tmpfs.
            arm.update(share_model='mapped-xattr', stage_image='tmpfs')
            arm['cpu_props'] = [p for p in arm.get('cpu_props', [])
                if not p.startswith('x-capstone-supervisor-slots=')]
            params = arm.get('module_params', '').split()
            arm['module_params'] = ' '.join([p for p in params
                if not p.startswith('supervisor_slots=')] + ['supervisor_slots=2'])
    return settings


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('commit')
    ap.add_argument('tag')
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', args.tag):
        raise ValueError('tag must contain only letters, digits, underscores and hyphens')
    base = envpath('CENSUS_BASE', '~/prof-dev')
    root = envpath('CENSUS_ROOT', base)
    repo = envpath('CENSUS_REPO', '~/llvm-capstone')
    disk(root)
    lock = (root / '.milestone-census.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise ValueError('another milestone census owns this output root')
    out = root / 'runs' / args.tag
    if out.exists():
        raise ValueError(f'refusing existing run directory: {out}')
    commit = git(repo, 'rev-parse', '--verify', args.commit + '^{commit}')
    hashes = identities(repo, commit)
    key = hashlib.sha256((json.dumps(hashes, sort_keys=True) +
                          git(repo, 'rev-parse', f'{commit}:capstone/ports/spec-cpu2017')).encode()).hexdigest()[:20]
    cache = root / 'census-cache' / key
    template = envpath('CENSUS_ARMS', base / 'arms-census4.json')
    arms = json.loads(template.read_text())
    source_platform = Path(arms['platform']['dir']).expanduser()
    platform_manifest = verify_platform(source_platform)
    pin = git(repo, 'rev-parse', f'{commit}:capstone/capstone-qemu')
    provenance = platform_manifest['provenance'].get('qemu-system-riscv64', '')
    if pin[:12] not in provenance:
        raise ValueError(f'UNRESOLVED: platform provenance does not attest QEMU pin {pin}; supply CENSUS_ARMS with a matching platform')
    kernel = envpath('KERNEL_BUILD', '~/llvm-capstone/capstone/caplifive-buildroot/build-qemu/build/linux-custom')
    cross = os.path.expanduser(os.environ.get('CROSS_COMPILE', '~/llvm-capstone/capstone/caplifive-buildroot/build-qemu/host/bin/riscv64-buildroot-linux-gnu-'))
    if not (kernel / 'Module.symvers').is_file() or not Path(cross + 'gcc').is_file():
        raise ValueError('UNRESOLVED: set KERNEL_BUILD and CROSS_COMPILE to a prepared matching kernel and compiler')
    cache_ready = (cache / 'ready.json').exists()
    candidates = [] if cache_ready else sorted(base.glob('cap[0-9]*'), reverse=True)
    reuse = None
    for candidate in candidates:
        record = candidate / 'toolchain-record.txt'
        if not record.exists() or 'CAP_DONE' not in record.read_text():
            continue
        match = re.search(r'llvm-capstone commit ([0-9a-f]{40})', record.read_text())
        if not match:
            continue
        old = match.group(1)
        # Explicit shallow fetch: probing a missing commit with cat-file in a
        # partial clone can implicitly fetch its entire ancestry.
        run(['git', '-C', repo, '-c', 'fetch.writeCommitGraph=false',
             '-c', 'fetch.recurseSubmodules=false', 'fetch', '--depth=1',
             '--filter=blob:none', 'origin', old])
        if identities(repo, old) == hashes:
            for field, path in [('clang', candidate / 'build/bin/clang'),
                                ('libc', candidate / 'musl/libc-capstone.a')]:
                expected = re.search(r'^' + field + r' ([0-9a-f]{64})', record.read_text(), re.M)
                if not expected or sha(path) != expected.group(1):
                    raise ValueError(f'{candidate}: {field} differs from toolchain record')
            sdk_identity(candidate)
            reuse = candidate
            break
    if cache.exists() and not cache_ready:
        raise ValueError(f'UNRESOLVED: incomplete cache {cache}; inspect it before retrying')
    out.mkdir(parents=True)
    started = time.monotonic()
    dump(out / 'source-hashes.json', hashes)
    print('compiler/SDK input hashes:', json.dumps(hashes, sort_keys=True), flush=True)
    if not cache_ready:
        disk(root)
        cache.mkdir(parents=True)
        src = cache / 'src'
        paths = ['capstone/runtime', 'capstone/ports/common', 'capstone/ports/musl-capstone',
                 'capstone/ports/spec-cpu2017', 'capstone/tests', 'capstone/sublet', 'compiler-rt']
        if reuse is None:
            paths += COMPILER
        export(repo, commit, paths, src)
        env = dict(os.environ, T=str(src), W=str(cache / 'cap'),
                   CAPSTONE_REPO_ROOT=str(src), CAPSTONE_TMP_ROOT=str(cache / 'tmp'),
                   CAPSTONE_QEMU_LOCK=str(cache / 'qemu.lock'))
        if reuse:
            cap = reuse
            print(f'reuse compiler+SDK {cap}: all listed source hashes equal', flush=True)
        else:
            cap = cache / 'cap'
            disk(root, compiler=True)
            shell('''mkdir -p "$W/build" "$W/musl-src"
cp "$MUSL_ARCHIVE" "$W/musl-src/musl-1.2.5.tar.gz"
export CAPSTONE_LLVM_BUILD_DIR="$W/build" CAPSTONE_LLVM_BIN="$W/build/bin"
export MUSL_CACHE_ROOT="$W/musl-src" CAPSTONE_APPLICATION_PROFILE=virtual MUSL_SURVEY_JOBS=16
source "$T/capstone/tests/capstone-test-env.sh"
cmake -G Ninja -S "$T/llvm" -B "$W/build" -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_TARGETS_TO_BUILD=Capstone -DLLVM_ENABLE_PROJECTS='clang;lld' -DLLVM_PARALLEL_LINK_JOBS=4 -DLLVM_INCLUDE_BENCHMARKS=OFF
ninja -C "$W/build" -j "${CAP_JOBS:-20}" clang lld llc opt FileCheck count not llvm-config llvm-objdump llvm-nm llvm-ar llvm-objcopy llvm-readobj llvm-readelf llvm-symbolizer llvm-ranlib split-file llvm-mc
MUSLDIR=$(bash "$T/capstone/ports/musl-capstone/prepare-musl-capstone.sh" | tail -1)
OUT_DIR="$W/musl" bash "$T/capstone/ports/musl-capstone/build-musl-capstone.sh"
bash "$T/capstone/ports/common/application/build-sdk.sh" "$W/sdk" "$MUSLDIR" "$W/musl/libc-capstone.a"
''', out / 'toolchain-build.log', dict(env, MUSL_ARCHIVE=str(envpath('MUSL_ARCHIVE', '~/spec-cap/kit2/kit5/artifacts/sources/musl-1.2.5.tar.gz'))))
        disk(root)
        specroot = cache / 'SPEC17'
        copy_spec(Path(arms['spec_root']).expanduser(), specroot)
        kit = src / 'capstone/ports/spec-cpu2017'
        # Always rebuild our own label: sources, flags and SDK are from this commit.
        label = 'qc' + key
        shutil.copy2(kit / 'config/capstone-virtual.cfg', specroot / 'config/capstone-virtual.cfg')
        # SPEC stores executable verification hashes at the end of its installed
        # config. Keep those for the unchanged plain binaries; replacing this
        # file with the source-only config makes --nobuild refuse every run.
        check_plain_config(kit, specroot)
        shell('''source "$T/capstone/tests/capstone-test-env.sh"
cd "$SPEC_ROOT"
set +u; source ./shrc >/dev/null; set -u
runcpu --config=capstone-virtual --action=build --rebuild --tune=base --define sdk="$SDK" --define kit="$KIT" --define arm_label="$LABEL" 502.gcc_r 538.imagick_r 519.lbm_r 544.nab_r 500.perlbench_r 505.mcf_r 557.xz_r
''', out / 'spec-build.log', dict(env, SPEC_ROOT=str(specroot), SDK=str(cap / 'sdk'), KIT=str(kit), LABEL=label,
            CAPSTONE_LLVM_BUILD_DIR=str(cap / 'build'), CAPSTONE_LLVM_BIN=str(cap / 'build/bin')))
        record = dict(commit=commit, cap=str(cap), label=label, hashes=hashes,
                      clang_sha256=sha(cap / 'build/bin/clang'),
                      sdk_sha256=sha(cap / 'sdk/sdk.json'), sdk_identity=sdk_identity(cap),
                      executables={str(p): sha(p) for b in BENCHES
                                   for p in (specroot / 'benchspec/CPU' / b / 'exe').glob('*_base.*')})
        dump(cache / 'ready.json', record)
    record = json.loads((cache / 'ready.json').read_text())
    if record['hashes'] != hashes:
        raise ValueError('cache source hashes differ')
    cap = Path(record['cap'])
    if sha(cap / 'build/bin/clang') != record['clang_sha256'] or sha(cap / 'sdk/sdk.json') != record['sdk_sha256']:
        raise ValueError('cached compiler/SDK identity changed')
    if sdk_identity(cap) != record['sdk_identity']:
        raise ValueError('cached SDK dependency changed')
    for path, digest in record['executables'].items():
        if sha(Path(path)) != digest:
            raise ValueError(f'cached SPEC executable changed: {path}')
    check_plain_config(cache / 'src/capstone/ports/spec-cpu2017', cache / 'SPEC17')
    specroot = cache / 'SPEC17'
    missing = [b for b in BENCHES if not any(
        Path(p).parent.parent.name == b and p.endswith('_base.' + record['label'])
        for p in record['executables'])]
    if missing:
        disk(root)
        shell('''source "$T/capstone/tests/capstone-test-env.sh"
cd "$SPEC_ROOT"
set +u; source ./shrc >/dev/null; set -u
runcpu --config=capstone-virtual --action=build --tune=base --define sdk="$SDK" --define kit="$KIT" --define arm_label="$LABEL" $BENCHES
''', out / 'spec-extend.log', dict(os.environ, T=str(cache / 'src'),
            CAPSTONE_REPO_ROOT=str(cache / 'src'), CAPSTONE_TMP_ROOT=str(cache / 'tmp'),
            CAPSTONE_LLVM_BUILD_DIR=str(cap / 'build'), CAPSTONE_LLVM_BIN=str(cap / 'build/bin'),
            SPEC_ROOT=str(specroot), SDK=str(cap / 'sdk'),
            KIT=str(cache / 'src/capstone/ports/spec-cpu2017'), LABEL=record['label'],
            BENCHES=' '.join(missing)))
        for b in missing:
            built = list((specroot / 'benchspec/CPU' / b / 'exe').glob('*_base.' + record['label']))
            if not built:
                raise ValueError(f'{b}: missing built executable')
            record['executables'].update({str(p): sha(p) for p in
                (specroot / 'benchspec/CPU' / b / 'exe').glob('*_base.*')})
        dump(cache / 'ready.json.new', record)
        (cache / 'ready.json.new').replace(cache / 'ready.json')
    print(f"compiler+SDK {cap}; cache {key}", flush=True)
    # Fresh source, kit and adapter per milestone even when compiler inputs match.
    src = out / 'source'
    export(repo, commit, ['capstone/runtime', 'capstone/tests', 'capstone/ports/spec-cpu2017'], src)
    kit = out / 'kit'
    shutil.copytree(src / 'capstone/ports/spec-cpu2017', kit)
    for name in ('prof-northstar.py', 'prof-buckets.py', 'milestone-census.py', 'milestone-census.sh',
                 'profile-watchdog.py', 'capstone-run.sh', 'prof-report.py', 'prof_images.py'):
        shutil.copy2(HERE / name, kit / 'tools' / name)
    header = source_platform / 'adapter/module/cap_rev_table_abi.h'
    target = src / 'capstone/capstone-qemu/target/riscv/cap_rev_table_abi.h'
    target.parent.mkdir(parents=True)
    shutil.copy2(header, target)
    env = dict(os.environ, CAPSTONE_REPO_ROOT=str(src), CAPSTONE_TMP_ROOT=str(out / 'tmp'),
               CAPSTONE_QEMU_LOCK=str(out / 'qemu.lock'), CAPSTONE_BUILD_JOBS='4',
               CAPSTONE_LLVM_BUILD_DIR=str(cap / 'build'), CAPSTONE_LLVM_BIN=str(cap / 'build/bin'),
               KERNEL_BUILD=str(kernel), CROSS_COMPILE=cross)
    disk(root)
    shell('source "$CAPSTONE_REPO_ROOT/capstone/tests/capstone-test-env.sh"\nbash "$CAPSTONE_REPO_ROOT/capstone/runtime/virtual/build-adapter.sh" "$ADAPTER"',
          out / 'adapter-build.log', dict(env, ADAPTER=str(out / 'adapter')))
    disk(root)
    platform = out / 'platform'
    # Zero-filled image extents need not consume physical disk blocks. These
    # are still independent regular files, verified byte-for-byte below.
    run(['cp', '-rL', '--sparse=always', source_platform, platform])
    verify_platform(platform)
    shutil.rmtree(platform / 'adapter')
    shutil.copytree(out / 'adapter', platform / 'adapter')
    (platform / 'platform.json.done').unlink(missing_ok=True)
    files = {str(p.relative_to(platform)): sha(p) for p in platform.rglob('*')
             if p.is_file() and p.name not in ('platform.json', 'platform.json.done')}
    if any(p.is_symlink() for p in platform.rglob('*')):
        raise ValueError('platform contains symlinks')
    platform_manifest['files'] = files
    for name in list(platform_manifest['provenance']):
        if name.startswith('adapter/'):
            del platform_manifest['provenance'][name]
    platform_manifest['provenance']['adapter/*'] = f'build-adapter.sh at {commit}; see adapter-build.log'
    dump(platform / 'platform.json', platform_manifest)
    arms['share'] = str(root)
    arms['spec_root'] = str(cache / 'SPEC17')
    arms['vm_host'] = str(src / 'capstone/runtime/host')
    arms['nm'] = str(cap / 'build/bin/llvm-nm')
    arms['platform']['dir'] = str(platform)
    caparm = next(a for a in arms['arms'].values() if a['kind'] == 'capstone')
    caparm.update(label=record['label'], sdk=str(cap / 'sdk'), commit=commit,
                  note='milestone-census; source-hashes.json and build logs in this run')
    plain = next(a for a in arms['arms'].values() if a['kind'] == 'plain')
    arms['arms'] = {'cap': caparm, 'rv64': plain}
    arms.pop('stop_at', None)
    for a in arms['arms'].values():
        a.pop('stop_at', None)
    dump(out / 'arms.json', arms)
    env.update(CAPSTONE_SPEC_VM_STATES=str(root / 'vm-states'))
    env.pop('PROFILE_STOP_AT', None)
    disk(root)
    limit = int(os.environ.get('CENSUS_RUN_TIMEOUT', '600' if args.quick else '14400'))
    jobs = int(os.environ.get('JOBS', '4'))
    if limit <= 0 or jobs <= 0:
        raise ValueError('CENSUS_RUN_TIMEOUT and JOBS must be positive')
    def profile(pair):
        bench, arm = pair
        begin = time.monotonic()
        dest = out / arm / bench
        dest.parent.mkdir(exist_ok=True)
        settings = benchmark_arms(arms, bench)
        armfile = out / (bench + '-' + arm + '-arms.json')
        dump(armfile, settings)
        profile_env = dict(env)
        if args.quick:
            profile_env['PROFILE_STOP_AT'] = SLICES[bench]
        with (out / (bench + '-' + arm + '.log')).open('w') as stream:
            result = subprocess.run(['python3', str(kit / 'tools/profile-watchdog.py'),
                '--seconds', str(limit), '--out', str(dest), '--',
                'bash', str(kit / 'tools/profile.sh'), str(armfile), arm, 'test', bench, str(dest)],
                env=profile_env, stdout=stream, stderr=subprocess.STDOUT)
        verdict = json.loads((dest / 'result.json').read_text())['verdict'] if (dest / 'result.json').exists() else 'FAIL'
        return dict(arm=arm, benchmark=bench, verdict=verdict, returncode=result.returncode,
                    wall_seconds=time.monotonic()-begin)
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        rows = list(pool.map(profile, [(b, a) for b in BENCHES for a in ('cap', 'rv64')]))
    dump(out / 'batch.json', rows)
    (out / 'batch.log').write_text('\n'.join(
        f"{r['arm']} {r['benchmark']} verdict={r['verdict']} rc={r['returncode']} wall={r['wall_seconds']:.3f}s"
        for r in rows) + '\n')
    if any(r['returncode'] or r['verdict'] != ('SLICE' if args.quick else 'PASS') for r in rows):
        raise ValueError('one or more profiles failed; see batch.json (no completed census)')
    spec = importlib.util.spec_from_file_location('northstar', HERE / 'prof-northstar.py')
    ns = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ns)
    metrics = {}
    for b in BENCHES:
        r = ns.compare(out / 'rv64' / b, out / 'cap' / b)
        wanted = {'symbol': SLICES[b].split(':')[0], 'count': int(SLICES[b].split(':')[1]), 'start': None} if args.quick else None
        if r['work'] != wanted:
            raise ValueError(f'{b}: requested work boundary was not reached')
        metrics[b] = r
        with (out / (b + '-paired-functions.txt')).open('w') as stream:
            run(['python3', kit / 'tools/prof-compare.py', out / 'rv64' / b,
                 out / 'cap' / b, '--top', '15'], stdout=stream)
    previous = None
    oldmetrics = None
    previous_arm = os.environ.get('CENSUS_PREVIOUS_ARM', 'cap')
    if os.environ.get('CENSUS_PREVIOUS'):
        previous = envpath('CENSUS_PREVIOUS', '')
        oldmetrics = {b: ns.compare(previous / 'rv64' / b, previous / previous_arm / b)
                          for b in BENCHES if (previous / previous_arm / b).is_dir()}
    else:
        for candidate in sorted((root / 'runs').glob('*/northstar.json'), key=lambda p: p.stat().st_mtime, reverse=True):
            data = json.loads(candidate.read_text())
            if all(b in data['metrics'] and data['metrics'][b]['work'] == metrics[b]['work'] for b in BENCHES):
                previous, oldmetrics = candidate.parent, data['metrics']
                break
        if previous is None and not args.quick:
            previous = base / 'runs/census4'
            previous_arm = 'cap-dev4'
            oldmetrics = {b: ns.compare(previous / 'rv64' / b, previous / previous_arm / b)
                          for b in BENCHES if (previous / previous_arm / b).is_dir()}
    if oldmetrics and any(oldmetrics[b]['work'] != metrics[b]['work'] for b in oldmetrics):
        raise ValueError('refusing previous milestone with different work slices')
    lines = [f'commit {commit}; mode {"quick (SLICE, prefix only)" if args.quick else "full (PASS)"}',
             f'previous {previous or "none; establishing quick baseline"}',
             'benchmark program% boundary% delta-program-pp delta-boundary-pp H_A H_B']
    for b, r in metrics.items():
        delta = (f"{r['program_pct']-oldmetrics[b]['program_pct']:+.4f} {r['boundary_pct']-oldmetrics[b]['boundary_pct']:+.4f}"
                 if oldmetrics and b in oldmetrics else 'n/a n/a')
        lines.append(f"{b} {r['program_pct']:+.4f} {r['boundary_pct']:+.4f} {delta} {r['a']['H']} {r['b']['H']}")
        lines.append(f"  P_A={r['a']['P']} K_A={r['a']['K']} P_B={r['b']['P']} K_B={r['b']['K']}")
        if previous and b in oldmetrics:
            with (out / (b + '-functions.txt')).open('w') as stream:
                run(['python3', kit / 'tools/prof-compare.py', previous / previous_arm / b,
                     out / 'cap' / b, '--top', '15'], stdout=stream)
    elapsed = time.monotonic() - started
    lines.append(f'elapsed_seconds {elapsed:.3f} (includes preparation/builds)')
    text = '\n'.join(lines) + '\n'
    (out / 'northstar.txt').write_text(text)
    dump(out / 'northstar.json', dict(commit=commit, metrics=metrics, elapsed_seconds=elapsed,
                                     previous=str(previous) if previous else None))
    print(text, flush=True)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        sys.exit(f'milestone-census: UNRESOLVED: {exc}')
