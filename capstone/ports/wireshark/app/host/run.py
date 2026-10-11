#!/usr/bin/env python3
"""Run delegated tshark stages or compare complete output with native tshark.

Two checks carried over from the pre-v2 runner (host/run-qemu.sh before #128), which judged them and
whose loss made the chunks arm's registered T3 and T5 uncheckable:

- The revocation-node budget per guest boot. A full run on the sublet heap spends about 12,600
  nodes (split + mrev on its TSAPP-HEAP line), and a boot that ran out died on QEMU's pool
  assertion (2026-09-25). So a boot holds at most 4 full sublet runs, and at most
  TSAPP_CHUNKS_RUNS_PER_BOOT (default 1) chunks runs: the chunk port spends nodes too, by an amount
  a measured run sets (wmem/PREREGISTRATION-tshark-step2.md, T5). The guest outlives this process,
  so the count is kept per guest boot (its boot_id) in the VM's state directory, and a request that
  would pass the limit is refused before anything runs.
- stderr against the native MINIMAL build (TSAPP_MINIMAL, default $TS_WORK/native-min-pa/run/tshark:
  the same whitelist, patches and generated dissectors.c), because patch 0002's registration
  notices go to stderr and stock tshark writes none. The guest's stderr, less its TSAPP-HEAP line
  (src/tsapp-heap.c) and, when the domain runs with CAPSTONE_DELEGATE_STATS, the runtime's and
  launcher's `capstone-domain:` / `capstone-exec:` report lines, must equal it byte for byte. An
  empty reference is an error, never a comparison: it would match anything. TSAPP_MINIMAL=none
  skips the check and says so on every verdict line.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(HERE.parents[2] / 'common/application'))
from verification import VM

# Full (M5) runs a guest boot may hold, per heap arm; None: not limited.
RUNS_PER_BOOT = {'sublet': 4, 'chunks': int(os.environ.get('TSAPP_CHUNKS_RUNS_PER_BOOT', '1'))}
REPORT_PREFIXES = (b'capstone-domain: ', b'capstone-exec: ')


def guest_boot_id(state):
    env = dict(os.environ, PYTHONPATH=str(REPO / 'capstone/runtime/host'))
    out = subprocess.run([sys.executable, '-m', 'capstone_vm', '--state', str(state), 'exec',
                          'cat', '/proc/sys/kernel/random/boot_id'],
                         capture_output=True, text=True, env=env, check=True).stdout.strip()
    if len(out) != 36:
        raise RuntimeError(f'no guest boot_id: {out!r}')
    return out


def claim_runs(state, heap, planned):
    """Charge `planned` full runs to this guest boot, or refuse. Returns a line for the log."""
    limit = RUNS_PER_BOOT.get(heap)
    if limit is None:
        return None
    boot = guest_boot_id(state)
    ledger = Path(state) / 'tsapp-full-runs.json'
    used = 0
    if ledger.exists():
        record = json.loads(ledger.read_text())
        used = record['runs'] if record.get('boot_id') == boot else 0
    if used + planned > limit:
        sys.exit(f'refused: {planned} full {heap} run(s) would make {used + planned} on guest boot '
                 f'{boot}, over the limit of {limit} (the revocation-node budget; '
                 + ('TSAPP_CHUNKS_RUNS_PER_BOOT' if heap == 'chunks' else 'fixed for the sublet arm')
                 + '). Restart the VM for a fresh boot.')
    # Charged before the runs: a run that fails has still spent its nodes.
    ledger.write_text(json.dumps({'boot_id': boot, 'runs': used + planned}))
    return f'guest boot {boot}: {heap} full runs {used} + {planned} of {limit}'


def guest_stderr(data, stats):
    lines = data.split(b'\n')
    kept = [l for l in lines
            if not l.startswith(b'TSAPP-HEAP ') and not (stats and l.startswith(REPORT_PREFIXES))]
    return b'\n'.join(kept)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', choices=['stages', 'oracle'])
    parser.add_argument('captures', nargs='*')
    parser.add_argument('--state', default=os.environ.get('CAPSTONE_VM_STATE'))
    parser.add_argument('--user', default=os.environ.get('TSAPP_USER'))
    args = parser.parse_args()
    if not args.state:
        parser.error('--state or CAPSTONE_VM_STATE is required')
    if args.mode == 'oracle' and not args.captures:
        parser.error('oracle needs capture names')
    work = Path(os.environ.get('TS_WORK', '/tmp/capstone/tshark-app'))
    heap = os.environ.get('TSAPP_HEAP', 'level0')
    if heap not in ('level0', 'shrink', 'sublet', 'chunks'):
        parser.error('TSAPP_HEAP must be level0, shrink, sublet or chunks')
    images = Path(os.environ.get('TSAPP_DOMAIN_DIR', work / ('domain' + ('-' + heap if heap != 'level0' else ''))))
    stock = Path(os.environ.get('TSAPP_STOCK', work / 'native-stock/run/tshark'))
    minimal = os.environ.get('TSAPP_MINIMAL', str(work / 'native-min-pa/run/tshark'))
    if minimal != 'none' and not Path(minimal).is_file():
        parser.error(f'TSAPP_MINIMAL: no native minimal build at {minimal} (build native-min-pa, '
                     'or set TSAPP_MINIMAL=none to skip the stderr verdict)')
    # TSAPP_DOMAIN_ENV: extra NAME=value pairs for the domain, comma-separated; the delegated
    # runtime prints its unserved-syscall report to stderr under CAPSTONE_DELEGATE_STATS=1
    extra_env = [e for e in os.environ.get('TSAPP_DOMAIN_ENV', '').split(',') if e]
    stats = any(e.split('=', 1)[0] == 'CAPSTONE_DELEGATE_STATS' for e in extra_env)
    captures = args.captures if args.mode == 'oracle' else [os.environ.get('TSAPP_STAGES_CAPTURE', 'dhcp')]
    for capture in captures:
        if Path(capture).name != capture:
            parser.error('capture names must not contain directories')
    charged = claim_runs(args.state, heap, len(captures))
    if charged:
        print(charged, flush=True)
    vm = VM(args.state, work / 'runs')
    passed = True
    try:
        config = vm.data / 'config'
        config.mkdir()
        guest_config = '/mnt/host/' + str(config.relative_to(vm.share))
        native_env = dict(os.environ, TZ='UTC', HOME=str(config), WIRESHARK_CONFIG_DIR=str(config))
        for capture in captures:
            source = work / 'xsrc/test/captures' / (capture.removesuffix('.flip') + '.pcap')
            data = bytearray(source.read_bytes())
            if capture.endswith('.flip'):
                data[-8] ^= 1
            local = vm.output / (capture + '.pcap')
            local.write_bytes(data)
            guest = vm.stage(local)
            for stage in (range(1, 6) if args.mode == 'stages' else [5]):
                name = capture + '-m' + str(stage)
                result = vm.run(name, images / f'tshark_m{stage}.dom', ['-r', guest, '-V', '-n'],
                                ['TZ=UTC', 'HOME=' + guest_config, 'WIRESHARK_CONFIG_DIR=' + guest_config] + extra_env,
                                args.user)
                want = 100 + stage if stage < 5 else 0
                if result.get('kind') != 'exit' or result.get('value') != want or result.get('fault'):
                    raise RuntimeError(f'{name} failed: {result}')
                output = (vm.output / (name + '.stdout')).read_bytes()
                if stage < 5:
                    diagnostics = (vm.output / (name + '.stderr')).read_bytes()
                    if f'TSAPP-STAGE {stage}'.encode() not in diagnostics.splitlines():
                        raise RuntimeError(f'{name}: missing stage marker')
                    print(f'M{stage} REACHED', flush=True)
                else:
                    native = subprocess.run([str(stock), '-r', str(local), '-V', '-n'], capture_output=True,
                                            env=native_env)
                    (vm.output / (capture + '.native.stdout')).write_bytes(native.stdout)
                    (vm.output / (capture + '.native.stderr')).write_bytes(native.stderr)
                    if native.returncode or not native.stdout:
                        raise RuntimeError(f'{capture}: native oracle did not complete')
                    if capture.endswith('.flip'):
                        original = subprocess.run(
                            [str(stock), '-r', str(source), '-V', '-n'], capture_output=True, env=native_env)
                        (vm.output / (capture + '.original.native.stdout')).write_bytes(original.stdout)
                        if original.returncode or not original.stdout or original.stdout == native.stdout:
                            raise RuntimeError(f'{capture}: changed-input control did not change native output')
                    # ntp is deliberately outside the whitelist: a negative control.
                    equal = output == native.stdout
                    good = not equal if capture == 'ntp' else equal
                    if minimal == 'none':
                        err = 'stderr not checked (TSAPP_MINIMAL=none)'
                    else:
                        ref = subprocess.run([minimal, '-r', str(local), '-V', '-n'], capture_output=True,
                                             env=native_env)
                        (vm.output / (capture + '.minimal.stderr')).write_bytes(ref.stderr)
                        if ref.returncode or not ref.stderr:
                            raise RuntimeError(f'{capture}: the native minimal build wrote no stderr '
                                               f'(status {ref.returncode}); an empty reference matches anything')
                        got = guest_stderr((vm.output / (name + '.stderr')).read_bytes(), stats)
                        err_equal = got == ref.stderr
                        good &= err_equal
                        err = f'stderr {"MATCH" if err_equal else "DIFFERS"}'
                    passed &= good
                    print(f'{capture}: {"PASS" if good else "FAIL"}, stdout {"MATCH" if equal else "DIFFERS"}, {err}',
                          flush=True)
        print(f'Results: {vm.output}')
    finally:
        vm.close()
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
