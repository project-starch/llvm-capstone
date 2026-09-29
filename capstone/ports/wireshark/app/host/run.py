#!/usr/bin/env python3
"""Run delegated tshark stages or compare complete output with native tshark."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'common/application'))
from verification import VM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
    vm = VM(args.state, work / 'runs')
    passed = True
    try:
        config = vm.data / 'config'
        config.mkdir()
        guest_config = '/mnt/host/' + str(config.relative_to(vm.share))
        captures = args.captures if args.mode == 'oracle' else [os.environ.get('TSAPP_STAGES_CAPTURE', 'dhcp')]
        for capture in captures:
            if Path(capture).name != capture:
                parser.error('capture names must not contain directories')
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
                                ['TZ=UTC', 'HOME=' + guest_config, 'WIRESHARK_CONFIG_DIR=' + guest_config], args.user)
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
                                            env=dict(os.environ, TZ='UTC', HOME=str(config), WIRESHARK_CONFIG_DIR=str(config)))
                    (vm.output / (capture + '.native.stdout')).write_bytes(native.stdout)
                    (vm.output / (capture + '.native.stderr')).write_bytes(native.stderr)
                    if native.returncode or not native.stdout:
                        raise RuntimeError(f'{capture}: native oracle did not complete')
                    if capture.endswith('.flip'):
                        original = subprocess.run(
                            [str(stock), '-r', str(source), '-V', '-n'], capture_output=True,
                            env=dict(os.environ, TZ='UTC', HOME=str(config), WIRESHARK_CONFIG_DIR=str(config)))
                        (vm.output / (capture + '.original.native.stdout')).write_bytes(original.stdout)
                        if original.returncode or not original.stdout or original.stdout == native.stdout:
                            raise RuntimeError(f'{capture}: changed-input control did not change native output')
                    # ntp is deliberately outside the whitelist: a negative control.
                    equal = output == native.stdout
                    good = not equal if capture == 'ntp' else equal
                    passed &= good
                    print(f'{capture}: {"PASS" if good else "FAIL"}, stdout {"MATCH" if equal else "DIFFERS"}', flush=True)
        print(f'Results: {vm.output}')
    finally:
        vm.close()
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
