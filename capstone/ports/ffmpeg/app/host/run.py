#!/usr/bin/env python3
"""Run FFmpeg milestones and its changed-input oracle on a running shared VM."""
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
    parser.add_argument('stage', choices=['1', '2', '3', '4', '5', 'all'], nargs='?', default='all')
    parser.add_argument('--state', default=os.environ.get('CAPSTONE_VM_STATE'))
    args = parser.parse_args()
    if not args.state:
        parser.error('--state or CAPSTONE_VM_STATE is required')
    work = Path(os.environ.get('FFAPP_WORK', '/tmp/capstone/ffmpeg-app'))
    heap = os.environ.get('FFAPP_HEAP', 'level0')
    if heap not in ('level0', 'shrink', 'sublet'):
        parser.error('FFAPP_HEAP must be level0, shrink or sublet')
    domain = 'domain' + ('-' + heap if heap != 'level0' else '')
    if os.environ.get('FFAPP_POOL'):
        domain += '-pool' + os.environ['FFAPP_POOL']
    clip = os.environ.get('FFAPP_CLIP_SECONDS', '1')
    suffix = '' if clip == '1' else '-' + clip + 's'
    images = work / (domain + suffix)
    vm = VM(args.state, work / 'runs')
    try:
        original = vm.stage(work / ('input' + suffix + '.mkv'))
        stages = range(1, 6) if args.stage == 'all' else [int(args.stage)]
        for stage in stages:
            result = vm.run(f'm{stage}', images / f'ffapp_m{stage}.dom', [original])
            if result.get('kind') != 'exit' or result.get('value') != stage or result.get('fault'):
                raise RuntimeError(f'M{stage} did not complete: {result}')
            print(f'M{stage} REACHED', flush=True)
        if 5 in stages:
            flipped = vm.stage(work / ('input' + suffix + '.flip.mkv'))
            result = vm.run('flip', images / 'ffapp_m5.dom', [flipped])
            if result.get('kind') != 'exit' or result.get('value') != 5 or result.get('fault'):
                raise RuntimeError(f'flipped-input decoder did not complete: {result}')
            subprocess.run([sys.executable, str(HERE / 'compare-md5.py'),
                            str(work / ('stock' + suffix + '.framemd5')),
                            str(vm.output / 'm5.stdout'), '--control', str(vm.output / 'flip.stdout')], check=True)
        print(f'Results: {vm.output}')
    finally:
        vm.close()


if __name__ == '__main__':
    main()
