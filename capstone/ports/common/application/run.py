#!/usr/bin/env python3
"""Stage an application image and run it through the shared VM launcher."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / 'capstone/runtime/host'))
from capstone_vm.cli import main as vm_main


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--cwd', default='/tmp')
    parser.add_argument('--user', help='Numeric guest UID:GID; the account must exist')
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--stdin', type=Path, help='Host file connected to application stdin')
    parser.add_argument('-e', '--env', action='append', default=[])
    parser.add_argument('image', type=Path)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    config = json.loads((args.state / 'config.json').read_text())
    share = Path(config['share'])
    staging = share / 'applications'
    staging.mkdir(exist_ok=True)
    # Snapshot before hashing: a concurrent rebuild cannot change the staged bytes.
    with tempfile.NamedTemporaryFile(dir=staging, prefix='.stage-', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            with args.image.open('rb') as source:
                shutil.copyfileobj(source, stream)
            stream.flush()
            with temporary.open('rb') as snapshot:
                digest = hashlib.file_digest(snapshot, 'sha256').hexdigest()
            image = staging / (args.image.stem + '-' + digest + '.dom')
            temporary.chmod(0o755)
            temporary.replace(image)
        finally:
            temporary.unlink(missing_ok=True)
    command = ['--state', str(args.state), 'run', '--cwd', args.cwd,
               '--result', str(args.result)]
    if args.user:
        command += ['--user', args.user]
    for value in args.env:
        command += ['-e', value]
    command += ['/mnt/host/' + str(image.relative_to(share)), *args.arguments]
    args.result.unlink(missing_ok=True)
    if args.stdin:
        with args.stdin.open('rb') as source:
            os.dup2(source.fileno(), 0)
            status = vm_main(command)
    else:
        status = vm_main(command)
    if args.result.exists():
        record = json.loads(args.result.read_text())
        record['image_sha256'] = digest
        args.result.write_text(json.dumps(record, indent=2) + '\n')
    return status


if __name__ == '__main__':
    sys.exit(main())
