#!/usr/bin/env python3
"""Expand fetched sources in scratch. Original trees remain untouched."""
import json
import os
from pathlib import Path
import shutil
import tarfile
import zipfile

DEST = Path(os.environ.get('CAPSTONE_TMP_ROOT', '/tmp/capstone')) / 'native-survey'

if __name__ == '__main__':
    for name, spec in json.loads((DEST / 'sources.json').read_text()).items():
        target = DEST / 'original' / name
        if target.exists():
            continue
        staging = DEST / 'unpack' / name
        staging.mkdir(parents=True, exist_ok=True)
        archive = DEST / spec['archive']
        if archive.suffix == '.zip':
            with zipfile.ZipFile(archive) as src:
                src.extractall(staging)
        else:
            with tarfile.open(archive) as src:
                src.extractall(staging, filter='data')
        roots = list(staging.iterdir())
        if len(roots) != 1 or not roots[0].is_dir():
            raise ValueError(f'{name}: expected one archive root')
        target.parent.mkdir(parents=True, exist_ok=True)
        roots[0].rename(target)
        staging.rmdir()
        print(name, flush=True)
