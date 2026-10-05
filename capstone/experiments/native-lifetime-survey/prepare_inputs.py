#!/usr/bin/env python3
"""Fetch and verify input files. Keep third-party programs in scratch."""
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get('CAPSTONE_TMP_ROOT', '/tmp/capstone')) / 'native-survey'

class FirstPre(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active = False
        self.done = False
        self.text = []
    def handle_starttag(self, tag, attrs):
        if tag == 'pre' and not self.done:
            self.active = True
    def handle_endtag(self, tag):
        if tag == 'pre' and self.active:
            self.active = False
            self.done = True
    def handle_data(self, data):
        if self.active:
            self.text.append(data)

if __name__ == '__main__':
    dest = WORK / 'downloads'
    dest.mkdir(parents=True, exist_ok=True)
    for name, spec in json.loads((HERE/'inputs.json').read_text()).items():
        path = dest/name
        if not path.exists():
            subprocess.run(['curl','--fail','--location','--silent','--show-error','--retry','2',
                            spec['url'],'-o',str(path)+'.part'], check=True)
            Path(str(path)+'.part').rename(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != spec['sha256']:
            raise ValueError(f'{name}: input hash mismatch')
    parser = FirstPre()
    parser.feed((dest/'binarytrees.html').read_text())
    code = ''.join(parser.text)
    if 'sub bottomup_tree' not in code or not parser.done:
        raise ValueError('missing upstream Perl program')
    (dest/'binarytrees.pl').write_text(code)
    target = WORK / 'tools'
    target.mkdir(exist_ok=True)
    if not (target/'memtier_benchmark-2.2.1').exists():
        with tarfile.open(dest/'memtier.tar') as archive:
            archive.extractall(target, filter='data')
    print('PASS input hashes and upstream source extraction')
