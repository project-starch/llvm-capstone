#!/usr/bin/env python3
"""Negative controls for the mallocng source guard (verified archive required)."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
HERE=Path(__file__).resolve().parent
ARCHIVE=Path(os.environ.get('MUSL_ARCHIVE','/tmp/capstone/musl-src/musl-1.2.5.tar.gz'))

class Guard(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        with tarfile.open(ARCHIVE) as t:t.extractall(self.tmp.name,filter='data')
        self.tree=Path(self.tmp.name)/'musl-1.2.5'
        for name in ('0009-virtual-mallocng-capabilities.patch','0010-virtual-mallocng-meta-area-at-once.patch',
                     '0011-virtual-mallocng-grow-mmapped-chunks-in-place.patch'):
            with (HERE/'musl-patches'/name).open() as f:
                subprocess.run(['patch','-s','--no-backup-if-mismatch','-p1','-d',str(self.tree)],stdin=f,check=True)
    def check(self,ok):
        r=subprocess.run([sys.executable,str(HERE/'check-mallocng-policy.py'),str(self.tree),str(ARCHIVE)],capture_output=True,text=True)
        self.assertEqual(r.returncode,0 if ok else 1,r.stdout+r.stderr)
    def test_reviewed_port(self): self.check(True)
    def test_changed_policy(self):
        p=self.tree/'src/malloc/mallocng/malloc.c'
        s=p.read_text();self.assertIn('usage <= 12',s)
        p.write_text(s.replace('usage <= 12','usage <= 13'));self.check(False)
    def test_removed_revoke(self):
        p=self.tree/'src/malloc/mallocng/free.c'
        s=p.read_text();self.assertIn('heap_alloc_end(p, issued);',s)
        p.write_text(s.replace('heap_alloc_end(p, issued);',''));self.check(False)
    def test_unreviewed_file(self):
        (self.tree/'src/malloc/mallocng/registry.c').write_text('int registry;\n');self.check(False)

if __name__=='__main__':unittest.main()
