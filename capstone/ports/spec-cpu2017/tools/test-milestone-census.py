#!/usr/bin/env python3
"""Source reuse and disk refusal regressions (no compiler or benchmark builds)."""
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('census', Path(__file__).with_name('milestone-census.py'))
census = importlib.util.module_from_spec(spec)
spec.loader.exec_module(census)


class CensusTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get('CAPSTONE_TMP_ROOT'))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_disk_reserve(self):
        with patch.object(census.shutil, 'disk_usage') as usage:
            usage.return_value.free = 15 * 1024**3 - 1
            with self.assertRaisesRegex(ValueError, 'less than 15 GiB'):
                census.disk(self.root)
            usage.return_value.free += 1
            census.disk(self.root)

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.root), *args], text=True).strip()

    def commit(self):
        self.git('add', '.')
        self.git('-c', 'user.name=test', '-c', 'user.email=fixture.invalid',
                 'commit', '-qm', 'fixture')
        return self.git('rev-parse', 'HEAD')

    def test_reuse_tracks_inputs_but_not_gate_docs(self):
        self.git('init', '-q')
        for path in census.COMPILER + census.SDK:
            dest = self.root / path
            if dest.suffix:
                dest.parent.mkdir(parents=True, exist_ok=True)
            else:
                dest.mkdir(parents=True, exist_ok=True)
                dest /= 'input.c'
            dest.write_text('input\n')
        virt = self.root / census.VIRTUAL
        virt.mkdir(parents=True)
        (virt / 'heap.c').write_text('heap\n')
        first = census.identities(self.root, self.commit())
        (virt / 'gates').mkdir()
        (virt / 'gates/test.py').write_text('gate\n')
        (virt / 'README.md').write_text('docs\n')
        self.assertEqual(first, census.identities(self.root, self.commit()))
        (virt / 'heap.c').write_text('changed heap\n')
        second = census.identities(self.root, self.commit())
        self.assertNotEqual(first, second)
        (self.root / 'llvm/input.c').write_text('changed compiler\n')
        self.assertNotEqual(second, census.identities(self.root, self.commit()))

    def test_spec_copy_is_private_and_keeps_suite_metadata(self):
        source = self.root / 'source'
        exe = source / 'benchspec/CPU/502.gcc_r/exe/app'
        exe.parent.mkdir(parents=True)
        exe.write_text('original')
        (source / 'benchspec/CPU/other').mkdir()
        (exe.parent.parent / 'run').mkdir()
        target = self.root / 'target'
        census.copy_spec(source, target)
        copied = target / exe.relative_to(source)
        copied.write_text('changed')
        self.assertEqual(exe.read_text(), 'original')
        self.assertTrue((target / 'benchspec/CPU/other').exists())
        self.assertFalse((target / 'benchspec/CPU/502.gcc_r/run').exists())


if __name__ == '__main__':
    unittest.main()
