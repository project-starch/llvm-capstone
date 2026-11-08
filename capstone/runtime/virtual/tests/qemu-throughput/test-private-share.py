"""Private 9p outputs must not become an accidental cross-VM shared directory."""
import concurrent.futures
import importlib.util
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import threading
import unittest

runtime = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(runtime))
spec = importlib.util.spec_from_file_location('staged', runtime / 'run-staged.py')
staged = importlib.util.module_from_spec(spec)
spec.loader.exec_module(staged)


class PrivateShareTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def test_private_output_is_new_and_private(self):
        output = self.root / 'output'
        staged.prepare_share(output, True, 'domain')
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o700)
        with self.assertRaises(FileExistsError):
            staged.prepare_share(output, True, 'another-domain')

    def test_existing_output_and_symlink_are_not_reused(self):
        output = self.root / 'output'
        output.mkdir()
        (output / 'receipt').write_text('keep')
        link = self.root / 'link'
        link.symlink_to(output, target_is_directory=True)
        for path in (output, link):
            with self.assertRaises(FileExistsError):
                staged.prepare_share(path, True, 'domain')
        self.assertEqual((output / 'receipt').read_text(), 'keep')

    def test_shared_output_requires_exclusive_mode(self):
        output = self.root / 'output'
        output.mkdir()
        with self.assertRaisesRegex(ValueError, 'requires --private-share'):
            staged.prepare_share(output, False, 'domain')
        staged.prepare_share(output, False, '')
        staged.prepare_share(None, False, 'serial-only-domain')
        with self.assertRaisesRegex(ValueError, 'requires --share'):
            staged.prepare_share(None, True, '')

    def test_competing_domains_cannot_reserve_the_same_path(self):
        output = self.root / 'output'
        barrier = threading.Barrier(2)
        def reserve(domain):
            barrier.wait()
            try:
                staged.prepare_share(output, True, domain)
                return 'reserved'
            except FileExistsError:
                return 'refused'
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            results = list(pool.map(reserve, ['a', 'b']))
        self.assertEqual(sorted(results), ['refused', 'reserved'])

    def test_timing_cannot_silently_use_a_domain(self):
        work = self.root / 'work'
        command = [sys.executable, str(runtime / 'run-staged.py'),
                   '--qemu', '/unused', '--images', '/unused', '--stage', '/unused',
                   '--work', str(work), '--host-window']
        result = subprocess.run(command, capture_output=True, text=True,
                                env=dict(os.environ, CAPSTONE_QEMU_DOMAIN_LOCK='domain'))
        self.assertEqual(result.returncode, 2)
        self.assertIn('global exclusive QEMU lock', result.stderr)
        self.assertFalse(work.exists())


if __name__ == '__main__':
    unittest.main()
