import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from capstone_vm.cli import BINFMT_SETUP


class BinfmtTests(unittest.TestCase):
    def run_setup(self, root, mount_body='touch "$4/register" "$4/status"'):
        mount = root / "mount"
        mount.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$0.calls"\n' +
                         mount_body + '\n')
        mount.chmod(0o755)
        script = BINFMT_SETUP.replace('/proc/sys/fs/binfmt_misc', str(root / 'fs'))
        return subprocess.run(['sh', '-eu', '-c', script], capture_output=True,
                              env={**os.environ, 'PATH': str(root) + ':' + os.defpath})

    def test_mount_then_register_complete_elf_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'fs').mkdir()
            result = self.run_setup(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((root / 'mount.calls').read_text().strip(),
                             '-t binfmt_misc binfmt_misc ' + str(root / 'fs'))
            record = (root / 'fs/register').read_bytes()
            self.assertNotIn(b'\0', record)
            fields = record.decode().strip().split(':')
            self.assertEqual(fields[:4], ['', 'capstone', 'M', '0'])
            self.assertEqual(fields[6:], ['/usr/bin/capstone-exec', 'P'])
            magic = fields[4].encode().decode('unicode_escape').encode('latin1')
            mask = fields[5].encode().decode('unicode_escape').encode('latin1')
            self.assertEqual(len(magic), 20)
            self.assertEqual(len(mask), 20)
            def matches(header):
                return all((a & m) == (b & m) for a, b, m in zip(header, magic, mask))
            self.assertTrue(matches(bytes.fromhex('7f454c4602010100000000000000000002000301')))
            self.assertFalse(matches(bytes.fromhex('7f454c460201010000000000000000000200f300')))
            self.assertFalse(matches(bytes.fromhex('7f454c4601010100000000000000000002000301')))
            self.assertEqual((root / 'fs/status').read_text(), '1\n')

    def test_existing_mount_is_reused_and_stale_rule_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'fs').mkdir()
            (root / 'fs/register').touch()
            (root / 'fs/capstone').write_text('disabled\n')
            result = self.run_setup(root, 'exit 99')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((root / 'mount.calls').exists())
            self.assertEqual((root / 'fs/capstone').read_text(), '-1\n')
            self.assertIn('/usr/bin/capstone-exec', (root / 'fs/register').read_text())

    def test_mount_and_registration_errors_fail_setup(self):
        for mount_body in ['exit 19', 'mkdir "$4/register"']:
            with self.subTest(mount_body=mount_body), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / 'fs').mkdir()
                result = self.run_setup(root, mount_body)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((root / 'fs/status').exists())

    def test_kernel_without_support_remains_usable_and_reports_limitation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.run_setup(root, 'exit 99')
            self.assertEqual(result.returncode, 0)
            self.assertIn(b'direct image execution unavailable', result.stderr)
            self.assertFalse((root / 'mount.calls').exists())
