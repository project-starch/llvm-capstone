#!/usr/bin/env python3
"""Reject incomplete application output, policy mismatches and broken accounting."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('runner', Path(__file__).with_name('cheribsd-run.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class VerdictTests(unittest.TestCase):
    def setUp(self):
        self.point = dict(expected_stdout='OK\n', expected_phases=['exit'], revocation=1)
        self.metrics = ('EXP-CHERI phase=exit revocation=1 heap_error=0 shadow_error=0 '
                        'allocated=32 active=64 resident=128\nEXP-GUEST-EXIT 0\n')

    def test_valid(self):
        self.assertEqual(runner.verdict(self.point, 0, 'OK\n', self.metrics), 'pass')

    def test_reject_false_passes(self):
        cases = [(255, 'OK\n', self.metrics),
                 (0, 'wrong\n', self.metrics),
                 (0, 'OK\n', self.metrics.replace('revocation=1', 'revocation=0')),
                 (0, 'OK\n', self.metrics.replace('shadow_error=0', 'shadow_error=14')),
                 (0, 'OK\n', self.metrics.replace('allocated=32', 'allocated=256')),
                 (0, 'OK\n', self.metrics.replace('allocated=32', 'allocated=-1')),
                 (0, 'OK\n', self.metrics.replace('allocated=32', 'allocated=x')),
                 (0, 'OK\n', self.metrics.replace('allocated=32', '')),
                 (0, 'OK\n', self.metrics.replace('phase=exit', 'phase=startup')),
                 (0, 'OK\n', self.metrics.replace('EXP-GUEST-EXIT 0', '')),
                 (0, 'OK\n', self.metrics.replace('EXIT 0', 'EXIT 162')),
                 (0, 'OK\n', self.metrics+'EXP-GUEST-EXIT 0\n')]
        for rc, out, err in cases:
            with self.subTest(rc=rc, out=out, err=err):
                self.assertNotEqual(runner.verdict(self.point, rc, out, err), 'pass')


if __name__ == '__main__':
    unittest.main()
