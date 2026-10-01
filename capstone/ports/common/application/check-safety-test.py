#!/usr/bin/env python3
"""A matching mark alone or an unrelated SIGSEGV must never pass a fixture."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('safety', Path(__file__).with_name('check-safety.py'))
safety = importlib.util.module_from_spec(spec)
spec.loader.exec_module(safety)


class SafetyTest(unittest.TestCase):
    def test_actual_exit_corresponds_to_full_mark(self):
        out = 'FFAPP-FIX 1 mark=100001\n'
        got, _ = safety.classify(out, '', dict(kind='exit', value=1), 1)
        self.assertEqual(got[:2], ('RETURN', '100001'))
        with self.assertRaises(ValueError):
            safety.classify(out, '', dict(kind='exit', value=2), 1)
        with self.assertRaises(ValueError):
            safety.classify(out, '', dict(kind='signal', value=11), 1)

    def test_fault_must_match_pc_and_target(self):
        result = dict(kind='signal', value=11, fault='domain fault cause=24 pc=0x1234 address=0x9000')
        diagnostic = 'Cap mem access OOB: pc = 1234, addr = 9000, size = 1, bounds = (8000, 9000)'
        out = 'TSAPP-FIX 2 target=9000\nTSAPP-FIX 2 touch\n'
        got, _ = safety.classify(out, diagnostic, result, 2)
        self.assertEqual(got[:2], ('FAULT', 'oob'))
        got, _ = safety.classify(out.replace('9000', '9010'), diagnostic, result, 2)
        self.assertEqual(got[0], 'FAULT-ELSEWHERE')
        got, _ = safety.classify('', diagnostic, result, 2)
        self.assertEqual(got[0], 'FAULT-BEFORE-TOUCH')
        with self.assertRaises(ValueError):
            safety.classify(out, diagnostic.replace('1234', '1235'), result, 2)


if __name__ == '__main__':
    unittest.main()
