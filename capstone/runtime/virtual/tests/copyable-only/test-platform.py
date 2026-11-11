#!/usr/bin/env python3
"""Adversarial checks for the privileged-operation fault oracle."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('platform_gate', Path(__file__).with_name('platform.py'))
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class RawFaultOracle(unittest.TestCase):
    def setUp(self):
        self.symbols = {'domain_main': 0x800}
        self.serial = ''
        for i, (op, word) in enumerate([('L', 0x428b), ('S', 0x500b), ('Q', 0x628b), ('A', 0x10628b)]):
            self.symbols['raw_site_'+op] = 0x1000+i*16
            # Each launch has a different slide, as in the real Linux gate.
            slide = (i+1)*0x400000
            self.serial += (f'COPYABLE_RAW_BEGIN:{op}\n'
                            f'domain fault cause=2 pc={slide+0x1000+i*16:#x} address={word:#x} '
                            f'ra=0x1234 entry={slide+0x800:#x} code=unused\n'
                            f'COPYABLE_RAW_REFUSED:{op}\n')

    def test_precise_faults_with_distinct_slides(self):
        self.assertTrue(gate.raw_faults_valid(self.serial, self.symbols))

    def test_unrelated_illegal_instruction(self):
        self.assertFalse(gate.raw_faults_valid(self.serial.replace('address=0x428b', 'address=0'), self.symbols))

    def test_right_opcode_wrong_pc(self):
        self.assertFalse(gate.raw_faults_valid(self.serial.replace('pc=0x401000', 'pc=0x401004'), self.symbols))

    def test_wrong_cause(self):
        self.assertFalse(gate.raw_faults_valid(self.serial.replace('cause=2', 'cause=5'), self.symbols))

    def test_missing_case(self):
        self.assertFalse(gate.raw_faults_valid(self.serial.replace('COPYABLE_RAW_BEGIN:Q', 'missing'), self.symbols))

    def test_duplicate_cases(self):
        self.assertFalse(gate.raw_faults_valid(self.serial+self.serial, self.symbols))


if __name__ == '__main__':
    unittest.main()
