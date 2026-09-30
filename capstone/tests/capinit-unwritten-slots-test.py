#!/usr/bin/env python3
"""Regression controls for capinit-unwritten-slots.py; needs the Capstone tools.

source capstone/tests/capstone-test-env.sh
python3 capstone/tests/capinit-unwritten-slots-test.py

Real ELF fixtures exercise missed stores, source-only references, large offsets,
spills, duplicate archive members, and code the analyzer must refuse to certify.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent


def tool(name):
    return os.path.join(os.environ.get('CAPSTONE_LLVM_BIN', ''), name)


class StoreCoverage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def assemble(self, body, name='test', extra=''):
        source = self.root / (name + '.s')
        obj = source.with_suffix('.o')
        source.write_text('''
.data
.p2align 4
.type table, @object
table:
.quad target
.zero 8
.quad target
.zero 8
.size table, .-table
.type target, @object
target: .word 0
.size target, .-target
''' + extra + '''
.text
.type __capstone_cap_init, @function
__capstone_cap_init:
.Ltable:
auipc a0, %pcrel_hi(table)
addi a0, a0, %pcrel_lo(.Ltable)
''' + body + '''
cjalr zero, 0(ra)
.size __capstone_cap_init, .-__capstone_cap_init
''')
        subprocess.run([tool('clang'), '-target', 'capstone64-unknown-elf', '-c',
                        str(source), '-o', str(obj)], check=True, capture_output=True)
        return obj

    def scan(self, path, status):
        run = subprocess.run([sys.executable, str(HERE / 'capinit-unwritten-slots.py'),
                              str(path)], capture_output=True, text=True)
        self.assertEqual(run.returncode, status, run.stdout + run.stderr)
        return run.stdout, run.stderr

    def test_partial_table(self):
        out, _ = self.scan(self.assemble('stc a0, 0(a0)'), 1)
        self.assertIn('\ttable\t16\ttarget', out)
        self.assertNotIn('\ttable\t0\t', out)

    def test_source_reference_is_not_a_store(self):
        obj = self.assemble('''
.Lother:
auipc a1, %pcrel_hi(other)
addi a1, a1, %pcrel_lo(.Lother)
stc a0, 0(a1)
''', extra='''.p2align 4
.type other, @object
other: .quad table
.zero 8
.size other, .-other
''')
        out, _ = self.scan(obj, 1)
        self.assertIn('\ttable\t0\ttarget', out)
        self.assertIn('\ttable\t16\ttarget', out)
        self.assertNotIn('\tother\t', out)

    def test_spilled_destination(self):
        self.scan(self.assemble('''
cincoffsetimm sp, sp, -16
stc a0, 0(sp)
li a0, 0
ldc a1, 0(sp)
stc a1, 0(a1)
stc a1, 16(a1)
cincoffsetimm sp, sp, 16
'''), 0)

    def test_large_offset_and_addend(self):
        obj = self.assemble('''
li a2, 4096
cincoffset a1, a0, a2
stc a0, -2048(a1)
stc a0, 0(a0)
stc a0, 16(a0)
''', extra='''.zero 2012
.type distant, @object
distant: .quad target
.zero 8
.size distant, .-distant
''')
        self.scan(obj, 0)

    def test_scalar_overwrite_removes_coverage(self):
        out, _ = self.scan(self.assemble('stc a0, 0(a0)\nstc a0, 16(a0)\nsd zero, 8(a0)'), 1)
        self.assertIn('\ttable\t0\ttarget', out)
        self.assertNotIn('\ttable\t16\t', out)

    def test_unknown_control_flow_is_incomplete(self):
        _, err = self.scan(self.assemble('beq a0, zero, .Lend\nstc a0, 0(a0)\n.Lend:'), 2)
        self.assertIn('INCOMPLETE', err)

    def test_duplicate_archive_members_are_all_read(self):
        bad = self.assemble('stc a0, 0(a0)', 'same')
        archive = self.root / 'dup.a'
        subprocess.run([tool('llvm-ar'), 'qc', str(archive), str(bad)], check=True)
        good = self.assemble('stc a0, 0(a0)\nstc a0, 16(a0)', 'same')
        subprocess.run([tool('llvm-ar'), 'q', str(archive), str(good)], check=True)
        out, err = self.scan(archive, 1)
        self.assertIn('(same.o#1)', out)
        self.assertNotIn('(same.o#2)', out)
        self.assertIn('objects analyzed: 2', err)

    def test_compiler_mixed_alias_table(self):
        ir = self.root / 'alias.ll'
        ir.write_text('''
target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"
@lock = internal addrspace(200) global i32 0, align 4
@dummy = internal addrspace(200) constant ptr addrspace(200) null, align 16
@real = dso_local addrspace(200) constant ptr addrspace(200) @lock, align 16
@alias = weak dso_local alias ptr addrspace(200), ptr addrspace(200) @dummy
@table = dso_local addrspace(200) constant [2 x ptr addrspace(200)] [
 ptr addrspace(200) @real, ptr addrspace(200) @alias], align 16
''')
        obj = ir.with_suffix('.o')
        args = [tool('llc'), '-mtriple=capstone64', '-filetype=obj', str(ir), '-o', str(obj)]
        subprocess.run(args, check=True)
        self.scan(obj, 0)
        # Emit real and table[0], but omit table[1]: the actual C-75 failure.
        subprocess.run(args + ['-capstone-cap-init-limit=2'], check=True)
        out, _ = self.scan(obj, 1)
        self.assertIn('\ttable\t16\talias', out)


if __name__ == '__main__':
    unittest.main()
