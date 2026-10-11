#!/usr/bin/env python3
"""Trigger for gh140492-large -- the reproducer from upstream issue #140492.

Sizes scaled x8192 from the upstream script so the victim object is
over SMALL_REQUEST_THRESHOLD (512 B) and libc malloc owns it. The
variant is kept only because it STILL fires: 524321-byte region.

Run on the pinned build it reports:
  AddressSanitizer: heap-buffer-overflow   (524321-byte region)
Needs ASAN_OPTIONS=detect_leaks=0 -- with leak checking on, the leak
summary buries the report.
"""
import ast
import unittest
class ASTConstructorTests(unittest.TestCase):
    def test_fields_and_types_no_default(self):
        class FieldsAndTypesNoDefault(ast.AST):
            _fields = (b'\xff' * 524288,)
            _field_types = {'a': int}
        with self.assertRaises(TypeError):
            FieldsAndTypesNoDefault()
if __name__ == "__main__":
    unittest.main()

