#!/usr/bin/env python3
"""Trigger for gh140492 -- the reproducer from upstream issue #140492.

Run on the pinned build it reports:
  AddressSanitizer: heap-buffer-overflow   (97-byte region)
Needs ASAN_OPTIONS=detect_leaks=0 -- with leak checking on, the leak
summary buries the report.
"""
import ast
import unittest
class ASTConstructorTests(unittest.TestCase):
    def test_fields_and_types_no_default(self):
        class FieldsAndTypesNoDefault(ast.AST):
            _fields = (b'\xff'*64,)
            _field_types = {'a': int}
        with self.assertRaises(TypeError):
            FieldsAndTypesNoDefault()
if __name__ == "__main__":
    unittest.main()

