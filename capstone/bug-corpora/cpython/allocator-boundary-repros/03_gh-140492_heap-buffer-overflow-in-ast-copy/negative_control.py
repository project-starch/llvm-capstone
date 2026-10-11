"""Negative control for case 03: a str field name of the same length.

The defect is a bytes object where ast expects a str: reading those 64 bytes as
a field name overruns. 'a'*64 is the same length and the same allocation class,
so the constructor does the same work on a valid object.
"""
import ast
import unittest

class ASTConstructorTests(unittest.TestCase):
    def test_fields_and_types_no_default(self):
        class FieldsAndTypesNoDefault(ast.AST):
            _fields = ('a' * 64,)          # the defect: b'\xff' * 64
            _field_types = {'a': int}
        with self.assertRaises(TypeError):
            FieldsAndTypesNoDefault()

if __name__ == "__main__":
    print("NEGATIVE-CONTROL no defect performed")
    unittest.main(exit=False)
