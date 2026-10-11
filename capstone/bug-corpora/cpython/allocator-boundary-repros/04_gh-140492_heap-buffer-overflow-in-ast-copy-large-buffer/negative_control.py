"""Negative control for case 04: the same, at the large size.

Case 04 is case 03's defect with a 524288-byte field name, which is what puts
the object in a different allocation path. The control keeps the size.
"""
import ast
import unittest

class ASTConstructorTests(unittest.TestCase):
    def test_fields_and_types_no_default(self):
        class FieldsAndTypesNoDefault(ast.AST):
            _fields = ('a' * 524288,)      # the defect: b'\xff' * 524288
            _field_types = {'a': int}
        with self.assertRaises(TypeError):
            FieldsAndTypesNoDefault()

if __name__ == "__main__":
    print("NEGATIVE-CONTROL no defect performed")
    unittest.main(exit=False)
