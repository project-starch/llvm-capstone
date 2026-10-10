"""Control bounds-block: a read just past the end of a pymalloc block.

_testinternalcapi.check_pyobject_forbidden_bytes_is_freed (Modules/_testinternalcapi.c) allocates
offsetof(PyObject, ob_type) bytes from pymalloc and reads ob_type, the field right after them. The
read lies in that function or in what it inlines (test_pyobject_is_freed, _PyObject_IsFreed). On a
stock pymalloc the next bytes are the same arena, so the call returns; with pymalloc's lifetime
port the block's capability is bounded to its request and the read faults.
"""
import os

import _testinternalcapi

os.write(1, b"CONTROL bounds-block mark\n")
try:
    _testinternalcapi.check_pyobject_forbidden_bytes_is_freed()
except AssertionError:
    pass
os.write(1, b"CONTROL bounds-block RETURNED\n")
os._exit(0)
