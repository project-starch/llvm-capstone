"""Control uaf-block: a pymalloc block written and read through its pointer after it was freed.

_testinternalcapi.check_pyobject_freed_is_freed (Modules/_testinternalcapi.c) creates an object(),
one pymalloc block, deallocates it, sets its reference count and reads its type through the
dangling pointer. The stale accesses lie in that function or in what it inlines
(test_pyobject_is_freed, _PyObject_IsFreed). On a stock pymalloc the block stays inside its arena,
so the call returns (AssertionError or None); with pymalloc's lifetime port the block's capability
is retired on free and the write faults.

The process exits right after the call: the write overwrites the free-list link pymalloc keeps in
the freed block, so a later allocation of that size class would follow a corrupt link.
"""
import os

import _testinternalcapi

os.write(1, b"CONTROL uaf-block mark\n")
try:
    _testinternalcapi.check_pyobject_freed_is_freed()
except AssertionError:
    pass
os.write(1, b"CONTROL uaf-block RETURNED\n")
os._exit(0)
