"""Negative control for case 32: the module stays alive, the traffic kept.

The defect is that `del sys.modules['unicodedata']` plus a collection drops the
last reference to the module, whose deallocation frees the
`_PyUnicode_Name_CAPI` struct, while the cached `_ucnhash_CAPI` pointer in the
interpreter survives and is then CALLED through.

Here the same statements run in the same order -- the same two namereplace
encodes, the same two compiles of a `\\N{...}` escape, the same
`del sys.modules['unicodedata']`, the same `gc.collect()` -- and one extra
reference is held across the del. The module object therefore stays alive, the
CAPI struct is never freed, and the cached pointer the second half of the
traffic calls through is valid. Everything else is identical, including the
dictionary mutation of `sys.modules` and the collection it triggers.

A fault here would mean the variant introduced a second defect rather than
removing the first, and the fault the trigger produces could not be attributed
to the defect. Running to its end is what lets the trigger's fault be read.
"""
import gc
import sys

assert '\N{GRINNING FACE}'.encode('ascii', errors='namereplace') == b'\\N{GRINNING FACE}'
compile(r"x = '\\N{LATIN CAPITAL LETTER A}'", '<x>', 'exec')

# The one line that differs from the trigger: hold the module across the del,
# so the same del and the same collection free nothing.
keep_alive = sys.modules['unicodedata']
del sys.modules['unicodedata']
gc.collect()

assert '\N{WINKING FACE}'.encode('ascii', errors='namereplace') == b'\\N{WINKING FACE}'
compile(r"x = '\\N{LATIN CAPITAL LETTER B}'", '<x>', 'exec')
assert keep_alive is not None
print("NEGATIVE-CONTROL no defect performed")
