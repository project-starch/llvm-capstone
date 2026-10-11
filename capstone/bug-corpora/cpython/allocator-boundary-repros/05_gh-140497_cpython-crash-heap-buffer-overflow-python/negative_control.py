"""Negative control for case 05: co_code replaced with itself.

The defect is an EMPTY co_code: the interpreter reads past the end of a
zero-length instruction stream. replace() is still called, still builds a new
code object, and the function is still called through it -- the bytecode it gets
is simply the one it already had, which is valid.
"""
def f():
    pass

f.__code__ = f.__code__.replace(co_code=f.__code__.co_code)   # the defect: b""
f()
print("NEGATIVE-CONTROL no defect performed")
