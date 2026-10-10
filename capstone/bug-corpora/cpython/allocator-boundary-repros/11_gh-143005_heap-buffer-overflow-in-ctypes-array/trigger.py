from ctypes import c_long

Base = c_long * 3
Wide = c_long * 5

victim = Base(1, 2, 3)
victim.__class__ = Wide # Make the CPython think we have a array whose length is 5.
victim[4] = 42  # OOB write