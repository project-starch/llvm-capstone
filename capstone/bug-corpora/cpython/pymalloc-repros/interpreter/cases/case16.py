#!/usr/bin/env python3
"""Python-level trigger for case 16 / gh-151403.

Use-after-free in ``_posixsubprocess.fork_exec`` (``Modules/_posixsubprocess.c``)
when an ``argv`` item's ``__fspath__`` mutates the args sequence.  In the argv
conversion loop ``borrowed_arg = PySequence_Fast_GET_ITEM(fast_args, arg_num)``
is borrowed and passed to ``PyUnicode_FSConverter`` with no ``Py_INCREF`` (the
fix adds one).  The item's ``__fspath__`` drops the list's last reference to it;
once ``__fspath__`` returns, that object is freed, and the converter then reads
its type (to format an error) through the stale ``borrowed_arg`` pointer.

There is no upstream Python regression test for this fix (the PR touched only C).
This driver reaches the exact hardened path directly through the public
``_posixsubprocess`` module -- confirmed to fault under a ``--without-pymalloc``
ASan build (heap-use-after-free in ``Py_TYPE`` while converting the argv item).
The fork_exec keyword order is the CPython 3.13 one; the fault happens in the
parent, before any fork.  Standalone; no _testcapi.
"""
import _posixsubprocess
import os


def main():
    errpipe_read, errpipe_write = os.pipe()

    class EvilPath:
        def __fspath__(self):
            # Drop the args list's last strong reference to this item, then
            # return an invalid type so the converter's error path reads the
            # (now freed) borrowed argument to name its type.
            args[1] = b"/bin/true"
            return 12345

    args = [b"/bin/true", EvilPath()]
    try:
        _posixsubprocess.fork_exec(
            args, [b"/bin/true"],
            True, tuple(),
            None, None,
            -1, -1, -1, -1,
            -1, -1,
            errpipe_read, errpipe_write,
            False, False,
            -1, None, None, None, -1,
            None, False,
        )
    except Exception:
        # A TypeError is expected on a fixed interpreter; on the pinned build
        # the stale read happens first.
        pass
    finally:
        for fd in (errpipe_read, errpipe_write):
            try:
                os.close(fd)
            except OSError:
                pass


if __name__ == "__main__":
    main()
    print("trigger-16 gh-151403: reached")
