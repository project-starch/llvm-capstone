#!/usr/bin/env python3
"""Python-level trigger for case 13 / gh-144833.

Use-after-free in the SSL module when ``SSL_new()`` fails
(``Modules/_ssl.c``, ``newPySSLSocket``).  On that error path the pre-fix code
ran ``Py_DECREF(self)`` and then ``_setSSLError(get_state_ctx(self), ...)`` --
reading a field of the ``PySSLSocket`` it just freed.  The fix is a two-line
swap: report the error before dropping the object.

REACHABILITY: this branch is entered only when OpenSSL's ``SSL_new(ctx)``
returns NULL, which happens only on an internal OpenSSL allocation failure.
There is no Python-level input that makes ``SSL_new`` fail, and CPython added no
Python regression test for this fix (the PR touched only ``Modules/_ssl.c``).
Unlike the decompressor case, there is also no ``_testcapi`` hook for OpenSSL's
allocator, so the defect cannot be driven from a standalone script on any build.
The demonstrable driver for this case is the C model (``case.c``), which
reproduces the free-then-read-own-field sequence directly.

This script performs the closest observable Python-level action (constructing an
SSL object, the operation whose failure path holds the defect) so the case has a
runnable trigger, then reports that the faulting branch is not Python-reachable.
"""
import socket
import ssl


def main():
    # Exercise the object-construction path that owns the defect.  On a healthy
    # OpenSSL, SSL_new() succeeds, so the buggy error branch is not taken.
    ctx = ssl.create_default_context()
    a, b = socket.socketpair()
    try:
        try:
            ctx.wrap_socket(a, server_hostname="example.invalid",
                            do_handshake_on_connect=False)
        except (ssl.SSLError, OSError, ValueError):
            pass
    finally:
        a.close()
        b.close()
    raise SystemExit(
        "trigger-13 gh-144833: the SSL_new()-failure branch is reached only on "
        "an OpenSSL allocation failure; no Python input triggers it and there is "
        "no fault-injection hook -- see case.c")


if __name__ == "__main__":
    main()
