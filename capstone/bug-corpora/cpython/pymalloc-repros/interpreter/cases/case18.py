#!/usr/bin/env python3
"""Python-level trigger for case 18 / gh-151695.

Use-after-free of the curses screen encoding (``Modules/_cursesmodule.c``).
``initscr()`` set the module-level ``curses_screen_encoding`` pointer to point
*into* the returned window object's encoding string.  Module-level functions
keep reading through that static after the window is deallocated.  The fix makes
the module keep a private copy of the encoding.

REACHABILITY: two obstacles keep this out of a standalone guest run.  First,
``initscr()`` needs a controlling terminal; the headless guest has none (run it
under a pty to get past this).  Second, and unlike the ``fork_exec`` case, the
window ``initscr()`` returns is also pinned by curses' own module state, so
freeing it at the point a module function next reads the global cannot be
arranged from Python without tearing curses down -- and CPython added no Python
regression test for this fix (the PR touched only ``Modules/_cursesmodule.c``).
The demonstrable driver for this case is the C model (``case.c``).

This script performs the reachable part (enter curses, read the screen
encoding, leave) so the case has a runnable trigger, guarded to skip cleanly
when there is no terminal.
"""
import sys


def main():
    try:
        import curses
    except ImportError:
        raise SystemExit("trigger-18 gh-151695: curses unavailable")

    try:
        stdscr = curses.initscr()
    except Exception as exc:
        raise SystemExit(
            "trigger-18 gh-151695: initscr() needs a terminal (%s); run under a "
            "pty. The window is pinned by curses state, so the freed-encoding "
            "read is not Python-reachable -- see case.c" % exc)

    try:
        _ = stdscr.encoding          # curses_screen_encoding is set here
    finally:
        curses.endwin()


if __name__ == "__main__":
    main()
    print("trigger-18 gh-151695: reached")
