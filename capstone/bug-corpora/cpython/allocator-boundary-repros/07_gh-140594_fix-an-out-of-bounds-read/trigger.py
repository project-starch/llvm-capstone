#!/usr/bin/env python3
"""Trigger for gh140594 -- out-of-bounds read in PyOS_StdioReadline().

The upstream regression test (CmdLineTest.test_null_byte_in_interactive_mode)
does fail at the pin, but it runs the interpreter in a SUBPROCESS, so the
sanitizer report goes to the child and the harness never sees it. That is why
this case read as "regression test fails, sanitizer silent" for so long.

A child is also a problem for the Capstone arms, and a worse one: the
application heap size is compiled into the binary
(CAPSTONE_APPLICATION_ARENA_BYTES), so a child wants a second heap of the same
size, and on the sublet arm the domain answered "cannot allocate application
heap: No space left on device". The parent then printed "trigger ran; asan=no",
which reads exactly like a silent arm while the defect had never run at all.

So this replaces the process instead of forking one. execv keeps a single
process and a single heap, and the fault happens in the process the harness is
already watching. The NUL byte is pushed into a pipe first because the
interpreter reads stdin only after the exec.

The signature, confirmed on the pinned 3.13.7 ASan build in this form:

  AddressSanitizer: heap-buffer-overflow   1 byte to the left of a 100-byte
  region, in PyOS_StdioReadline   Parser/myreadline.c:345

The buffer is PyMem_RawMalloc'd (myreadline.c:207, 223, 325, 347), so it is a
libc block whatever its size. This case is why the "<= 512 B means nested"
shortcut was dropped: at 100 bytes the shortcut would call it nested, and the
pymalloc arm proves otherwise.
"""
import os, sys

r, w = os.pipe()
os.write(w, b"\x00\n")        # 2 bytes, so the pipe buffer holds it with no reader
os.close(w)
os.dup2(r, 0)
os.environ.setdefault("ASAN_OPTIONS", "detect_leaks=0:abort_on_error=0")
os.execv(sys.executable, [sys.executable, "-i"])
