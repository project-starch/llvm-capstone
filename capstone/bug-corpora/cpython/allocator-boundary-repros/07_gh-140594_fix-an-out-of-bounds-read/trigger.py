#!/usr/bin/env python3
"""Trigger for gh140594 -- out-of-bounds read in PyOS_StdioReadline().

The upstream regression test (CmdLineTest.test_null_byte_in_interactive_mode)
does fail at the pin, but it runs the interpreter in a SUBPROCESS, so the
sanitizer report goes to the child and the harness never sees it. That is why
this case read as "regression test fails, sanitizer silent" for so long.

Feeding the NUL byte to an interactive interpreter and reading the child's
combined output back makes the report visible:

  AddressSanitizer: heap-buffer-overflow   (100-byte region)
  in PyOS_StdioReadline   Parser/myreadline.c:345

The buffer is PyMem_RawMalloc'd (myreadline.c:207, 223, 325, 347), so it is a
libc block whatever its size. This case is why the "<= 512 B means nested"
shortcut was dropped: at 100 bytes the shortcut would call it nested, and the
pymalloc arm proves otherwise.
"""
import os, subprocess, sys

exe = sys.executable
env = dict(os.environ)
env.setdefault("ASAN_OPTIONS", "detect_leaks=0:abort_on_error=0")
p = subprocess.run([exe, "-i"], input=b"\x00\n", env=env,
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
out = p.stdout.decode("utf-8", "replace")
sys.stderr.write(out)
print("gh140594 trigger ran; asan=%s"
      % ("yes" if "AddressSanitizer" in out else "no"))
