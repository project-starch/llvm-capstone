# Real file I/O from a pure-capability domain, through musl, 2026-08-14

**Result, QEMU, verified.** `open` → `write` → `fsync` → `close` → `open` →
`read` → `close` on a guest file, from ordinary POSIX code in a domain, with the
bytes read back and compared:

```
FILE S2: opened for write
FILE S3: wrote payload
FILE S4: synced and closed
FILE S5: reopened for read
FILE S6: read back and compared equal
__CAPSTONE_FILE_PROBE_PASSED__
__CAPSTONE_HOSTCALL_HOST_DONE__ status=0 serviced=14
```

Harness: `capstone/musl-capstone/file-probe/run-file-probe.sh`.

**The check compares bytes, not counts.** A `write` returning 32 proves only that
a request reached the host: it could have written the wrong file, at the wrong
offset, or the domain's copy into the payload region could be wrong in a way a
count does not show. Reading back and comparing fails if any of those is true.

## What was wired

`SYS_openat`, `SYS_read`, `SYS_write` (to a real fd), `SYS_close`, `SYS_fsync`,
`SYS_fdatasync`, `SYS_ftruncate` now translate onto the HostCall v0 file
opcodes, which already existed and were already serviced by the
`hostcall-file-*` probes. The host side reuses
`tests/runtime-qemu/hostcall-file-service-probe-common.h` for the handle-token
table rather than growing a second one.

Two impedance mismatches, handled rather than papered over:

- **HostCall v0 read/write take an explicit file offset; POSIX uses the
  descriptor's implicit position.** The position lives in the domain, one per
  open handle, advanced on every transfer.
- **The host's handle tokens start at 1, and fds 1 and 2 are already stdout and
  stderr** on the WRITE_STDOUT path. A token is exposed to the program as
  `token + 2`, so tokens 1..8 become fds 3..10 and cannot collide.

## The finding that came out of it: C-21

`open()` could not simply come from musl, because `src/fcntl/open.c` does not
compile. Writing our own hit the *same* assertion, which ruled out the file and
pointed at what it does. Bisecting the cast:

```
(void *)1                    OK
(void *)-100                 FAILS   getActiveBits() <= 64
(void *)(long)runtime_value  OK
```

**A negative integer CONSTANT cast to a capability crashes the backend**, and
`AT_FDCWD` is `-100`, so every `*at()` wrapper in musl performs exactly this
cast. Recorded as ISSUES.md **C-21**, separate from C-20: same assertion text,
different cause, and this one has a three-line workaround (route the constant
through a `volatile`) and probably a small backend fix.

**A correction that came with it.** `open.c` and `fopen.c` were attributed to the
long-double family earlier today, on the strength of the shared assertion text.
That was wrong; they are C-21. 19 of the 119 non-compiling musl files name
`AT_FDCWD` — an indicator, not an apportionment, since the 119 have not been
split between the two causes and some files may hit both.

## Instrument note

The first `open()` wrapper was blamed on `va_arg`, because `vfprintf` fails the
same way and both are variadic. That was a guess, and it was wrong: a four-case
test showed `va_arg` works for `int`, `long`, `void *` and `double`. Bisecting
the individual casts found the real one in two minutes. The lesson is the usual
one in a different costume — two failures sharing a symptom are not evidence of
a shared cause.

## State after this

| | |
|---|---|
| syscalls implemented | `write` (stdout/stderr and files), `read`, `openat`, `close`, `fsync`, `fdatasync`, `ftruncate`, `exit`, `exit_group` |
| still refused | everything else, `-ENOSYS`, reported by number |
| known-missing and wanted | `clock_gettime`/`gettimeofday` — Lua's hash seed depends on them (see the Lua note) |
| `fstat` | opcode exists (`FILE_STAT_BASIC`), not yet translated |

Next in this direction: `fstat`, then `lseek`, which together are what a stdio
implementation needs. That would also be the point at which `fopen` becomes
worth revisiting — it is blocked by C-21, not by a missing service.
