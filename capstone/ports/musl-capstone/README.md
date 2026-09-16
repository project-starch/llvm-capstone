# musl for Capstone pure-capability domains

A port of musl's **arch layer** so that a POSIX libc can be compiled for
`capstone64-unknown-elf` and run inside a domain. The point is to stop
hand-writing a libc subset per workload: SQLite needed
`adapted/capstone_sqlite_libc.c` plus freestanding string routines plus VFS
stubs, and the next workload would need its own.

## Status

The compiler accepts **99.6 %** of musl's sources (2026-09-16, below; the
93.3 % this section carried until then is the 2026-08-14 figure), `libc-capstone.a` links, and
the domain boundary a syscall needs is **proven working under QEMU**:

- `write(1, ...)` linked against the partial archive leaves `__capstone_hostcall`
  as the **only** undefined symbol, so the 91 files that do not compile are not
  on the path of POSIX I/O.
- `yield-probe/` shows a pure-capability domain making a blocking hostcall and
  resuming with its C frame intact (`__CAPSTONE_YIELD_PROBE_PASSED__`). See
  `docs/history/14-08-2026_16-30-00_pure-cap-domain-resumable-hostcall-works.md`.

Not done: musl's own `write()` has not yet been run through that yield.
`runtime/hostcall.c` is written but unexercised.

musl is **not vendored**. `fetch-musl.sh` downloads the official 1.2.5 archive
and verifies its SHA-256; the upstream tree stays immutable under
`$CAPSTONE_TMP_ROOT/musl-src`. Everything of ours is in this directory as an
overlay, so `diff -r arch/riscv64 arch/capstone64` in the staged tree is the
entire delta.

## The one file we wrote

`arch-capstone64/syscall_arch.h` replaces musl's riscv64 arch layer. Two
deviations, both forced by the capability ABI:

**1. `syscall_arg_t` is capability-width, not `long`.** Upstream marshals every
syscall argument through `((long)(X))` (`src/internal/syscall.h:22`). Under
pure-cap a pointer *is* a capability, so that cast destroys it and the backend
then refuses to rebuild one from an integer:

```
fatal error: error in backend: Capstone PureCap: Cannot materialize arbitrary
>64-bit constants as capabilities; capabilities are unforgeable
```

musl anticipates the override: its definition is guarded by `#ifndef __scc`.
Defining `__scc` here suppresses both the macro and the `typedef long
syscall_arg_t`, and pointers reach the boundary intact. **This one change fixes
27 files** (`fopen`, `fstatat`, `mkdir`, `lchown`, `sem_timedwait`, ...).

**2. No `ecall`.** A domain cannot trap to Linux: its caller is a user process,
not a kernel, and the trap vector belongs to the monitor. The boundary is a call
to `__capstone_hostcall()`, which is declared here and not yet implemented.
Keeping it an extern call is what lets the whole libc compile before any
transport exists.

## Measured, 2026-08-14

Three arms, each differing from the previous one in exactly one thing. All on
`clang` built from this tree, `-target capstone64-unknown-elf -O1`, per-file
compile only. 1361 sources are surveyed; 169 foreign-architecture sources
(`src/**/x86_64/`, `aarch64/`, ...) are excluded and counted separately, because
musl's own build would not compile them for this target either.

| arm | compiled | | |
|---|---:|---:|---|
| A `arch/riscv64` unchanged, `+m` only | 1208 / 1361 | 88.8 % | 35 files fail on `lr.d`/`sc.d` |
| B A + `-target-feature +a` | 1243 / 1361 | 91.3 % | those 35 were a missing flag, not a port problem |
| C B + `arch/capstone64` (`syscall_arg_t`) | **1270 / 1361** | **93.3 %** | **+27** |

A fourth arm was run and **rejected as unsound**: setting
`LDBL_MANT_DIG` to 53 in `bits/float.h` reaches 96.1 %, but only by telling musl
`long double` is `double` while the compiler still has it at 128 bits. musl
catches the lie itself — `src/stdio/vfprintf.c` fails with
`'compiler_defines_long_double_incorrectly' declared as an array with a negative
size`. The 39 long-double files are counted as **unresolved**, not fixed.

## Measured again, 2026-09-16

Re-run with clang from `f7b50f081ca4`, which is `dev` plus three commits, only one of
which touches the backend and it is an unrelated `ra`-spill fix. So this is very
probably `dev`'s number too, and confirming that needs a `dev`-built compiler.

| | 2026-08-14 | 2026-09-16 |
|---|---:|---:|
| compiled | 1270 / 1361 | **1355 / 1361** |
| | 93.3 % | **99.6 %** |
| failing | 91 | **6** |

**The long-double item is gone, and it was the only genuinely new one.** The table
below used to list 39 files in `src/math` and `src/complex` that died in i128 shift
legalisation because `long double` is 128 bits and so is a capability. `fmodl.c`,
`powl.c` and `csqrtl.c` were built ONE AT A TIME with the survey's own flags, not read
off a total, and all three compile. `src/stdlib/qsort.c` compiles too, which retires
the `Cannot select: i128 = xor` blocker recorded against `__qsort_r` on 2026-08-20.
The string and the mixed pointer-as-integer groups are gone as well; none of them
needed the replacement work the old table planned.

**What is left is six files, and they are the ones a domain does not want.**

| files | where | cause | what to do |
|---:|---|---|---|
| 6 | `src/malloc/mallocng` | `sizeof(void*)` static assert; a 16-byte pointer makes the assert expression zero, so its negative-size array reports `array is too large (2^64-1 elements)` | Nothing. Every port here brings its own level 0, so no domain links musl's allocator. `src/malloc/mallocng/malloc.c` is now the survey's MUST_FAIL control for exactly that reason: it fails structurally and it blocks no milestone. |

So the compile side of this port is finished for practical purposes. **What remains is
the transport**, which a compile count never measured: `runtime/hostcall.c` translates
Linux syscall numbers to HostCall v0 opcodes and the probe harness beside it
(`tests/runtime-qemu/run-hostcall-stdout-probe.sh`) exercises that protocol from a
hand-written guest. musl's own `write()` has still not been run through it end to end.
That single path, not the 91 files, is now the libc question.

`__syscall_cp` needs **no patch**, contrary to an earlier plan here: musl
weak-aliases `__syscall_cp_c` to a `sccp` that calls `__syscall` directly
(`src/thread/__syscall_cp.c`), and the strong definition lives in
`pthread_cancel.c`. As long as that object is not linked, the alias wins and
cancellation points route through our hostcall like any other syscall.

## Run

```bash
source capstone/tests/capstone-test-env.sh
bash capstone/ports/musl-capstone/survey-musl-capstone.sh          # fetch, prepare, survey
bash capstone/ports/musl-capstone/survey-musl-capstone.sh --list-failures
```

Exit codes: `0` pass, `1` regression against the pinned `BASELINE_OK`, `2` the
harness could not measure (unprepared tree, no compiler, empty file list, or a
flipped control).

**The survey has two positive controls**, because a survey that cannot fail is
not evidence: `src/stdlib/abs.c` must compile and `src/string/strlen.c` must
fail. If either flips, the script prints ERROR instead of a number. When the
string routines are replaced, the second control has to be retired on purpose —
that is the intent, not an accident to work around.
