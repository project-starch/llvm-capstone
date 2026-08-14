# musl for Capstone pure-capability domains

A port of musl's **arch layer** so that a POSIX libc can be compiled for
`capstone64-unknown-elf` and run inside a domain. The point is to stop
hand-writing a libc subset per workload: SQLite needed
`adapted/capstone_sqlite_libc.c` plus freestanding string routines plus VFS
stubs, and the next workload would need its own.

## Status

**Survey stage.** The compiler accepts 93.3 % of musl's sources. Nothing is
linked yet and there is no hostcall transport yet, so nothing runs. The runnable
check is the survey; see *Run* below.

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

## What is left

91 files, grouped by what has to be done. Counts are by directory and sum to 91.

| files | where | dominant cause | shape of the fix |
|---:|---|---|---|
| 39 | `src/math` (31), `src/complex` (8) | **`long double` is 128-bit and so is a capability**, so it dies in i128 shift legalisation and in APInt | Backend: support `-mlong-double-64` for this target, or separate integer-i128 from capability-i128. The only genuinely new item on this list. |
| 10 | `src/malloc` (3), `mallocng` (6), `oldmalloc` (1) | static asserts over `sizeof(void*)`, violated by 16-byte pointers | Replace with an arena, as memsys5 does for SQLite today. |
| 9 | `src/string` | word-at-a-time routines: `(uintptr_t)s % ALIGN` | Replace, as SQLite already does with `beebs_freestanding_string.c`. |
| 33 | `thread` (8), `locale` (7), `network` (5), `aio`, `mman`, `regex`, `signal` (2 each), `exit`, `ldso`, `stdio`, `stdlib`, `multibyte` (1 each) | mixed, mostly the same pointer-as-integer family | Stub to `ENOSYS` for the first milestone; threads and `fork` are out of scope. |

Not in the survey because it only compiles `.c` files: `src/thread/__syscall_cp.c`
needs a `__syscall_cp_asm` for this target, and a single-threaded port can
forward it to `__syscall`. That is the next patch.

## Run

```bash
source capstone/tests/capstone-test-env.sh
bash capstone/musl-capstone/survey-musl-capstone.sh          # fetch, prepare, survey
bash capstone/musl-capstone/survey-musl-capstone.sh --list-failures
```

Exit codes: `0` pass, `1` regression against the pinned `BASELINE_OK`, `2` the
harness could not measure (unprepared tree, no compiler, empty file list, or a
flipped control).

**The survey has two positive controls**, because a survey that cannot fail is
not evidence: `src/stdlib/abs.c` must compile and `src/string/strlen.c` must
fail. If either flips, the script prints ERROR instead of a number. When the
string routines are replaced, the second control has to be retired on purpose —
that is the intent, not an accident to work around.
