# musl for Capstone pure-capability domains

A port of musl's **arch layer** so that a POSIX libc can be compiled for
`capstone64-unknown-elf` and run inside a domain. The point is to stop
hand-writing a libc subset per workload: SQLite needed
`adapted/capstone_sqlite_libc.c` plus freestanding string routines plus VFS
stubs, and the next workload would need its own.

## Status

**A POSIX program runs.** `musl-hello/` is `#include <unistd.h>` plus two
`write()` calls, compiled against musl, running in a pure-capability domain,
with its I/O serviced by the host across the capability boundary:

```
musl-hello: write #1 through musl in a domain
S2: musl write() RETURNED
musl-hello: write #2, so write() RETURNED
__CAPSTONE_HOSTCALL_HOST_DONE__ status=0 serviced=4
```

| | |
|---|---|
| musl sources the compiler accepts | **1242 / 1361 (91.3 %)** |
| `libc-capstone.a` | links; `write(1,…)` leaves only `__capstone_hostcall` undefined |
| resumable hostcall from a pure-cap domain | works — `yield-probe/` |
| musl `write()` end to end | works — `musl-hello/` |
| syscalls implemented | `openat`, `read`, `write`, `close`, `fsync`, `fdatasync`, `ftruncate`, `exit`, `exit_group`; all else `-ENOSYS`, reported by number |
| **real file I/O** | **works** — `file-probe/`, open/write/fsync/close/open/read/close with the bytes compared |
| **reference Lua 5.4** | **runs** — `lua-probe/`, 22 core TUs against musl, `t[20] == 400` |

`lua-probe/` replaces the 1008 lines of hand-written libc that
`xlang/lua-cdp/capstone-lua/` carries. Three functions are stubbed and say so out
loud (`fopen`, `vfprintf`, `strtod`), because the musl files that define them do
not compile — all three are blocked by **`long double`**, which is unusable on
this target in compiler-rt as well as musl: every 128-bit builtin
(`comparetf2`, `addtf3`, `multf3`, `divtf3`, `extenddftf2`, `floatsitf`,
`trunctfdf2`, …) fails with the same backend assertions, because i128 is both a
capability and a `long double`.

Trail: `agent-handoff/history/14-08-2026_18-45-00_musl-write-runs-in-a-pure-cap-domain.md`,
`…_21-30-00_reference-lua-runs-on-musl-in-a-domain.md`, and `ISSUES.md` C-19 for
the compiler bug found on the way.

musl is **not vendored**. `fetch-musl.sh` downloads the official 1.2.5 archive
and verifies its SHA-256; the upstream tree stays immutable under
`$CAPSTONE_TMP_ROOT/musl-src`. Everything of ours is in this directory as an
overlay, so `diff -r arch/riscv64 arch/capstone64` in the staged tree is the
entire delta.

## The one file we wrote

`arch-capstone64/syscall_arch.h` replaces musl's riscv64 arch layer. Two
deviations, both forced by the capability ABI:

**1. `syscall_arg_t` is `void *`, not an integer.** Upstream marshals every
syscall argument through `((long)(X))` (`src/internal/syscall.h:22`). Under
pure-cap a pointer *is* a capability, and casting through ANY integer emits `mv`
and strips the tag. Measured on this target:

```
sizeof(void *)            == 16
sizeof(__UINTPTR_TYPE__)  ==  8      <- a plain integer
__uintcap_t / __intcap_t  do not exist
```

so there is no capability-carrying integer type; the only type that carries a
capability is a pointer. A pointer-to-pointer cast emits nothing at all.
Integer arguments (fd, count, flags) become untagged capabilities whose cursor
holds the value, which needs `-Wno-int-conversion`: int→pointer is an *error* in
current clang rather than a warning, and this ABI requires it.

musl anticipates the override — its definition is guarded by `#ifndef __scc`.

A first version used `__UINTPTR_TYPE__`, on the CHERI-shaped assumption that
uintptr_t is capability-width. It is not, and the result was musl's `write`
emitting `mv a2, a1` for the buffer while using `movc` for everything else: the
tag was stripped and the domain faulted in `helper_cscincoffset`.

**2. No `ecall`.** A domain cannot trap to Linux: its caller is a user process,
not a kernel, and the trap vector belongs to the monitor. The boundary is a call
to `__capstone_hostcall()`, implemented in `runtime/hostcall.c`, which
translates Linux syscall numbers into HostCall v0 opcodes. Keeping it an extern
call is what let the whole libc compile before any transport existed.

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
| C B + `arch/capstone64`, `syscall_arg_t` as an integer | 1270 / 1361 | 93.3 % | +27 |
| D C with `syscall_arg_t` as `void *` (**correct**) | **1242 / 1361** | **91.3 %** | **-28** |

Arm D is the shipped one. It is a *smaller* number than arm C on purpose: arm C
compiles more files and strips the tag off every pointer argument, so its libc
does not work. Trading 28 compilable files for a `write()` that works is the
right way round, and `BASELINE_OK` was lowered to 1242 with that reason recorded
in the survey.

A fourth arm was run and **rejected as unsound**: setting
`LDBL_MANT_DIG` to 53 in `bits/float.h` reaches 96.1 %, but only by telling musl
`long double` is `double` while the compiler still has it at 128 bits. musl
catches the lie itself — `src/stdio/vfprintf.c` fails with
`'compiler_defines_long_double_incorrectly' declared as an array with a negative
size`. The 39 long-double files are counted as **unresolved**, not fixed.

## What is left

119 files, grouped by what has to be done. Counts are by directory and sum to 119.

| files | where | dominant cause | shape of the fix |
|---:|---|---|---|
| 39 | `src/math` (31), `src/complex` (8) | **`long double` is 128-bit and so is a capability**, so it dies in i128 shift legalisation and in APInt | Backend: support `-mlong-double-64` for this target, or separate integer-i128 from capability-i128. The only genuinely new item on this list. |
| 10 | `src/malloc` (3), `mallocng` (6), `oldmalloc` (1) | static asserts over `sizeof(void*)`, violated by 16-byte pointers | Replace with an arena, as memsys5 does for SQLite today. |
| 9 | `src/string` | word-at-a-time routines: `(uintptr_t)s % ALIGN` | Replace, as SQLite already does with `beebs_freestanding_string.c`. |
| 61 | `unistd` (9), `thread`, `locale`, `network`, `stdio`, `stat`, `mman`, `regex`, `signal`, `aio`, … | mixed, mostly the same 128-bit-value family | Stub to `ENOSYS` for the first milestone; threads and `fork` are out of scope. The `void *` ABI moved ~28 files into this group, which is the cost recorded above. |

`__syscall_cp` needs **no patch**, contrary to an earlier plan here: musl
weak-aliases `__syscall_cp_c` to a `sccp` that calls `__syscall` directly
(`src/thread/__syscall_cp.c`), and the strong definition lives in
`pthread_cancel.c`. As long as that object is not linked, the alias wins and
cancellation points route through our hostcall like any other syscall.

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
