# musl for Capstone pure-capability domains

A port of musl's **arch layer** so that a POSIX libc can be compiled for
`capstone64-unknown-elf` and run inside a domain. The point is to stop
hand-writing a libc subset per workload: SQLite needed
`adapted/capstone_sqlite_libc.c` plus freestanding string routines plus VFS
stubs, and the next workload would need its own.

## Status

**A domain runs musl.** Not compiles, runs: musl's own functional test suite
builds and runs inside a pure-capability domain under QEMU, one boot per test,
and the table below accounts for every one of its 77 sources.

| | |
|---|---|
| sources the compiler accepts | 1355 / 1361 (99.6 %) |
| `write` to stdout, `errno` on failure | works |
| `open` `read` `write` `close` `lseek` incl. SEEK_END | works |
| `writev` `readv` `fstat` `fsync` `ftruncate` `unlink` `access` | works |
| `fopen` `fprintf("%f")` `fgets` `fseek` `printf` `fflush` | works |
| `malloc` `calloc` `realloc` `free` | works, from this port's own level 0, under all five names musl calls it by |
| `exit` from anywhere | works, and it has to: a refused exit is an unbreakable loop |
| `setjmp` `longjmp` `sigsetjmp` | works, from this port's own assembly |
| `clock_gettime` | works, through a new HostCall opcode (`CLOCK_GETTIME`, 25) |
| `stat` `getuid` `getgid` | works, path stat as open, fstat, close |
| real pthreads | **no**, see C-47 |
| `fork` `exec` `pipe` `socket` `dlopen` | **no**, and none of them is on the way: a domain is one process |
| musl's own test suite | 77 tests accounted for, see below |

Three probes, each green under QEMU with zero faults:

| probe | what it establishes |
|---|---|
| `write-probe/` | musl's `write` reaches the hostcall, and its error reaches `errno` |
| `file-probe/` | the file service, with an exact round count that a stale descriptor would break |
| `stdio-probe/` | the layer above, where musl stops calling `write` and starts calling `writev` |

## What running it cost, and why compiling did not predict it

Nine places sent a capability through something that carries 64 bits. Four were
in this port, two were in musl, two were missing syscalls and one was an include
line. **Every file involved is on the compiling side of the 99.6 %.**

| where | what |
|---|---|
| `syscall_arch.h` | `__UINTPTR_TYPE__` is `unsigned long` here, so the fix was the type it replaced |
| `pthread_arch.h` | `__get_tp()` returned `uintptr_t` and read `tp` with `mv` |
| `__set_thread_area` | upstream's riscv64 `.s` is `mv tp, a0` |
| `start-musl.S` | `__capstone_yield` saved `ra`, `gp` and `s0-s11`, not `tp` |
| `hostcall.c` | used `syscall_arg_t` without including the header that defines it |
| musl's string routines | seven read a machine word at a time, over the end of the object |
| musl's `lite_malloc` | reads `brk` as a `uintptr_t` and returns `(void *)(brk - req)` |
| stdio | calls `writev` and `readv`, never `write` or `read` |
| link | eleven undefined `__*tf*` for any program that links `printf` |

The last one is the only one a link would have caught. The others need a domain
that actually runs, which is what the probes are for.

**What is not served is recorded, not merely refused.** The default arm of
`__capstone_hostcall` keeps the syscall numbers it turned away, and a probe
prints them after its work is done. `-ENOSYS` alone is invisible: musl turns most
of them into a plausible failure and the caller continues, so a missing opcode
surfaces later as wrong behaviour somewhere unrelated. A full stdio run currently
reports an empty list, which is the only state in which the list is worth reading.

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

## libc-test: musl's own suite, in a domain

`libc-test/` builds every functional test of musl's libc-test as its own pure-capability
domain and runs it under QEMU in its own boot. The result is one table that accounts for
all 77 sources, not only the ones that produced a domain.

```bash
bash capstone/ports/musl-capstone/libc-test/run-libc-test.sh   # builds, one boot per test, summarises
cat  "$CAPSTONE_TMP_ROOT/musl-libc-test/logs/latest/results.txt"   # one directory per run, with toolchain.txt
```

Verdicts: `PASS` and `FAIL` are the test's own, `t_status` folded with the unserved-syscall
count so a test that "passed" while a syscall it needed was refused does not pass.
`NOBUILD` carries the first compile or link error, `EXCLUDED` the reason from
`build-libc-test.sh`, `FAULT` a capability fault, `HUNG` a test that never came back and took its
chunk's guest with it, `NOBOOT` a boot that never reached a shell, `NOTRUN` a chunk-mate of one
that hung. A `FAIL` that says `UNSERVED` names the syscalls the domain asked for and the host
refused, so the row says what the test needed and not only that it failed.

**Why one boot per test.** A domain that never comes back takes the guest with it: a fault
after a yield does (M-1), and so does a domain that spins, because it holds the only hart.
Nothing inside the guest can recover it, not `alarm()` in the host process, which is inside the
ioctl, not `kill -9` from the shell. The unit of loss is the boot, so the only lever is how much
rides on one, and boot-to-login here is about twenty seconds. `CHUNK=10` gets batching back
where a boot is expensive; there `quarantine.txt` orders known faulters last so they cost only
the tail of their chunk. Quarantine is ordering, not exclusion. Every run writes its own
`logs/<RUN_ID>/` with the toolchain that built the domains beside the chunk logs, so two
compilers are two tables and not one overwritten.

**What the suite found that no probe would have.** Two compiler defects. C-48: outgoing
stack-passed varargs were packed at 8 bytes while `va_arg` reads at 16, so `inet_ntop` printed
`::1` as `::1:0`, `getmntent_r` spun forever, and five tests took a capability fault in a
`t_error` call rather than printing what was wrong. Fixed, and the before-and-after is 31 PASS
and 7 FAULT against 40 and 3. C-49: the `shrink` that bounds a pointer before a
call reads a size operand holding a code address, and the three writes to that register in the
whole function are one `li` of the size and two `add`s that replace it with a top; `setjmp`
faults on exactly that instruction and is the only source of the 77 that still does. Three runtime defects too, each a case
where returning from a syscall is worse than serving it: `exit()` put musl's `_Exit` in its
retry loop, musl's own `__libc_malloc` bypassed this port's allocator into a `lite_malloc.c`
that stores pointers as integers, and `stat` on a path had no route at all. The first two
were each worth a whole boot: a refused `exit` and a faulting allocator both end the guest. And the narrowing
of M-1: a fault on first entry returns to the guest, a fault after a yield does not.

| verdict | n | |
|---|---:|---|
| `PASS` | 40 | the test's own `t_status`, with nothing it needed refused behind its back |
| `FAIL` | 6 | named below, five of them for a reason that is not the port's |
| `FAULT` | 3 | a capability fault: `setjmp` is C-49, `mbc` and `swprintf` are open |
| `NOBUILD` | 5 | the five `tls_*` sources, all C-47 |
| `EXCLUDED` | 23 | a service a domain does not have: processes, threads, sockets, SysV IPC, dynamic loading |
| **total** | **77** | every source in libc-test's `src/functional` |

Measured 2026-09-17 with the C-48 fix in the compiler. Against the same tree without it the
same run reads 31 PASS, 9 FAIL, 7 FAULT, 2 HUNG and 2 NOTRUN.

**What the reds are.** Of the nine, five are not about capabilities at all, one is a
registered compiler defect, and three are open.

| test | what it is |
|---|---|
| `strptime` | musl 1.2.5 has a case for none of `%F`, `%s`, `%z` and returns 0 for all three |
| `mntent` | musl 1.2.5 leaves the trailing newline in `mnt_opts`; the same `sscanf` line against the host's libc returns the same `n[6]=16 n[7]=25` and the same `"defaults\n"` |
| `fscanf` | reads its input through a pipe, and a domain has no second end for one |
| `utime` | needs `utimensat`; the stat service carries size and mode, not times |
| `sscanf_long` | wants an 8 MiB buffer; a domain is one contiguous kernel allocation and `MAX_ORDER` caps that at 4 MiB here |
| `setjmp` | C-49, the one compiler defect still open |
| `mbc` | faults at `lw a7, 4(a4)` walking a locale table four bytes at a time, cause 5, out of bounds |
| `swprintf` | faults at `cincoffsetimm a2, a0, 4` on a `FILE` field loaded with `ldc` that carries no tag, cause 24 |
| `strtold` | the last bit of a 113-bit mantissa, a soft-float question |

`mbc` and `swprintf` both used to die inside musl's `__simple_malloc`; with one allocator under
all five names that function is not in the image at all any more, and both faults moved further
in, which is progress and not a fix. They are the three worth chasing next, with `strtold`.

The first two are libc-test tracking musl's master while this port builds 1.2.5, so they
fail the same way on any 1.2.5 and are not evidence about capabilities at all.

**Patches to the tests, not to musl.** `patches/` holds the minimal changes that remove a
flat-memory assumption from a test: `search_lsearch` reads 80 bytes out of a two-byte literal,
and the string tests' `aligned()` sends a pointer through `uintptr_t` and back, which on this
target returns an address without a tag. Both are harmless on flat memory and fault here before
the test measures anything; both would fault under CHERI too. The rule in `patches/README.md`:
a patch may change how a test reaches memory, never what it checks. `fetch-libc-test.sh` resets
the tree to the pinned commit and applies them, so a run depends on the commit and the patch set
and on nothing that happened in the tree before.

## Run

```bash
source capstone/tests/capstone-test-env.sh
bash capstone/ports/musl-capstone/survey-musl-capstone.sh          # fetch, prepare, survey
bash capstone/ports/musl-capstone/survey-musl-capstone.sh --list-failures

bash capstone/ports/musl-capstone/build-musl-capstone.sh           # libc-capstone.a
bash capstone/ports/musl-capstone/write-probe/run-write-probe.sh   # each boots QEMU once
bash capstone/ports/musl-capstone/file-probe/run-file-probe.sh
bash capstone/ports/musl-capstone/stdio-probe/run-stdio-probe.sh

bash capstone/ports/musl-capstone/libc-test/run-libc-test.sh      # one boot per test
TESTS="inet_pton mntent" bash capstone/ports/musl-capstone/libc-test/run-libc-test.sh
```

`MUSL_WRITE_PROBE_BADFD=1` adds the write probe's negative control, which reads
`errno` after a refused descriptor and so cannot pass unless the thread pointer
survives the domain boundary. A worktree has the buildroot submodule present and
empty, so point `CAPSTONE_BUILDROOT_DIR` at the main clone before building the
guest side.

Exit codes: `0` pass, `1` regression against the pinned `BASELINE_OK`, `2` the
harness could not measure (unprepared tree, no compiler, empty file list, or a
flipped control).

**The survey has two positive controls**, because a survey that cannot fail is
not evidence: `src/stdlib/abs.c` must compile and `src/string/strlen.c` must
fail. If either flips, the script prints ERROR instead of a number. When the
string routines are replaced, the second control has to be retired on purpose —
that is the intent, not an accident to work around.
