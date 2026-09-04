# Can musl be compiled pure-cap? Survey, 2026-08-14

**Question.** A domain has no libc, so every workload so far has grown its own
subset (SQLite: `adapted/capstone_sqlite_libc.c` + `beebs_freestanding_string.c`
+ VFS stubs). If a real POSIX libc compiled for `capstone64-unknown-elf`, the
syscall layer becomes the one place the domain boundary lives, and any POSIX
program becomes a candidate workload instead of only self-contained
amalgamations. Does musl compile?

**Answer. Yes, 93.3 % of it, and the residue is four named groups, only one of
which is new work.**

## Method

Per-file compile only, no linking, no build system: musl's `configure` cannot
probe a target it cannot link for, and the question is about the compiler, not
the build. Headers generated exactly as `Makefile:98-106` does. 1361 sources
surveyed; 169 foreign-architecture sources (`src/**/x86_64/`, `aarch64/`, ...)
excluded and counted separately, since musl's own build would not compile those
for this target either.

Harness: `capstone/musl-capstone/survey-musl-capstone.{sh,py}`. musl 1.2.5, not
vendored, SHA-256 pinned.

## Three arms, each differing in exactly one thing

| arm | compiled | | delta |
|---|---:|---:|---|
| A `arch/riscv64`, `+m` only | 1208 / 1361 | 88.8 % | |
| B A + `-target-feature +a` | 1243 / 1361 | 91.3 % | +35 |
| C B + `arch/capstone64` | **1270 / 1361** | **93.3 %** | **+27** |

**Arm B is a measurement artifact, not a result.** The 35 files failed on
`instruction requires the following: 'Zalrsc'` because musl's riscv64 atomics
are `lr.d`/`sc.d` and the flag was missing from the invocation. Worth recording
because the first survey reported them as a porting cost.

**Arm C is the finding.** One file, `arch-capstone64/syscall_arch.h`, changes
two things: `syscall_arg_t` becomes capability-width instead of `long`, and
`ecall` becomes an extern `__capstone_hostcall()`. The first is what buys the 27
files. Upstream marshals every syscall argument through `((long)(X))`
(`src/internal/syscall.h:22`); under pure-cap that destroys the capability and
the backend then refuses to rebuild one from an integer (`Capstone PureCap:
Cannot materialize arbitrary >64-bit constants ... capabilities are
unforgeable`). musl anticipates the override — the definition is guarded by
`#ifndef __scc` — so this is a supported arch-port hook, not a patch against
upstream.

## Rejected arm

A fourth arm reached **96.1 %** by setting `LDBL_MANT_DIG` to 53 in
`bits/float.h`, which turns every `src/math/*l.c` into a wrapper around
`double`. **It is unsound and the number must not be quoted.** The compiler
still has `long double` at 128 bits and rejects `-mlong-double-64` for this
target, so the header is lying. musl catches it itself:
`src/stdio/vfprintf.c` fails with `'compiler_defines_long_double_incorrectly'
declared as an array with a negative size`. The 39 long-double files are counted
as unresolved.

## What is left (91 files, by directory, sums to 91)

| files | where | fix |
|---:|---|---|
| 39 | `math` 31, `complex` 8 | **`long double` is 128-bit and so is a capability.** Backend work: `-mlong-double-64`, or separate integer-i128 from capability-i128. The only new item. |
| 33 | `thread` 8, `locale` 7, `network` 5, `aio`/`mman`/`regex`/`signal` 2 each, `exit`/`ldso`/`stdio`/`stdlib`/`multibyte` 1 each | stub to `ENOSYS` for the first milestone |
| 10 | `malloc` 3, `mallocng` 6, `oldmalloc` 1 | static asserts over `sizeof(void*)`; replace with an arena, as memsys5 does today |
| 9 | `string` | `(uintptr_t)s % ALIGN` word-at-a-time; replace as SQLite already does |

Not covered by the survey, because it only compiles `.c`:
`src/thread/__syscall_cp.c` needs a `__syscall_cp_asm` for this target. A
single-threaded port forwards it to `__syscall`.

## The 91, as the compiler labels them

```
 27  backend assert: VT.isVector() && "Unable to legalize non-vector shift"
 25  backend assert: getActiveBits() <= 64 && "Too many bits for uint64_t"
 15  backend: Cannot select (i128 integer op on a capability)
 10  backend assert: getSignificantBits() <= 64 && "Too many bits for int64_t"
  8  backend: cannot materialize >64-bit constant (pointer via integer)
  4  static assert: sizeof(void*) assumption (mallocng)
  1  backend assert: castIsValid(getOpcode(), S, Ty) && "Illegal ZExt"
  1  backend assert: S1->getType() == S2->getType() && "Cannot create binary operator ..."
```

87 of 91 are the backend meeting a 128-bit value it cannot treat as an integer.
That is one defect family with two sources — `long double` and a pointer cast to
an integer — and it is the same family that produced SQLite's nine capability
gaps.

## Limits of this instrument

- Compilation only. A file that compiles may still be wrong at runtime; nothing
  here has been linked or executed.
- The survey has two positive controls (`src/stdlib/abs.c` must compile,
  `src/string/strlen.c` must fail) and exits 2 rather than printing a number if
  either flips, because the first version of this survey mislabelled 62 backend
  crashes as one undifferentiated bucket — the captured error line was truncated
  before the assertion text.
- `BASELINE_OK` is pinned at 1270; the check fails on a regression below it.
