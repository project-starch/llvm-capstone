# printf and the allocator move into libc-capstone.a, 2026-08-14

**Result, QEMU, verified.** Formatted output and a working `malloc` family now come out of the
archive, so a program no longer supplies either. Nine `snprintf` cases match, byte for byte, what
glibc produces for the same format strings on the host:

```
PRINTF S1: entered
PRINTF S2: snprintf cases checked
PRINTF S3: allocator checked
PRINTF STDOUT: 42 ok 1.50 <end>
__CAPSTONE_PRINTF_PROBE_PASSED__
__CAPSTONE_HOSTCALL_HOST_DONE__ status=0 serviced=7
```

Harness: `capstone/musl-capstone/printf-probe/run-printf-probe.sh`.

`%.17g` of `0.1` gives `0.10000000000000001` and `%.0f` of `2.5` gives `2`, so the formatter is
correctly rounded to the full 53 bits and does round-half-to-even. Those two cases are the ones
that would fail if the narrowing below had cost precision; the integer cases would not have
noticed.

**The `PRINTF STDOUT:` line is a separate claim from the rest.** Everything else is written
through the raw hostcall that already worked. That line goes through musl's FILE buffering,
`__stdout_write` and `SYS_writev`, and it only reaches the console because `domain_main` flushes on
the way out. It is in the run script's marker list for that reason.

## What was added

| | |
|---|---|
| `libc-ext/gen-vfprintf-double.py` | generates musl's own vfprintf with `long double` narrowed to `double` |
| `libc-ext/malloc.c` | first fit, splitting, forward coalescing, 256 KiB static heap |
| `libc-ext/string.c` + `memcpy.c`, `memmove.c`, `strlen.c` | the 9 `src/string` files musl cannot compile here |
| `libc-ext/locks.c` | `__lock`/`__unlock` as no-ops; the domain is single-threaded |
| 22 compiler-rt soft-float builtins | in the archive rather than in each program's build script |
| `SYS_writev` in `runtime/hostcall.c` | musl's stdio uses writev exclusively; write() alone is never called |
| `__stdio_exit()` in `domain_main` | nothing else flushes, since a domain cannot call `exit()` |

**vfprintf is generated, not vendored, and not reimplemented.** musl's `fmt_fp` expands the
mantissa into `uint32_t` limbs and does base-1e9 long division on them, so it needs no 128-bit
arithmetic at all; the only long-double dependence is the type of the value, one `frexpl`, and
three `LDBL_` constants. Narrowing those keeps musl's algorithm, which matters because a
hand-written dtoa that is subtly wrong produces plausible digits and a passing test. Generating at
build time also keeps `prepare-musl-capstone.sh`'s invariant that the upstream tree is
byte-identical, and the generator fails if any substitution stops matching.

**Two macro substitutions were needed beyond the obvious ones.** `signbit()` and `isfinite()` are
ternary chains over `sizeof(x)` whose last arm is the long-double one. The arm is dead for a
double, but only after folding, and vfprintf is built at `-O0` (see C-22), where clang emits all
three arms and the link then wants `__signbitl`, `__fpclassifyl` and `__extenddftf2`. They are
substituted for `__builtin_signbit` / `__builtin_isfinite`.

## Three defects found on the way, none of them in the new code

**1. The archive was built without `-fno-jump-tables`, which ISSUES.md C-4a records as
mandatory.** 15 members carried absolute-addressed switch tables, `vfscanf`, `strftime` and
`lgamma_r` among them. They compile and archive cleanly and fault only when called, so the survey
— which counts compiles — could never have seen it. Every *application* build here already passed
the flag; only the archive did not. Adding it changes the compile count not at all (1280 before
and after) and is now checked after archiving.

**2. C-22, an integer selected as a `cincoffsetimm` base at `-O1`.** This is what the probe
originally died on, at the first `%f`. Full evidence in ISSUES.md; the short form is that
`fmt_u`'s digit pointer gets split into a capability base and an i128 index, and the index's
decrement is then selected as capability arithmetic (`li a1, 12` followed by
`cincoffsetimm a1, a1, -1`). Worked around by building vfprintf at `-O0`, which has zero such
sites against six at `-O1`, with `libc-ext/scan-cap-base.py` as a build gate. **Reduction failed**
— the obvious small digit loop compiles clean at every level — so there is no lit test for it yet.

**3. C-23, the address of an undefined weak symbol is not null.** The flush was first written as
`if (__stdio_exit) __stdio_exit();` so that a domain never touching stdio would not pull the
machinery. The emitted `auipc`/`addi` pair is pc-relative and therefore non-zero, the branch is
never taken, and the domain calls a symbol that does not exist. It presented as `file-probe`
printing `__CAPSTONE_FILE_PROBE_PASSED__` and *then* taking a capability fault — a verdict that
stopped reading at PASSED would have called it green.

## The checks, and the fact that they fire

`printf-probe` built with `-DPRINTF_PROBE_NEGATIVE_CONTROL` corrupts one expectation and one block
stamp. Run 2026-08-14: all nine format cases reported `want |wrong-...| got |...|`, the allocator
reported `handed out overlapping blocks`, the domain returned 1 and the run script failed. Both
arms of the probe are therefore known to work rather than assumed to.

`scan-cap-base.py --self-test` runs before every real scan, for the same reason.

## Ceilings, stated so they are not discovered later

- **Allocations are not individually bounded.** A block carries the whole heap's bounds and
  nothing is revoked on free. This is a bring-up allocator, not the configuration any security
  number may be measured on; that is `xlang/common/revoke_arena_domain.c`, which needs a
  host-granted linear capability. Bounding here would need the same slot table, because a pointer
  narrowed to its own allocation can no longer reach the header behind it.
- **The heap is 256 KiB of `.bss` fixed at libc build time**, so a program cannot size it. Only
  programs that reference `malloc` pay for it.
- **`exit()` still hangs**: musl's `_Exit` loops on `SYS_exit`, and our `SYS_exit` returns. A
  domain leaves by returning from `capstone_main`.
- **The libc is single-threaded by assumption**, not by check (`locks.c`).

## What this unblocks

`WHY-SHIM.md`'s libc obstacle is now smaller than it was this morning: a program can format
numbers and allocate memory without bringing either. The next items on the same list are `fstat`
and `lseek`, which together are what `fopen`/`fread` need, and a time opcode, which is what stops
Lua and mruby seeding their string hashes.
