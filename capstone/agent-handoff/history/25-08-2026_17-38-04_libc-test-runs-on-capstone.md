# libc-test builds and runs on Capstone: what works, what it costs

musl compiles: **1355 of 1361**, and the six that do not are structural. The
question is now testing, and musl's own suite is the instrument that exists.

## The suite

`https://repo.or.cz/libc-test.git` is reachable and is musl's own test suite,
same author. 464 .c files: 77 functional, 69 regression, 199 math, 79 api,
plus a small framework in `src/common`.

## api: a compile-only gate, and it is CLEAN

The 79 `api/` tests check declarations only, so they need no QEMU at all.

**78 of 79 compile, and the failing one is identical on riscv64.** That control
matters: run with libc-test's own `-pedantic-errors -Werror`, 78 of 79 FAIL on
both targets, because the suite's C99 style trips a 2026 clang. Dropping those
two flags gives 78/79 on both, with the same file failing --
`_PC_TIMESTAMP_RESOLUTION`, a POSIX 2024 addition musl 1.2.5 does not have.

So the arch overlay's declaration surface is equivalent to upstream riscv64's,
measured rather than assumed. This is the cheapest test we have: no QEMU, no
archive, seconds to run.

## functional + regression: 133 of 146 build as domains

Each test is a `main()` returning `t_status`. A three-line shim bridges it to the
domain entry (`capstone_main`), and one `.dom` per test avoids symbol collisions.

The 13 that do not build, by category, none of them surprising for a
single-threaded static domain:

    5   TLS            "Cannot select: c128 = GlobalTLSAddress"
    4   pthreads       cancellation, flockfile
    2   dynamic load   dlopen_dso, tls_align
    1   setjmp
    1   test-specific

**Soft float from compiler-rt is what unlocked 18 of them.** The link failures
were dominated by `__trunctfsf2` (8), `__multf3` (5), `__eqtf2` (2),
`__extendsftf2`, `__extenddftf2`, `__fixsfdi` -- C-20's runtime half. All 21
builtins tried compile for capstone64 now, archive to 41 KB, and 18 previously
unlinkable tests link against them. This is the concrete answer to the README's
"unusable in compiler-rt as well as musl".

## Running them: one boot, many domains

The existing probes boot once PER DOMAIN. At 133 tests that is over four hours,
which is what CLAUDE.md's batching rule exists to prevent. `run-domain-smoke.py`
takes an arbitrary guest command, so a shell loop over `/mnt/host/*.dom` runs the
whole set from ONE boot.

Demonstrated, first results:

    basename          retval 0     pass
    clocale_mbfuncs   retval 0     pass
    dirname           retval 4294967296  fail
    env               cause 24     FAULT -- and the run ends here

The fault behaves exactly as the batching rule says it will: everything after the
first one is lost. So the harness needs the rule's other half -- order the set so
the expected-to-return ones run first, and accept that the tail after a fault
must be re-run. A resumable runner that records the last completed test and
restarts from the next is the shape this wants.

`dirname`'s 4294967296 is 2^32 with a zero low word, which looks like how the
retval is marshalled rather than what `t_status` held. Worth one look before it
is recorded as a failure.

## Where the pieces live

The build is not yet a committed script: musl tree under
`$CAPSTONE_TMP_ROOT/musl-src`, libc-test beside it, domains in
`$CAPSTONE_TMP_ROOT/libc-test-build/dom`. Making it one is the next step, and it
should carry the api gate (no QEMU) separately from the domain gate (one boot),
because they have very different costs and the cheap one should run far more
often.
