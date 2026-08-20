# Validation programme: emulator, libc, toolchain, in that order

## Why an order at all

Each of these is worth doing on its own. The order below is not a ranking by
value, it is a dependency chain, and one link in it was only found on
2026-08-20:

**A test suite whose failures kill the emulator is not a test suite.**

`helper_csshrink` checked all four of its preconditions with `assert()`, so a
domain that got a SHRINK operand wrong ended the QEMU process: no cause, no pc,
the serial log cut off mid-line, and every remaining test in the run lost with
it. One unimplemented `symlink()` in mruby's gem suite took the whole run. Until
a libc bug produces a domain fault with a cause and a pc, measuring anything
larger than a single probe is wasted.

That is why the emulator work comes before the test programme, and why `delin`
is not hardening for its own sake.

## Phase 0: close the debts from 2026-08-20 -- DONE, and the premise was wrong

`link.ld` moved to `capstone-infra` and the whole stack was rebased onto it. The
change puts `.gct` before `.bss` (dropping 19.5% of the mruby image, measured, as
zero padding in the file) and defines the init/fini array markers. It moves
loaded bytes, and this script's own `.bss` note records a layout sensitivity that
has flipped a passing run before, so it was flagged as needing one validation run
each of `jerryscript` and `micropython`.

**THAT WAS THE WRONG PAIR.** Those are precisely the two branches that do NOT use
this linker script: both link gp-free, through
`tests/runtime-qemu/gp-free-domain/link-gpfree.ld`
(`build-jerryscript-silicon.sh:166`, `build-micropython-silicon.sh:290`). The
change cannot reach them.

The real consumers are the other 86 scripts that reference
`my_first_domain/link.ld`, and they were already exercised on 2026-08-20 AFTER
the change:

- `tests/runtime-qemu/build-domain.sh:13` uses it, and built all ten
  mrev-codegen probes: 9 PASS plus one rc=75 boot flake that passed on rerun,
  each asserting a specific fault cause.
- `musl-capstone/mruby-probe/build-mruby-probe.sh:23` uses it: the mruby probe
  passed S1-S5, and the core mrbtest suite ran 678 assertions.
- `musl-capstone/yield-probe/build-yield-probe.sh:17` uses it, and links.

That is more coverage than the two runs originally asked for, and it is on the
consumers that actually exist. No further action.

**SEPARATE FINDING, not caused by any of this:** `micropython` does not build.
Pass 1 of its gp-free link ends with two undefined symbols,
`mp_type_bytearray` and `mp_obj_new_bytearray_by_ref`, i.e. `uctypes` is compiled
in without the bytearray object type. Proven pre-existing rather than assumed:
relinking the same objects with the pre-rebase `link.ld` and the new one gives an
identical result. Left for whoever owns that branch.

## Phase 1: make failures survivable and visible

- [ ] `helper_csdelin`: convert the two asserts, per spec 24 and 26.
      **The NONLIN early return must NOT change.** The spec (cap-man-insn.adoc
      384-386) says DELIN on a non-linear capability raises `Unexpected
      capability type (26)`; the implementation makes it a silent no-op, and
      9,688 of the 9,690 `delin` in the mruby image follow a
      `cincoffset ..., gp, ...`, i.e. operate on a gp-derived NONLIN capability.
      Making the spec literal would fault every domain at its first global.
      **This is a question for the spec owner, not a fix.**
- [ ] `helper_csscc` (2 asserts, 3 sites in the image) and the 5 remaining type
      asserts in `helper_cslcc` (49,109 sites), each read from the spec.
- [ ] Turn OFF `gp` fabrication in our runs. QEMU currently FABRICATES gp when an
      untagged x3 reaches `cincoffset` under PRV_C. The other line's own comment
      says "the RTL never establishes gp this way", and the toggle
      (`CAPSTONE_GP_FABRICATE=0`) already exists there. Default is on, so an
      entire class of tag-loss bugs is silently repaired by the emulator and
      appears only on silicon. This costs a flag.

Not reachable from our workload and therefore not urgent: `shrinkto`,
`tighten`, `split`, `init`, `seal`, `drop`, `mrev`, `revoke` are emitted zero
times by our compiler.

## Phase 2: unblock mruby's full suite

`_Exit` is `__syscall(SYS_exit_group, ec); for (;;) __syscall(SYS_exit, ec);`
and our hostcall returns 0 for both, so the loop spins. `hostcall.c` says "musl
copes: exit_group falling through to the domain return is exactly what a domain
does anyway" -- true while nothing called `exit()`, and mruby's driver falsifies
it.

A non-returning exit needs the glue's `domreturn` tail reachable from arbitrary
depth. The parts exist: `__capstone_yield` already does the register save and
`domreturn`, and the tail after `domain_main` is the shape to mirror, with the
sealed-return capability taken from `__capstone_dom_ret` rather than from the
glue's frame.

Expected result: the gem suite runs to a report instead of stopping at 445
assertions.

## Phase 3: the cheap compiler wins

Detail belongs in `plans/backend-compiler-fixes.md`; this is the ordering.

- [ ] **`long double` = `double` on capstone64.** Verified:
      `__LDBL_MANT_DIG__ 113`, `__SIZEOF_LONG_DOUBLE__ 16`, and the 128-bit
      integer type IS the capability carrier, so long-double bit manipulation
      collides head-on with capability lowering. Closes **30 of the 40** musl
      failures (27 "cannot lower a 128-bit right shift", 2 "cannot materialize
      >64-bit constant", 1 i128 select). Also makes
      `libc-ext/gen-vfprintf-double.py` redundant: that generator exists
      solely to narrow long double by hand, so the decision is already taken in
      practice and merely absent from the target.
- [ ] **i128 constant operands in the logical lowering.** `~x` becomes
      `xor X, i128 -1`, and `lowerScalarI128Logical` bails because a constant is
      not a recognised extension. Treat an i128 constant equal to the sign- or
      zero-extension of its low 64 bits as that extension. Closes `qsort.c` and
      `dn_comp.c` and removes the `-O0` rescue added on 2026-08-20.
- [ ] Two clang assertions, one file each: `dcngettext.c` ("Illegal ZExt") and
      `if_nameindex.c` ("Cannot create binary operator with two operands of
      differing type").

Not a compiler fix: the 6 mallocng files. musl's allocator computes a table size
from `sizeof(void*)` and underflows on 16-byte pointers. That is a design
assumption of mallocng and this port uses its own allocator.

After Phase 3 the survey should read about 1353 of 1361.

## Phase 4: the test programme

musl ships NO tests. Verified against the live tree
(`git.musl-libc.org/cgit/musl/tree/`): `arch compat crt dist include ldso src
tools`, no `test`. The suite is the separate `libc-test`, from the musl project,
actively maintained (last commit 2026-07-26, 464 test files).

It is NOT a safety net we inherit. Nothing gates a musl commit on it, and more
to the point, musl's review cannot cover our failure mode: all three bugs found
on 2026-08-20 were in correct, review-passed musl code that is right on every
other target.

Measured runnable subset, by scanning every test for facilities a domain lacks
(fork, threads, signals, mmap, sockets, clocks, filesystem):

| | files | no OS need |
|---|---|---|
| `src/math` | 199 | **199** (61 of them long double) |
| `src/functional` | 77 | 45 |
| `src/regression` | 69 | 44 |
| `src/api` | 79 | 78 (compile-only) |

- [ ] Stage 0: `src/api` compiled with our clang on the host. 78 files, no boot,
      checks that our arch overlay declares things correctly.
- [ ] Stage 1: the 138 non-long-double math tests in one domain. No syscalls;
      exercises soft float and `capstone-math-double.h`.
- [ ] Stage 2: the 89 functional/regression tests with no OS need. This is the
      first real test our hand-written `strlen`, `string.c`, `memcpy` and
      `memmove` have ever had.
- [ ] Stage 3: report the remainder explicitly as "not applicable, needs X", so
      the number is never mistaken for a pass rate.

Mechanics: 142 of the 146 functional tests carry their own `main()` and report
through one `t_error`/`t_printf` plus a global `t_status`. Compile each with
`-Dmain=test_<name>` and dispatch from a driver -- exactly the pattern built for
`mrbtest` on 2026-08-20.

**WHAT THIS CANNOT DO.** libc-test checks VALUES, not TAGS. A `memcpy` that
strips tags passes every memcpy test in it. It closes the value dimension and
leaves provenance to the static scanners below.

## Phase 5: the structural toolchain work

- [ ] **A diagnostic on `(uintptr_t)p`.** `ref/capstone-purecap-pointer-model.md`
      and the cast policy in `backend-compiler-fixes.md` already SAY that
      pointer-to-integer casts do not preserve provenance. The policy is not
      mechanically checkable, so third-party code violates it silently: musl has
      106 such casts across 52 files, and three of them cost a day each. This
      does not change any type; it turns a latent stream of runtime faults into a
      list worked through once.
- [ ] **Operand-role classification.** `scan-cap-base.py` flags 22 sites across
      11 files where `-Os` leaves an integer in a capability base position; at
      `-O0` it flags none. Fixing it makes `-Os` usable, which HALVES the image
      (mruby core measured 1,115,566 -> 506,939) and retires three `-O0`
      workarounds (`vfprintf`, `cap-copy.c`, `qsort.c`). Image size is what
      forced the libc heap out of the image, so this is the constraint behind
      the current shape of things.
- [ ] Then reopen capability-wide `uintptr_t`. Today `__SIZEOF_POINTER__` is 16
      while `__UINTPTR_TYPE__` is 8 bytes, and `Capstone.h:240` calls it what it
      is: "(Workaround for Clang consistency check)", because
      `TargetInfo::IntType` has no 128-bit member. That contradiction is the bug
      factory.

## Standing rule, not a phase

Every detector gets its negative control the day it is written. On 2026-08-20
that rule caught three instrument errors in one session: an assert counter that
matched comment text, an archive-wide `objdump` that silently produced nothing
for some members, and a probe selector that scored never-built probes as
failures. Each would have carried a false statement.

## Decisions that are not the lane's to take

| | |
|---|---|
| QEMU submodule merge | `capstone-bootstrap` (LCC totality, gp toggles) and `capstone-qemu` (CINCOFFSET raise, SHRINK raise, probe) have diverged; `merge-tree` conflicts in `op_helper.c`, six hunks a side, both in exception semantics, and the other line is in-flight work |
| `long double` at 64 bits | an ABI decision for the target, though it only formalises what the build already does by hand |
| `delin` on NONLIN | spec says fault, implementation says no-op, 9,688 sites depend on the no-op |
| `caplifive-buildroot` | push refused with 403; four commits stranded, including the two-region module without which mruby does not start here |
