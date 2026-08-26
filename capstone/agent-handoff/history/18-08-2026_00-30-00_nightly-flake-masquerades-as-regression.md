# The nightly reported a codegen regression that does not exist

Run: `nightly-20260817_232443`, core tier, against `capstone-codegen-cap-constants`
(the branch's `llvm/` tree, reached via `nested-allocators` at `d9ff983`).

## Verdict

**No codegen regression. Every failure in this run is boot flakiness.**

| result | suites |
|---|---|
| PASS | lit (60/60), coremark, rv8, revoke-on-free, borrow-cost, tree-cost-O2, hier-revoke, shared-region |
| FAIL(75) | authority, smoke, beebs, revoke-matrix, static-cap-globals |
| TIMEOUT | intra-domain-mrev |
| FAIL(1) | linear-uninit-corpus |

## Why the five FAIL(75) are not results

All five carry the marker the harness itself emits:

    __CAPSTONE_INFRA_FLAKE__ heap_coalesce: boot-login

and all five logs stop mid-boot, at the MEDELEG line or in the SBI messages.
Exit 75 is the harness's infra-flake code. No test ran and failed.

The TIMEOUT is the same thing wearing a different code: `intra-domain-mrev` logs
twelve "no boot/fault ... retrying" events and spends its whole 1800 s budget on
them.

## The one that matters: FAIL(1) is ALSO a flake, and that is a harness defect

`linear-uninit-corpus` exited 1, not 75, which is what a genuine regression looks
like. One probe failed:

    FAIL  uninit_negative_offset_fault  (no fault line after 3 attempts)

It expects a fault and got none, at -O0 only, while the same probe passes at -O1
and -O2. That is a very plausible shape for a codegen bug, and this session had a
specific suspect ready: `getI128NumericValueOrFatal` widened the accepted
constant forms in the load/store and `CIncOffset` displacement paths, and this
probe is about a NEGATIVE OFFSET.

The suspect is innocent. The probe's own log shows the domain never ran:

    domain starts   0
    login reached   0
    fault lines     0

against its passing sibling `uninit_use_before_init_fault` at the same -O0, which
reports 2 fault lines. All three attempts died at boot.

**So the defect is in the reporting, not in the compiler.** When a suite exhausts
its retry budget on an infra flake, it reports FAIL(1), which is indistinguishable
from a real failure. Anyone reading the report table alone would open the compiler
first. The information needed to tell them apart exists only in the per-probe log,
one directory further down.

## What to change, if someone picks this up

Two candidates, both cheap, neither done here:

1. Propagate the flake code. A suite that gives up after N flaked attempts should
   exit 75, not 1, so the nightly's table classifies it correctly.
2. Report the retry count per suite in `report.md`. `intra-domain-mrev` burning
   twelve retries is the single most useful number in this run and it is only
   visible by grepping the log.

## The underlying problem, unaddressed

Roughly half the boots in this run never reached login. Of the nine suites with no
retry logic, four passed and five flaked, so it is not the retry logic that
separates them, it is luck. Resources are not the cause: load 1.31, 223 GB
available, one QEMU at a time, nothing competing.

Until that rate comes down the nightly cannot gate anything. The compiler-side
gates still stand on their own: Capstone lit 60/60 here, and 7952 tests across
Capstone, RISCV, TableGen and X86 on 2026-08-17 with six pre-existing
emutls/tls-android failures that were verified pre-existing by reverting the two
shared-code hunks and rebuilding.

---

## SECOND INSTANCE, 2026-08-18, and it is a different suite

Run `nightly-20260818_184537`, core tier, on `capstone-i128-mul-and-capinit-test`
(the i128 widening-mul fix plus the C-19 test change, on `cb459e7fe2dc`).

Same shape, different suite, which is what makes it a pattern rather than an
accident. `linear-uninit-corpus`, the FAIL(1) of the first instance, PASSED here in
952 s. The FAIL(1) this time is `intra-domain-mrev`, and it is again a boot flake:

    FAIL  held_protected_value_lifecycle  (no fault line after 3 attempts)

The discriminator that worked, and the one from the first instance did NOT: this
suite's per-probe logs contain no `Created domain ID` line at all, so the marker
used last time reports 0 for the passing probes too and separates nothing. What
does separate them is SIZE.

    held_protected_value_lifecycle    3,131 bytes   <- the failure
    held_mem_alias_fault             73,683
    held_ambient_miss                74,059
    held_no_revoke_ok                74,216
    held_unrelated_ok                74,221
    held_revoke_fault                74,268
    held_split_sibling_ok            74,306
    held_arena_survives_revoke       74,308

Its seven siblings all produce 73-74 KB. The failing probe produces 3 KB and stops
just after MEDELEG. It never reached the point where it could fault, so "no fault
line" is not a result about the compiler.

The other five red suites -- authority, coremark, beebs, static-cap-globals,
shared-region -- all exit 75 and all carry `__CAPSTONE_INFRA_FLAKE__`.

So: NINE genuine passes, SIX boot flakes, and no failure attributable to anything
under test. Overall FAIL, and the overall verdict is worthless again.

Worth noting for the flake rate itself: `coremark` passed in 23 s in the first
instance and here burned 488 s in retries before giving up. The boot flakiness is
not stable between runs, so a suite's colour is not comparable across nightlies.

WHAT THIS ADDS TO THE FIX LIST. The first instance proposed propagating the flake
code so an exhausted retry budget exits 75 rather than 1. Still not done, and this
run is the second time it cost a manual investigation. Add one thing to it: the
per-probe check must not key on a marker only some suites emit. Log size against
sibling probes worked here and needs nothing suite-specific.
