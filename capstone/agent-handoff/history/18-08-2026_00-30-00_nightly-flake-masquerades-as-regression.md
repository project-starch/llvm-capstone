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
