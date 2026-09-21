<!-- The operating manual for the PoisonCap target. The corpus README
describes the corpus; this describes how to run it here. -->

# Running the corpus on PoisonCap

The same `case.c` builds as an ordinary CheriBSD purecap program against the
port's [PoisonCap build](../../../../ports/memcached/allocators/host/cheribsd/poisoncap/README.md):
one `mmap`'d arena with poison authority, the same ledger the Capstone domain
runs, and every alias handed to memcached bounded to its unit and stripped of
the manager's permissions. Both arms come from one binary and differ only in
the mode argument.

    source capstone/tests/capstone-test-env.sh
    export CHERI_SDK=/tmp/capstone/poisoncap-work/sdk
    export CHERI_SYSROOT=/tmp/capstone/poisoncap-work/output/rootfs-riscv64-purecap
    BUILD=/tmp/capstone/memcached-allocator-repros/poisoncap
    bash capstone/bug-corpora/memcached/allocator-repros/shared/build-cases.sh poisoncap "$BUILD"

    python3 capstone/bug-corpora/memcached/allocator-repros/runners/poisoncap/run-defects.py \
      "$BUILD" /tmp/capstone/memcached-poisoncap-1 \
      --sdk "$CHERI_SDK" --rootfs "$CHERI_SYSROOT" \
      --image /tmp/capstone/poisoncap-work/output/cheribsd-riscv64-purecap.img \
      --disable-default-revocation

and the control that makes either arm mean something, which must exit 0:

    python3 .../runners/poisoncap/run-defects.py "$BUILD" /tmp/capstone/memcached-poisoncap-control \
      --sdk ... --rootfs ... --image ... --disable-default-revocation --negative-control

`--cases` takes a diagnostic subset and `--modes spatial,protected` (or `0,1`)
a single arm.

## Two modes, one binary

| mode | what a release does | required outcome |
|---|---|---|
| `spatial` (0) | nothing: bounded leases, no poison, no sweep | the sequence COMPLETES, and the adapter's own line reports `sweeps=0` |
| `protected` (1) | `cpoison` per granule, one synchronous sweep, `cclearpoison`, `memset` | the stale access FAULTS at the labelled probe |

The spatial arm's `sweeps=0` is part of the oracle, not decoration. A control
that swept would be a second protected arm wearing the control's name, and
nothing in the pair would then be a control.

## What is deliberately not accepted as protection

Exit 162 is the status of EVERY `SIGPROT` on this platform, so on its own it
says nothing. The supervisor reports what the KERNEL saw -- signal, `si_code`
and faulting PC from `PT_LWPINFO` and `PT_GETCAPREGS`, and the expected
address resolved from the child's own memory map and ELF -- and a protected
arm passes only on a complete fault line whose signal is 34, whose `si_code`
is 2 (`PROT_CHERI_TAG`) and whose faulting address and PC are both the
supervisor's resolved `mc_defect_read`. The program under test prints nothing
about itself and judges nothing.

`si_code` is load-bearing, and the platform's own controls are what make it
so. An access to a granule that is poisoned but not yet swept faults with
`si_code=3` while the capability is still tagged; an access through an alias
the sweep has revoked faults with `si_code=2`. A protected case arm must be
the second. (The stock header names 3 `PROT_CHERI_SEALED`; this platform's
kernel reports the poison fault with it, which is why the runner writes the
number and not that name.)

## The platform's controls, in the same boot, before any case

| control | requires |
|---|---|
| `poisoncap-live` | a fresh lease reads and writes |
| `poisoncap-reuse` | poison, sweep, `cclearpoison`, and the storage is usable again; the old alias lost its tag and an unrelated sibling did not |
| `poisoncap-read`, `poisoncap-write` | an access to a poisoned granule is refused: `SIGPROT si_code=3` |
| `poisoncap-reused-read` | an access through the revoked old alias is refused: `SIGPROT si_code=2` |

They run under the same supervisor as the cases and take no fixture, so they
are required in the negative control run too: nothing done to a fixture can
excuse one of them failing. If any does not behave, the run exits 75 and
records no case verdict. The shared ABI and bounds controls run first, as
everywhere else.

## Negative control

`--negative-control` corrupts every fixture so the case's `CHECK(700)` refuses
it before any cache is created, runs every selected arm anyway, and exits 0
only when every oracle reports a failure with no fault and no completed
report.

## What this does not do

It does not protect the process: libc's own revocation is off here, for the
published platform's documented VM-locking workaround, and only the adapter's
explicit sweeps run. It does not cover the page mover, which the port does not
build. And it measures the two cases this corpus has, both in consumers of
`cache.c`; the corpus README says why there is no slabs case.
