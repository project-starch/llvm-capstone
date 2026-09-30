# Bare-metal tests for the Stage-1 mapping tables on capstone-qemu

These tests drive the emulator without a Linux guest: `run.sh` assembles each
`tests/*.S` with the system clang (`.insn` encodings from `capstone.h`), links
it at 0x80000000, runs `qemu-system-riscv64 -M virt -smp 1 -bios none -kernel`
and reads the verdict from virt's test device. A test declares
`// EXPECT: <exit code>`: 0 for a pass, `0x40 + cause` for the capability
fault it expects, `0x3f` when a fault hit an instruction other than the one
named with `EXPECT_FAULT_AT`, and small codes for test-defined failures.

```bash
capstone/tests/mapping-qemu/run.sh                 # the repo's emulator build
QEMU=/path/to/qemu-system-riscv64 capstone/tests/mapping-qemu/run.sh
capstone/tests/mapping-qemu/run.sh tests/create-deliver.S
```

`TEST_PROLOGUE` enters capability mode, keeps the low genesis capability in
`s10` with its cursor on the test device, installs an execute-only trap vector
that reports `0x40 + mcause` after checking `cepc` against the named site, and
zeroes `s11`. Tests never touch `s10` or `s11`. Domains are sealed contexts
built by `MAKE_CONTEXT`; the monitor side calls them, and a domain returns with
`DOM_RETURN(label)` so that the next call resumes at `label`.

## What each test pins

The design is [caplified mapping tables](../../docs/design/caplified-mapping-tables.md)
with the [encoding decision](../../docs/design/caplified-mapping-encoding-decision.md);
the emulator work follows [the M1 plan](../../docs/plans/mapping-qemu-stage1.md).
Rows refer to the design's §10.2 table; scenarios to the
[executable model](../mapping-model/README.md).

| Test | Pins | §10.2 row / model scenario |
|---|---|---|
| smoke-store-load, trap-store-oob | the harness: a round trip and a bounds fault | |
| perm-store-ro, perm-load-wo, perm-store-rw | permission checks on the access path (cause 27), control | pre-existing gap closed in commit 1 |
| type-rev-load, type-rev-control | a REV handle carries no data authority (cause 26), control | I3 |
| harness-wrong-site | a fault outside the named site exits 0x3f | harness positive control |
| genesis-end-physical, mint-above-physical, mint-physical-top | genesis and minting confined to `[0, 2^56)` | E2 |
| lcc-binding-physical | `lcc 8` reads 0 on physical, split and revocation capabilities | E3, E8 |
| create-deliver | CREATE delivers into the named register of a called context; REV handle; root consumed | `check_protected_delivery` |
| create-not-sealed, create-slot-occupied | no monitor destination; delivery exactly once | CREATE with a raw destination / occupied slot |
| create-root-ro, create-root-not-page | root page must be one writable aligned page | |
| create-below-region, create-overlap, create-id-in-use | partition, global disjointness, id in use | `check_address_geometry` |
| create-handle-revoke | REVOKE refuses the detach handle | REVOKE on the detach handle |
| populate-store-load, populate-scattered | translation; three scattered pages as one interval | three scattered pages |
| populate-zeroed | supplier bytes and a stored capability are gone | POPULATE with supplier bytes and tags; `check_anonymous_initialization` |
| populate-absent-page | a PTE that is none faults, no allocation on access | §1.1 |
| populate-frame-nonlinear, populate-frame-ro | PRIVATE takes exclusive writable frames | POPULATE with a non-linear frame |
| populate-twice | a used PTE is never replaced | replace a table entry under a live mapping |
| populate-no-table-page | all or nothing when the path needs a table page | POPULATE whose path needs a table page |
| populate-cross-page | a page-straddling access is refused in this prototype | |
| populate-amo, populate-fp | atomics and FP accesses take the same translated path | |
| populate-ro-mapping-store, populate-ro-mapping-load | max protection is the ceiling; control | |
| detach-faults, delin-then-detach | DETACH kills every logical capability; DELIN first yields the token only | DELIN the mapping capability, then DETACH |
| detach-twice, unmap-before-detach, unmap-twice, destroy-without-detach | state machine refusals | UNMAP without the token or twice; `check_reclamation` |
| unmap-uninit-read, unmap-scrub-then-read | the returned frame is UNINIT until written | read an unmapped frame before writing it |
| destroy-then-create | a freed id and range come back with the next generation | CREATE twice for one id, then DESTROY the first |
| frame-revoke-locks | a frame revoked from above returns UNINIT, its page faults, the neighbour lives | revoke the frame handle of a mapped frame |
| table-revoke-subtree | a leaf table revoked after a warm cache makes its subtree fault | revoke a table page with a warm TLB |

Not testable at one vCPU, as the plan records: the §8 window, foreign issue
during a barrier, and every two-hart row. Not covered yet: the fault-in-domain
exit path (a domain with an invalid trap vector exits the emulator with 0),
the registry's generation exhaustion at 2^20-1, and the second-level limit of
256 MiB per mapping.
