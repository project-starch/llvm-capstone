# SQLite's system-allocator cases on CheriBSD at the default quarantine (2026-11-18)

The system-allocator arm of this corpus (`cheribsd-platform`,
`../2026-10-30-cheribsd/non-nested`) runs with `NOMEM5=1`, which puts SQLite on CheriBSD's `malloc`
**and** forces a revocation sweep on every `free` at SQLite's allocator boundary
(`ports/sqlite/cheribsd/repro322_common.h`). That arm answers whether revocation can see the defect
once the release is a `free`: 17 of 17 at the site. The same 17 cases had never been run with
CheriBSD's default asynchronous revocation. This record is that run. It is a separate
configuration and is **not** folded into `cheribsd-platform`'s cells.

## Configuration

- Cases: the 17 temporal cases that can run on the system allocator, 02 03 05 06 10 11 12 13 14 15
  16 17 19 20 21 23 24. Cases 07 and 22 install their own heap and cannot.
- Built `SYSALLOC=1` from this corpus's `case.c` files; run with `NOMEM5=1 NOREVOKE=1`, so the
  allocator is the guest's `malloc` and nothing forces a sweep. Every case prints
  `allocator=system revoke=stock-quarantine`.
- Guest: stock CheriBSD riscv64-purecap under CHERI-QEMU, sysctls read back before the run:
  `security.cheri.runtime_revocation_default=1`,
  `security.cheri.runtime_revocation_every_free_default=0`,
  `security.cheri.runtime_revocation_async=1`.
- Gate: `poscontrol` faulted (exit 162) and its fault resolved, in both runs.
- Runner: `shared/run-cheribsd.py OUT --mode non-nested --stock-quarantine` (the option is new with
  this record). Case timeout 300 s.

## Result

Rejected at the stale operation: **0 of 17**. Released memory held in quarantine at the stale
operation: **17 of 17**.

| outcome | cases | count |
|---|---|---:|
| runs to completion, exit 0 | 02 03 06 10 11 12 13 14 15 16 17 19 21 23 24 | 15 |
| the stale access completes, then a later tag fault | 05 | 1 |
| the stale access completes, then the process is aborted | 20 | 1 |

Two runs, identical verdicts (`run1.log`, `run2.log`).

- **Case 05** takes SIGPROT, `PROT_CHERI_TAG`, at pc `0x19f9b8` (image base `0x100000`), which
  symbolises to `sqlite3DropTriggerPtr`. With the forced sweep the same case faults earlier, in
  `findElementWithHash` (pc `0x16fc5c`), the first read of the freed schema's hash. Here that read
  goes through and the fault is in the caller, on a value the stale read returned: a later use,
  after the stale operation completed.
- **Case 20** dies on SIGABRT ("Abort trap") after printing `before dangling-hash query`, with no
  capability fault. With the forced sweep it faults in `nodeHashLookup` on the dangling node. Here
  the lookup completes; the abort follows when the dangling node is released again.
- The other fifteen complete.

All seventeen fault at a declared site under the forced sweep
(`../2026-10-30-cheribsd/non-nested`, `non-nested-r2-case23`) and none does at the default
quarantine. That difference is the evidence that each freed block was in the quarantine, not yet
swept and not reissued, when the stale access happened. No quarantine probe was run.

## Files

- `non-nested-stock-quarantine.json`: the runner's record of the second run (`--no-build`, the
  binaries of the first run).
- `run1.log`: the first run, which built the binaries and ran all 17. Its record was not written:
  OUT was given as a directory and the runner expects a file. The verdict lines are in the log.
- `run2.log`: the second run, with the record. `<OUT>` stands for the output path.
