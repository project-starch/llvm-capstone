# cheri-baseline-xlang — CHERI security baseline for the xlang corpus

The empirical **CHERI column** for the xlang cross-language FFI corpus,
produced the same way `capstone/tests/cheri-baseline/` produced it for the sqlite corpus:
compile each row's lifecycle shim CHERI-RISC-V **purecap**, run it on CheriBSD
under CHERI-QEMU in three revocation configs, and classify **BLOCKED-SYNC /
BLOCKED-SWEEP / MISS**.

**Results: `RESULTS.md`.** Read that first — it carries the verdict table, the
toolchain versions, and the caveats that must travel with the numbers.
**Method definitions:** `capstone/tests/cheri-baseline/README.md` (configs, taxonomy, `-O0`
rationale) apply here unchanged, so the two columns are comparable.
**Plan and design decisions:** `agent-handoff/plans/cheri-baseline-xlang.md`.

## Status: complete

- **14 of the corpus's 15 rows measured**, three configs each, on real
  CheriBSD purecap. Row 7 is dropped — its defect does not exist (`RESULTS.md`
  §Scope).
- **All 14 predictions confirmed.** They were committed to `rows.tsv` before
  each boot, so this is a test rather than a description.
- **Reproduced from an empty `CHERI_ROOT`**: vehicle rebuilt from scratch and
  the measurement re-run, all 14 verdicts byte-identical.

## Run it

```bash
cd xlang/cheri
./check_shim_fidelity.py                                        # 8s, no CHERI stack needed
../../capstone/tests/cheri-baseline/provision-cheri-vehicle.sh  # build the vehicle (~40 GB)
./run-cheri-baseline-xlang.sh                                   # gate → compile → boot → classify
```

Start with `check_shim_fidelity.py`: it needs only clang, and if it fails
nothing downstream is worth running.

**What the pieces are.** `xlang/repro/<key>/` reproduces each defect in the *real*
software. The results below do not use it: purecap mruby now runs (`mruby-port/`),
but a trigger exiting 0 is not yet a verdict, so each defect is instead
re-expressed as a **shim**: a
small C program performing the same allocate / free / offending-access events
with the same geometry. `check_shim_fidelity.py` proves the shim matches the
real reproduction; the CHERI run then measures the shim. The shim does not
trigger the CVE and never runs the real code: it is an equivalence of **memory
events**, which is all a capability system can observe.

**Before trusting any row:** `sanity_mock` must be rc=0 (else the harness
itself is faulting, not the defect), and `REVOKE_ENABLED` must read 0/1/1
across the three passes (else a config did not take). Both are printed by the
runner.

## Files

| File | Role |
|---|---|
| `RESULTS.md` | the verdict table, config reality, and the upper-bound caveat |
| `rows.tsv` | per defect: defect class, allocator route, geometry, pinned trigger parameters, and the **predicted** verdict per config |
| `check_shim_fidelity.py` | **the gate** — proves each shim reproduces the same memory-safety event as the real defect, before any CHERI verdict is trusted. x86 + ASan, no CHERI needed |
| `mock-mruby/` | minimal lifecycle harness the shims link against; plain `malloc`/`free` by design |
| `shims/<defect>.c` | one shim per defect, named for what it is, distilled from `xlang/repro/<key>/target.md` + `asan.txt` |
| `shims/vm_stack_uaf.h` | shared template for the six defects that are one shape |
| `compile-purecap.sh` | builds one purecap ELF per defect + the sanity probe |
| `run-in-guest.sh` | runs inside CheriBSD: sets each config's sysctls, emits one result line per defect |
| `run-cheri-baseline-xlang.sh` | end-to-end: gate → compile → provision → boot → classify |
| `mruby-port/` | purecap mruby: builds with three one-line ABI changes, faults at runtime; where to resume |

## Two things to know before quoting the numbers

**On naming:** each measured thing is a *defect* — a reproduced upstream CVE
or issue — and is named for what it is (`mruby_vmstack_uaf`). The corpus/paper
table number survives only as a join key (`rows.tsv` column 1, `xlang/repro/<key>/`,
and the `ROW <key>` lines the shared `classify.py` parses).

1. **They are an upper bound for CHERI.** The shims allocate with plain
   `malloc`/`free`, so every object death is visible to revocation. Real FFI
   runtimes hide frees in pools, arenas and lookaside allocators. The worst
   case was deliberately **not simulated** — see `RESULTS.md`.
2. **Row 1 duplicates the allocator-visible shape by construction.** It is
   included for completeness of the corpus table, not as independent evidence.
