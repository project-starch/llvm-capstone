# Temporal-safety overhead — Capstone side (revoke-cost microbenchmark)

**First Capstone-QEMU data point for the PI's CHERI-vs-Capstone performance
comparison** (`plans/perf-cheri-vs-capstone-qemu.md`; PI 2026-07-14: "test the
perf. distinction on QEMU-Capstone and QEMU-cheri"). This is the **Capstone
half**; the CHERI half runs on the task-015 CHERI-QEMU stack and is not yet done.

**Vehicle:** `capstone-qemu` functional model, `-icount shift=0,sleep=off`,
counting via `rdcycle` (one-per-retired-instruction under `-icount`; see below).
**Deterministic** — the numbers below are exact and reproduce run-to-run.

## Workload

A `malloc(64)` → touch one byte → `free()` loop, `ITERS = 512`, run under three
allocator configs (one domain build each, `-DROF_COST_MODE`), each bracketed by
the instruction-count read. Per-op cost = `(allocfree_total − empty_total)/ITERS`.
The empty calibration loop is identical across configs (total 1540 every run), so
the deltas are purely the allocator.

## Result (functional-model instruction-count proxy — NOT silicon timing)

| Config | raw total | per-op (instr) | what it includes |
|--------|-----------|----------------|------------------|
| **bump** (baseline) | 5128 | **7.0** | broad NONLIN heap cap, interior pointers, `free` no-op — no per-object caps |
| **norevoke** (alloc-side) | 32259 | **60.0** | revoke-on-free allocator with the revoke suppressed: SPLIT+mrev+delin per malloc + slot table + `rof_find`, but no revoke |
| **revoke** (full) | 34819 | **65.0** | full revoke-on-free: alloc-side + REVOKE per free |

### Overhead breakdown (the PI's "where does the cost come from")

```
alloc-side overhead (norevoke − bump) : +53 instr/op   (8.56x)   -- make each allocation revocable
revoke overhead     (revoke − norevoke): +5 instr/op              -- the free-time revoke itself
total temporal cost (revoke − bump)    : +58 instr/op   (9.28x over baseline)
```

## Reading

- **Revoke-at-free is a cheap O(1) op: +5 instr/op.** Consistent with the
  borrow-cost probe (a full borrow = mrev+delin+access+revoke = +4 over a raw
  read). Capstone's temporal-safety *primitive* is nearly free at the point of
  use — this is the contrast with CHERI, which pays its temporal cost in a
  quarantine sweep, not at the free.
- **The cost of temporal safety here is dominated by making each allocation
  independently revocable** (+53 instr/op): one SPLIT + mrev + delin per malloc,
  plus this allocator's slot-table bookkeeping. That is a property of *this naive
  non-coalescing allocator* (`revoke_on_free_alloc.h`), not of the revoke
  mechanism. A production allocator would amortise much of it; the honest first
  number does not.
- **The 9.28× ratio is inflated by how lean the bump baseline is** (7 instr/op).
  The absolute **+58 instr/op** is the more transferable figure; quote the ratio
  only alongside the baseline.

## Caveats / honesty

- **Proxy, not timing.** QEMU has no pipeline/cache/cycle model. This is a
  dynamic instruction count. The cycle-accurate number is the FPGA follow-on
  (`tests/rtl-smoke/`).
- **`rdcycle`, not `csrdicount`.** The checked-in `capstone-qemu` binary
  (2026-07-10, built from `dd97f994`) predates the `csrdicount` op
  (`fb3217d1`, 2026-07-13), so csrdicount is illegal in it (cause 2). `rdcycle`
  under `-icount` gives the same instruction count (verified in the rtl-smoke run:
  raw 2 / borrow 6, identical to task-014's csrdicount) and is also what the FPGA
  port reads. To use csrdicount instead, rebuild `qemu-system-riscv64` from the
  current submodule HEAD.
- **Naive allocator.** `revoke_on_free_alloc.h` never coalesces (one-way SPLIT),
  so the arena depletes at `ITERS×BLOCK`; `ITERS=512`, `BLOCK=64` ⇒ 32 KiB ≪ the
  256 KiB arena. This is the Phase-0 allocator, not the #78 production allocator.

## Next

- **CHERI half** (task-015 stack): same workload, `CHERI_CAPREVOKE` off / async /
  eager, instruction-count readout on `qemu-system-riscv64cheri`. Then the
  side-by-side table fills `paper/evaluation.tex` §`sec:eval-perf-compare`.
- **Applied case:** the SQLite workload (plan candidate #2), if the CHERI-side
  SQLite harness is cheap.
