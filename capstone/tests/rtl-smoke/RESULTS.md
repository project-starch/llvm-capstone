# RTL/FPGA borrow-cost port — QEMU plumbing-validation results

**Status: plumbing VALIDATED under QEMU + Capstone (2026-07-14). Not yet run on
hardware.** This de-risks the port before an FPGA slot: the two-region result
hand-off, the `rdcycle` read inside a domain, and the controller read-back are
all confirmed working on the functional model. Two real defects were caught here
that would otherwise have wasted a hardware session.

## What was run

`build-borrow-cost-fpga.sh` (domain `-O2`, Capstone clang; controller buildroot
gcc) → one QEMU boot via `runtime-qemu/run-domain-smoke.py` with
`-icount shift=0,sleep=off`, guest runs
`borrow_cost_fpga.user borrow_cost_fpga.dom`. Serial log:
`$CAPSTONE_TMP_ROOT/rtl-smoke-qemu.log`.

## Result (functional-model proxy — same vehicle as task-014, NOT silicon)

```
borrow-cost-fpga: created arena region ID = 10, results region ID = 12
borrow-cost-fpga: arena transferred (LIN, RW); results shared (host retains)
borrow-cost-fpga: call retval = 0x22380000
borrow-cost-fpga: RAW iters=1024 empty=2052 raw=4105 borrow=8198 copy256=36867 copy1024=135171
borrow-cost-fpga: RESULT cycles/op  raw=2  borrow=6  copy@256B=33  copy@1024B=129
borrow-cost-fpga: RESULT vs-raw     borrow=3.00x  copy@256B=16.50x  copy@1024B=64.50x
```

**Cross-check against the task-014 instruction-count proxy (raw 2 / borrow 6 /
copy 34@256B / 130@1024B): identical shape and identical raw/borrow/+4.** The
copy figures differ by one (33/129 vs 34/130) purely from integer-floor of
`(total-empty)/iters` in the controller vs the probe's rounding — not material.
This confirms the port measures the *same* code the QEMU proxy does; on hardware
these become real cycles (with a pipeline/cache model behind them).

Under QEMU `-icount`, `rdcycle` advances one-per-retired-instruction, so the
"cycles" here equal instruction counts. That is expected — QEMU has no pipeline
model. The number's *value* is not the deliverable; the *plumbing* is.

## Findings (both would have bitten on hardware)

### 1. Both `mcycle` and `rdcycle` are readable inside a Capstone domain (open item 1 — resolved for QEMU)

The domain executed **`csrr mcycle`** (0xB00, the M-mode counter — now the
default in `fpga_instrument.h`) with **no fault**, and separately `rdcycle`
(0xC00 `cycle`) with no fault. Both give identical results under `-icount`
(each advances one-per-retired-instruction). This resolves the port's top open
item two ways:

- The collaborator confirmed the on-board setup **gates the unprivileged
  `cycle`** (`ccsr_en`/`mcounteren`), so the probe must read `mcycle`. `mcycle`
  is a machine-level CSR, so the concern was whether a *domain* (PRV_C, not
  M-mode) may read it at all. **Under our QEMU + OpenSBI Capstone monitor it
  can** — so the domain-payload measurement model (not a bare-metal M-mode
  harness) is viable with `mcycle`.
- `rdcycle` also works here (our monitor exposes `counteren.CY` to the domain),
  retained as `-DFPGA_CYCLE_USE_RDCYCLE` for setups that expose the counter.

**Still verify on first board boot** — the on-board monitor/core build could
differ. If the domain faults on `mcycle`, fall back to
`-DFPGA_CYCLE_USE_RDCYCLE` (+ monitor `counteren.CY`) or an M-mode harness.

### 2. Single-region read-back is UNSOUND — the task-007 host-landmine (FIXED)

The port originally handed **one** region `REV_TRANSFERRED`, had the domain write
results through the reclaimed borrow handle, and had the controller read that
same region back. Under QEMU this aborted:

```
qemu-system-riscv64: ../target/riscv/op_helper.c:666: helper_cslcc: Assertion `rs1_v->tag' failed.
```

Cause: `REV_TRANSFERRED` surrenders the host's mapping; after the domain revokes,
the monitor has dropped the host's `cpmp` entry, so the controller's read-back
traps — exactly the task-007 "host must not touch a transferred region" landmine.
The **same OpenSBI Capstone monitor runs on the FPGA**, so this would have
aborted there too.

**Fix:** two regions. `regions[0]` = the LINEAR arena (`REV_TRANSFERRED`, the
borrow loop must `mrev` it); `regions[1]` = a results region handed `REV_SHARED`
(0x2) so the host **retains** its mapping and reads the 8 results back cleanly.
The two shares arrive as two `REGION_SHARE` domain-entries and are stored by
arrival order. Confirmed working above.

### 3. Capstone backend ICEs at `-O2` on a conditional capability store (WORKED AROUND; codegen lane)

Storing the delivered capability conditionally into two *distinct named* globals
(`if (first) arena = arg; else results = arg;`) **segfaults the Capstone backend
in codegen at `-O2`** (clang 22.0.0git `909c8722`; `+assertions`). `-O0`/`-O1`
compile. Bisected: not the null-test, not the borrow loop, not the write target —
purely the two-named-global conditional store. An **array-indexed** store
(`regions[i++ & 1] = arg`) compiles at `-O2` and is what the port now uses.

This was a real `llvm/` codegen defect (B-lane) — flagged in `COORDINATION.md`.
It has since been **FIXED** (2026-07-15,
`history/15-07-2026_03-43-21_cap-select-o2-ice-fixed.md`): `lowerSELECT` now
rematerialises the constant arms of an i128 capability select. The `regions[]`
array-indexed store is kept anyway (it never formed the offending node and is
harmless), so the ports do not depend on the fix. Repro of the original ICE:
revert to two named globals + a conditional store, build `-O2`.

## Revoke-cost port — temporal-safety number (added 2026-07-15)

The **temporal-safety overhead** arm (the paper's headline perf comparison vs
CHERI) now has an FPGA port too, next to borrow-cost:
`revoke_cost_fpga.c` / `revoke_cost_probe_guest_fpga.c` /
`build-revoke-cost-fpga.sh`, driven by `run-revoke-cost-fpga-qemu.sh`. It is the
hardware port of `../runtime-qemu/revoke-cost-probe/revoke_cost.c` — same
malloc/touch/free loop under three allocator configs (bump / norevoke / revoke),
selected per `.dom` build, reading `mcycle`, with the 4 counters handed back
through a retained results region.

**QEMU plumbing VALIDATED (`run-revoke-cost-fpga-qemu.sh`), matches the reference
instruction-count probe exactly:**

```
bump      (unprotected baseline) : 7
norevoke  (alloc-side)           : 60
revoke    (full temporal safety) : 65
alloc-side overhead (norevoke-bump)  : +53  (8.57x)
revoke-at-free op   (revoke-norevoke): +5  (the O(1) op)
total temporal cost (revoke-bump)    : +58  (9.29x over baseline)
```

Identical to `revoke-cost-probe/RESULTS.md` (bump 7.01 / norevoke 60 / revoke 65,
revoke-at-free +5) — confirming the port measures the same code. On silicon these
become real cycles; the **+5 revoke-at-free** is the O(1) number to place opposite
CHERI's ~14–17 M-instr eager sweep.

### Finding: two large shared regions starve the domain — keep the results region small

The revoke-cost port needs the arena `REV_TRANSFERRED` (the allocator
SPLIT/mrev/revokes it) plus a `REV_SHARED` results region the host reads back.
Initially both were `ROF_COST_REGION_SIZE` (256 KiB). With two 256 KiB regions
the **arena arrived unusable**: `rof_malloc` returned NULL on the first
iteration, the loop dereferenced NULL, and the domain derailed (a stray
`csrr`-decoded `csdebugprint` — `[CAPSTONE] Print = Scalar(0x1234)` — then hung).
A single 256 KiB arena (the reference probe) and two *4 KiB* regions (the
borrow-cost port) both work; two *256 KiB* regions do not. **Fix:** the results
region only holds 4 words, so it is now a single 4 KiB page
(`ROF_RESULTS_REGION_SIZE`). Worth remembering for any multi-large-region domain.

## Hardware run — still to do (unchanged)

The human-driven FPGA sequence (build `fw_payload.bin` via `caplifive-system`,
overlay the two artifacts, boot, run, copy the `RESULT` lines) is in `README.md`.
The QEMU validation means the only genuinely-unknown items left for the board are
(a) `rdcycle` under the *on-board* monitor (item 1, now likely fine), (b) the
overlay wiring (item 2), and (c) RTL identity (item 3).
