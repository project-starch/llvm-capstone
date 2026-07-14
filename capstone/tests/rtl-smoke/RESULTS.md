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

### 1. `rdcycle` is readable inside a Capstone domain (open item 1 — resolved for QEMU)

The domain executed `rdcycle` (0xC00 `cycle` CSR) with **no fault**. The port's
top open item was whether the monitor exposes the counter to a domain context
(`[m|s]counteren.CY`). Under our QEMU + OpenSBI Capstone monitor it does. This is
a strong signal the FPGA path will also work without a monitor change; the
`mcycle`-in-M-mode fallback in `fpga_instrument.h` stays documented but is likely
unnecessary. **Still verify on first boot** — the on-board monitor build could
differ.

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

This is a real `llvm/` codegen defect (Agent-B's lane) — flagged in
`COORDINATION.md`, not fixed here (shared LLVM tree). Repro: revert the
`regions[]` array in `borrow_cost_fpga.c` to two named globals + a conditional
store, build `-O2`.

## Hardware run — still to do (unchanged)

The human-driven FPGA sequence (build `fw_payload.bin` via `caplifive-system`,
overlay the two artifacts, boot, run, copy the `RESULT` lines) is in `README.md`.
The QEMU validation means the only genuinely-unknown items left for the board are
(a) `rdcycle` under the *on-board* monitor (item 1, now likely fine), (b) the
overlay wiring (item 2), and (c) RTL identity (item 3).
