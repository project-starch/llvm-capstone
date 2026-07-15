# RTL/FPGA borrow-cost port — QEMU plumbing validation (task-016 Step 3)

**Date:** 2026-07-14. **Outcome:** the staged FPGA borrow-cost
port (`capstone/tests/rtl-smoke/`) now builds `-O2` and runs green end-to-end on
the QEMU functional model; two real defects found and handled before any hardware
slot. Full results: `tests/rtl-smoke/RESULTS.md`.

## Why

The port was staged UNTESTED (task-016 Step 3). Rather than wait for an FPGA
window, validate the plumbing — two-region hand-off, `rdcycle` read in a domain,
controller read-back — on QEMU first, since the same OpenSBI Capstone monitor
runs on the board. Cheap to do; expensive to discover a bug mid-session on shared
hardware.

## Trail

1. **First run aborted QEMU:** `helper_cslcc: Assertion rs1_v->tag failed` after
   `call retval` printed. Diagnosed as the **task-007 host-landmine**: the port
   handed one region `REV_TRANSFERRED`, and the controller read that region back
   from its host mapping after the domain revoked it — the monitor had dropped the
   host `cpmp` entry. Not a mechanism failure; a design bug in the port's
   read-back path. Would abort on the FPGA identically.

2. **Fix = two regions.** `regions[0]` LINEAR arena (`REV_TRANSFERRED`, the borrow
   loop must `mrev` it); `regions[1]` results region handed `REV_SHARED` (0x2, the
   same annotation the shared-region-probe uses for a host-retained buffer). Two
   annotated shares arrive as two `REGION_SHARE` entries, stored by arrival order.
   The domain writes the 8 results to `regions[1]`; the host reads them back
   cleanly. `measure_borrow` reverted to byte-identical with the QEMU probe (the
   `out_final` thread-out is no longer needed).

3. **Fix #2 exposed a Capstone-backend `-O2` ICE.** With the order-based receive
   (`if (first) arena=arg; else results=arg;`) the domain **segfaults clang's
   Capstone backend in codegen at `-O2`** (`909c8722`, `+assertions`); `-O0`/`-O1`
   compile. Bisected against the pristine port: not the null-test, not the borrow
   loop, not the write target — purely a **conditional store of a capability into
   two distinct named globals**. An array-indexed store (`regions[i++ & 1] = arg`)
   compiles at `-O2`; the port uses that. This is an `llvm/` codegen defect
   (compiler lane) — flagged in COORDINATION.md, not fixed (shared tree).

4. **Green run.** raw=2, borrow=6 (+4 over raw, payload-independent), copy
   33@256B / 129@1024B — identical shape and identical raw/borrow/+4 to the
   task-014 instruction-count proxy (raw 2 / borrow 6 / 34 / 130; copy off-by-one
   is integer-floor vs round). And **`rdcycle` did not fault inside the domain** —
   resolves the port's top open item for QEMU (the monitor exposes the counter to
   the domain context); strong signal for the board.

## Net

- Port de-risked: builds `-O2`, runs green, plumbing proven. The only genuinely
  unknown hardware items left are the on-board monitor's `rdcycle` exposure
  (likely fine), the buildroot overlay wiring, and RTL identity.
- One new `llvm/` codegen bug for the codegen lane (the `-O2` two-named-capability
  -global conditional-store ICE), with a minimal repro.
- No `llvm/` change, no `capstone-qemu` change, no gitlink bump; additive test
  files only.
