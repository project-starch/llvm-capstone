# FPGA ladder-perf bridge — built + QEMU-validated (board run pending)

**Date:** 2026-07-24
**Task:** `plans/fpga-ladder-perf-task-B.md` (deadline-critical FPGA cycle counts).
**Status:** the QEMU→board bridge is built and validated end-to-end on QEMU — all
7 ready-set rungs return the correct checksum through the real controller + perf
domain + region path. The on-board `mcycle` capture is the next step.

## The bridge
The ladder `.dom` binaries are already board-identical (the entry glue
`start-gp-captable-generic.S` delivers the shared-region cap as `domain_main`'s
arg and unwinds via `domreturn`), so the bridge is just cycle instrumentation +
a controller:
- `tests/runtime-qemu/silicon-ladder/ladder_perf_domain.h` — perf `domain_main`:
  brackets `<rung>_compute()` with `mcycle` reads, writes `res[0]=retval`,
  `res[1]=cycles`, `res[2]=0xD09E` (ran-marker) into the shared region.
- `tests/runtime-qemu/silicon-ladder/<rung>_fpga_app.c` (7) — per-rung 3-liners
  (`#include` kernel, `#define LADDER_COMPUTE`, `#include ladder_perf_domain.h`).
- `tests/rtl-smoke/ladder_perf_ctl.c` — generic freestanding soft-float controller
  (raw Linux syscalls, no glibc): create domain, create+share ONE 4096-B
  REV_SHARED region (= the entry), read back res[0..2], print
  `RESULT <name> retval=<r0> cycles=<r1> ran=<r2>`.
- `tests/rtl-smoke/build-ladder-fpga.sh` — builds the controller (buildroot gcc) +
  each rung's perf `.dom` (silicon gp-captable config) + the native oracle.

## QEMU validation (full board path minus the FPGA)
Ran the controller on all 7 perf domains inside the QEMU guest in one boot
(`run-domain-smoke.py --guest-command`, `CAPSTONE_GP_FABRICATE=0`, region 4096 B).
Every rung: `retval == native cc -O0 oracle`, `ran=0xD09E`, no fault:

| rung | retval == oracle |
|---|---|
| matmult_int | 774662735 |
| coremark_matrix | 14343 |
| rv8_primes | 99991 |
| beebs_crc32 | 1703161001 |
| beebs_insertsort | 271779359 |
| beebs_prime | 582955588 |
| beebs_recursion | 1579141629 |

(Cycle counts under QEMU are not meaningful — QEMU has no pipeline model; the FPGA
`mcycle` is the real number. The `mcycle` read works in-domain on QEMU: the domain
reads it, computes, reads again, and writes both slots.)

Note: `run-domain-smoke.py`'s default loader (`/capstone-test.user`) hands the
domain only an 8-byte region, so the perf domain (3 slots) must be driven by the
controller (4096 B), not the default loader.

## Toolchain
The `-b` LLVM/clang was rebuilt from scratch (system `/usr/bin/clang++`, targets
`RISCV;Capstone`, `LLVM_OPTIMIZED_TABLEGEN`, `BUILD_SHARED_LIBS`, `clang;lld`);
all 7 perf domains build with `cjalr=0`, `ldc-gp>=1`.

## Caveat carried to the board run
The gp-captable codegen has an OPEN, un-root-caused silicon miscompile (a loop
that stores to a global array while keeping a live accumulator can return an
address-contaminated value on hardware; QEMU-correct, silicon-wrong — see
`plans/gp-captable-codegen-plan.md` §Stage-4 / `history/23-07-2026_17-30-00_*`).
Some rungs match that shape, so a subset may fail `retval==oracle` on the board;
the correctness gate captures that as a finding, not a bridge bug.
