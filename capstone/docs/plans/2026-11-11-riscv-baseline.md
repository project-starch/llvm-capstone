# Capstone against ordinary RISC-V: the baseline and the best Capstone build (plan, 2026-11-11)

## Decision (project lead, 11 Nov)

The SPEC baseline is ordinary RISC-V. Plain runs on a bitstream of the same SoC with the Capstone extension off, with
stock OpenSBI and the same Linux kernel. Until now Plain ran on the Capstone bitstream and the Capstone monitor
(train PREREG Addendum 40), so every Plain-relative ratio so far measured only the cost of Capstone's software.

The Capstone side uses the best compiler and options available, chosen per benchmark on SPEC test before SPEC train.
Every per-benchmark choice is recorded with its exe hash and flags, and is disclosed.

## Track A: the ordinary RISC-V baseline

| Step | What | Gate |
|---|---|---|
| A1 | Config `baseline_cv64a6_imafdc_sv39` = `capstone_cv64a6_imafdc_sv39` with only CAPSTONE_EXT = 0 and no `CAPSTONE_ITFC_EN`. Caches, TLBs, predictors, commit ports and PMP stay as they are | diff of the two config files: those two lines only |
| A2 | Elaboration and a smoke in Verilator (rtl-sim), against the Capstone config | rtl-lint-gate numbers; riscv-tests smoke passes |
| A3 | Synthesis on apollo under vivado-guard (launch-job.sh), from the same RTL commit as the reference Capstone bitstream | no loops, routed clean, WNS read beside the Capstone seal; sealed hash |
| A4 | Stock OpenSBI fw_payload (fpga/ariane) with the same Linux Image, dtb and initramfs; the harness's Plain-only path without the Capstone monitor markers | QEMU or sim boot first where possible |
| A5 | Board Tier 1: boot, hiccup2 and tickcost, then the Plain test stages | Plain test outputs pass specdiff |

What A answers:
- the extension's hardware cost on ordinary code, as Plain on Capstone hardware against Plain on the baseline;
- the Capstone monitor's cost on ordinary code (the tick path);
- a Plain baseline that a reviewer accepts as "ordinary RISC-V".

## Track B: the best Capstone build per benchmark (SPEC test)

- **Candidates**, all lifetimes (ObjLife = Temporal):
  - base: devmk, llvm-c1;
  - dcnofp: c4, `-capstone-direct-calls -fomit-frame-pointer`;
  - dcnofprs: dcnofp plus `-capstone-drop-redundant-shrink`;
  - shrinkto-globals variants;
  - gcc's arena-cache link (ac15).
- **B1. Screen in QEMU.** Build each candidate for the 8 benchmarks and twin it in QEMU at test size: specdiff PASS,
  in-context instret from capmix. Keep the best one or two per benchmark.
- **B2. Choose on the board.** Run each kept candidate on SPEC test, same board, kit and bitstream. Choose per
  benchmark by wall time, with the in-context cycles beside it.
- **B3. Native route.** Build the chosen configuration with the native SDK, so native uses the best compiler (the
  lead's rule).
- **Constraints.** No `+movc-keeps-integer-source` unless Q-04(b) reads PRESENT on the bitstream (C-32). This is open
  for C.5a and V1f.

## Track C: SPEC train

- Plain on the baseline, and Capstone with B's choice, per benchmark.
- Every run is pre-registered, with exe hashes and flags.
- The tick-neutral column of Addendum 40 is re-derived on the baseline with its own t_P.

## Order

1. A1-A2 and B1 in parallel. They need no board.
2. A3 synthesis runs in the background, about 2-3 h.
3. Meanwhile B2 uses the free boards.
4. A4-A5 once the bitstream is sealed.
5. Then C.
