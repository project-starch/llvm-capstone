# The sub-object corpus and FFmpeg's AVBufferPool corpus on silicon: pre-registration (2026-10-16, before the boot)

This is the second board set of `docs/plans/2026-10-15-bug-corpus-on-fpga.md`, after the plain corpora
(`../2026-10-16-plain`, 72 of 72 in both arms). The five arms run in one boot on `caplifive_sublet_d93a8d66b`, from
one private image whose firmware is 37ae782f7718b018. The firmware is built on the exact-regions supervised-call
monitor (wrapper 1269c77, monitor ebe0a25) with the 127-slot process-ABI module. Each arm has its own corpus rung,
and its run keys are prefixed with the arm (`run-silicon-cases.py stage --prefix-keys`).

## The arms

| arm | corpus | heap | the corpus's own allocator |
|---|---|---|---|
| silicon-subobject | ffmpeg/subobject-repros (9) | level0 + field bounds (`-Xclang -fcapstone-subobject-bounds` on every source and the controls) | the buffer-pool port's pool core, as recorded |
| silicon-level0 | ffmpeg/subobject-repros (9) | level0, bounds only | the same |
| silicon-lifetimes | ffmpeg/subobject-repros (9) | level0 + CAPSTONE_LEVEL0_LIFETIMES | the same |
| silicon-pool-stock | ffmpeg/pool-repros (4) + control 90 | level0 + lifetimes | AVBufferPool/AVRefStructPool stock: the port's "traced" FFmpeg variant (patch 0001) |
| silicon-pool-sublet | ffmpeg/pool-repros (4) + control 90 | level0 + lifetimes | the same pools as Sublet lifetimes: the "protected" variant (patches 0001 + 0003) |

The FFmpeg variants were prepared by the port's own `cmake/prepare-source.py` from the pinned 9.0.1 archive (sha256
cf38e0e2...).

**How each corpus builds,** as recorded in each arm's `qemu-and-predictions.json` build block:
- **Sub-object:** -O0, as the recorded runs' cc_args, with `aligned_alloc(a,n)=malloc(n)`. The driver takes 64 MiB of
  payload and 16 MiB of metadata up front (trace.h), so these images run with a level0 arena of 84 MiB in an 88 MiB
  data region (B0_ARENA / B0_DATA).
- **Pool:** `-O1 -std=gnu11 -DFF2_SYSTEM_MEMORY`, as `shared/build-cases.sh`, with a 2 MiB data region.
  - Plus `-fno-inline`: under full LTO, -O1 otherwise inlines the corpus's probes and fault sites into main, and a
    fault there could not be attributed.
  - Plus `tools/silicon/level0-memalign.c`, posix_memalign and aligned_alloc over level0: av_malloc's 64-byte request
    gets level0's 16.

**level0 lifetimes, as revised for this set:**
- A block's header records the CDERIVE child it was handed out as.
- free finds the block by address, validates the pointer with a load, and revokes the recorded child. This is the
  virtual heap's contract.
- The revision was needed because FFmpeg's refstruct frees `obj - offset`, a pointer derived from a pool's own child,
  which the first version (CREVOKE of the freed pointer itself) refused with 29 (pool case 03's fixed run on QEMU).
- The default build stays byte-identical (b0-hello 0042544c7ec7e2cf).
- The plain corpora's lifetimes arm is being re-run on QEMU with the revised level0 as a regression check.

## The silicon-mimic QEMU (capstone-qemu d35af47b163c, GP_FABRICATE=0)

Every arm reads all its rows as predicted (`qemu-and-predictions.json`), with every control as declared.
- **silicon-subobject:**
  - 7 CAUGHT at the labelled probe (00 01 02 03 05 07 08), the cases QEMU's capstone-subobject arm caught
    (case.json);
  - 04 and 06 MISSED, whose images are byte-identical to the level0 arm's: field bounds change nothing in them.
- **silicon-level0** and **silicon-lifetimes:** 9 MISSED each. Every crossing stays inside one allocation.
- **silicon-pool-stock:** cases 00-03 MISSED, and control 90 completes (the returned buffer read through its old
  pointer).
- **silicon-pool-sublet:** cases 00-03 CAUGHT in ff2_case_run, a fault site the cases declare (case.json fault_sites).
  Control 90 is CAUGHT at read_probe.

## The board prediction

Every row as QEMU read it, by outcome and site. Causes map as the plain set read them on this bitstream: QEMU's 5 / 7
read 28 (the RTL's OUT_OF_BOUNDS) and QEMU's 24 reads 25 (ISSUES Q-11). The controls are as declared per arm:
- silicon-subobject: subobj CAUGHT, uaf RETURNED;
- the lifetimes and pool arms: uaf CAUGHT 25.

**New on silicon, and what a failure would say:**
- **The 88 MiB domains of the sub-object images,** larger than any run so far (B5's 64 MiB). A launch failure (CMA)
  ends capstone-exec with an error, not a verdict, and reads NO-READING.
- **The pool's nested CDERIVE / CREVOKE inside the lifetimes heap:** a child of a child of level0's group. B6 read
  the two-level pattern on silicon; this is the first application allocator on it.
- **Field bounds on silicon,** which no board run has used before.
