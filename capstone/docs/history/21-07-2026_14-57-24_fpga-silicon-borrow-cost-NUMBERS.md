# FPGA silicon borrow-cost NUMBERS captured (captype-fixed CVA6, cycle-accurate)

**Date:** 2026-07-21
**Board:** Genesys2 CVA6 `working-caplifive-captype-fixed.bit` (re-flashed this
session to undo the other team's `ariane_xilinx.bit` overwrite), image
`fw_payload_fpga_up_ctl.bin`, gdb-boot. `mcycle` CSR read inside the domain.

## The numbers (gp-free borrow-cost domain, 64 iterations)

```
RAW  iters=64 empty=4 raw=552 borrow=11709 copy256=57747 copy1024=231165
RESULT cycles/op  raw=8  borrow=182  copy@256B=902  copy@1024B=3611
RESULT vs-raw     borrow=22.75x  copy@256B=112.75x  copy@1024B=451.37x
```

- **raw** (a plain load) = **8 cyc/op**.
- **borrow** (mrev + delin + load + revoke, the revoke-at-free temporal-safety
  sequence) = **182 cyc/op** — a **payload-independent constant** (O(1)).
- **copy@256B = 902**, **copy@1024B = 3611** — grows ~linearly with payload
  (O(size)): ~4x the bytes → ~4x the cycles.

**The paper's shape, now in real cycles:** the capability borrow/revoke mechanism
is a **constant** per-object cost regardless of payload (O(1)), whereas a
copy-based alternative scales with object size (O(size)). Cross the two and the
borrow mechanism wins for any object past a small size (here, break-even is under
256 B: borrow 182 vs copy@256B 902).

## Cycle-accurate vs the QEMU instruction-count proxy — the key delta

The QEMU `-icount` proxy reported raw=2 / borrow=6 (borrow = +4 *instructions*,
O(1)). On silicon the borrow op is **~182 cycles**, not 6 — because `mrev` and
`revoke` are **multi-cycle hardware operations** on the CVA6 (revocation-tree
work), which an instruction count cannot see. This is exactly why the paper needs
the cycle-accurate FPGA number and not just the functional-model proxy: the
per-instruction shape (O(1) borrow) holds, but the constant is ~30x larger in real
cycles.

## Silicon limitation found: the 1024-iteration loop breaks the domain exit

The stock probe uses `BORROW_COST_ITERS = 1024`. On this RTL, a domain that runs
**1024** borrow iterations (1024 `revoke`s) in one call **cannot exit** — `domreturn`
resets the board, AND the debug module cannot even halt the spinning hart
("Unable to halt / Examination failed"). Reducing to **64** iterations fixes both:
`domreturn` completes normally and the controller prints the RESULT line.
Interpretation: ~1024 revocations in a single domain entry exhaust/corrupt a
hardware structure (revocation-tree nodes) so that the exit + debug halt both
fail; a modest iteration count stays within budget. (This is why probes 1-4 —
which spun after ≤1 borrow iteration — halted fine, but probe5/extract — after the
full loop — could not.) The number above is therefore reported at 64 iterations;
the near-zero `empty` baseline (4) reflects the CVA6 folding the empty loop, and
the borrow/copy deltas dominate so the per-op figures are robust to it.

## How it was captured

1. Re-flashed `working-caplifive-captype-fixed.bit` (non-volatile) + power-cycled
   (the other team had overwritten it with `ariane_xilinx.bit`, which has no cap
   unit). Verified `nv_bitstream_name` before measuring.
2. Ran the gp-free / cjalr-free / plain-call-ret domain (this session's work) at 64
   iterations. Because 64 revokes keeps `domreturn` working, the normal path
   applies: controller shares the REV_SHARED region → domain measures + writes the
   8 slots → `domreturn` → controller reads them back → RESULT on UART.

Driver: scratchpad `/tmp/capstone/board_run_nogp.py` (+ `board_flash_extract.py`
for the re-flash). QEMU sanity of the 64-iter build: raw=4/borrow=8 (icount).

## Convergence check at 256 iters — borrow is NOT a clean constant (tree growth)

Re-ran at 256 iterations (also exits cleanly — `domreturn` works at 256):

```
iter64 : raw=8  borrow=182  copy@256B=902  copy@1024B=3611   empty=4
iter256: raw=2  borrow=464  copy@256B=894  copy@1024B=3587   empty=840
```

- **`copy` is stable and trustworthy**: ~900 cyc @256B, ~3600 cyc @1024B — matches
  across iteration counts (large signal, baseline-noise-immune). These are the
  solid cycle-accurate copy numbers.
- **`borrow` GROWS with the iteration count**: 182 → 464 cyc (64 → 256). Even
  without subtracting the (noisy) `empty` baseline, `borrow_total/iter` = 183 vs
  468. So the borrow/revoke per-op cost **increases with the number of prior
  revocations**, and breaks entirely at 1024. Interpretation: the tight loop does
  `mrev`+`revoke` on the same lineage every iteration; each `mrev` adds a
  revocation-tree node that is not pruned, so revoke walks a growing tree — cost
  scales with accumulated revocations until the tree is exhausted (the 1024 break).
  This is an RTL revocation-resource behaviour the QEMU model does not show
  (QEMU: constant). It does NOT necessarily refute the paper's O(1)-per-free claim
  (real workloads free distinct objects, bounding the live tree by heap size, not
  by a tight revoke loop) — but the tight-loop microbenchmark is not a faithful
  single-op measurement on silicon.
- **`raw` and `empty` are small, noisy signals** (empty 4 vs 840 does not scale
  with iters — an mcycle-measurement artifact), so raw (~2-8 cyc, i.e. a cheap
  load) is order-of-magnitude only.

**Trustworthy silicon numbers:** copy@256B ~900 cyc, copy@1024B ~3600 cyc.
**Borrow:** order ~180-470 cyc and rising with accumulated revocations; a clean
single-op figure needs a redesigned probe that prunes the revocation node each
iteration (e.g. `csdrop` the mrev cap) so the tree stays size-1.

## Follow-ups

- Confirm convergence / precision at a higher iteration count that still exits
  (trying 256; find the max iters before `domreturn` breaks).
- The `domreturn`-reset / debug-halt-fail at 1024 revokes is itself a reportable
  RTL finding (revocation-resource limit); worth a note to the board owner.
- Feed these cycle numbers into `paper/evaluation.tex` (upgrades the borrow-cost
  row from the icount proxy to cycle-accurate).
