# FFmpeg's carved corpus on silicon: pre-registration (2026-10-16, before the boot)

This is the third board set of `docs/plans/2026-10-15-bug-corpus-on-fpga.md`, after the plain corpora
(`../2026-10-16-plain`) and the sub-object and pool corpora (`../2026-10-16-subobject-pool`). The three arms run in
one boot on `caplifive_sublet_d93a8d66b`, from one private image whose firmware is 124238b7d5b41abd (Image
0bba51e9baec9816). The firmware is built on the same line as set 2's: the exact-regions supervised-call monitor
(wrapper 1269c77, monitor ebe0a25) with the 127-slot process-ABI module. Each arm has its own corpus rung, and its run
keys are prefixed with the arm.

## The arms

| arm | carve | heap | controls |
|---|---|---|---|
| silicon-level0 | upstream's pointer arithmetic (`block + off`) | level0, bounds only | 99 must COMPLETE |
| silicon-carve-bounds | `FFC_CARVE_BOUNDS`: each region narrowed by SHRINK at `ffc_carve()` | level0, bounds only | 99 must be CAUGHT |
| silicon-carve-sublet | `FFC_SUBLET_CARVE`: a group CDERIVEd over the block, each region a CDERIVE child of it, a re-carve CREVOKEs it | level0 + `CAPSTONE_LEVEL0_LIFETIMES` | 99 and 98 must be CAUGHT |

Every arm also runs the tool's four controls (clean, oob, uaf, subobj) as in the earlier sets. The corpus's own carve
controls run as rows and gate the arm like them (`tools/run-silicon-cases.py`, CONTROL_ROWS):
- 99: one byte past a 16-byte carve, inside its 64-byte block;
- 98: a region's old alias, written after the region was carved again.

The build is set 1's: case.c + shared/driver.c at -O0, gp-captable, full LTO, the pool carve, check-repr under
full+renorm, no-fp-gate. The images use a 2 MiB data region and a 1 MiB level0 arena.

**The plane corpus (1 case) is not in this set.** Its case links FFmpeg's own `libavutil` (`av_frame_get_buffer`),
which has not been built for the silicon ABI. It is in the manifests as BUILD-FAILED and is not staged.

## The instrument: a silicon-mimic QEMU that rounds CDERIVE as the RTL does

Sets 1 and 2 were predicted on capstone-qemu d35af47b163c. That build cannot run this corpus's Sublet carve as silicon
does, for two reasons.
- **CDERIVE.** It refuses an inexact CDERIVE child with 29 (ISSUES Q-15). The RTL rounds the child's bounds outward
  instead (`capstone_dyn_unit.anvil` CDERIVE: no representability check). Case 00's group over a 34,816- or
  35,840-byte block at a 16-byte cursor is inexact, so d35af47 would read 29 in `ffc_block_alloc`.
- **The encoder.** d35af47 predates R-54's renormalisation, which the flashed RTL carries (2a3a87dfc is an ancestor of
  d93a8d66b).

**capstone-qemu `silicon/cderive-round` ce1be0c642** is d35af47b163c plus two changes:
- the renormalisation (origin/bounds-encoder-renormalise, 4a21b485b01f);
- a switch, `CAPSTONE_CDERIVE_ROUND=1` (e8d97b3cfd). With it, an inexact CDERIVE child takes its decoded, rounded
  bounds instead of 29. Default off. The first rounded child prints one line to stderr.

**Checks:**
- **When the RTL rounds** (read at d93a8d66b): every capability result is compressed at the functional unit's
  writeback (`core/ex_stage.sv:1515` and `:1528`, `compress_cap`), and commit writes that compressed word
  (`core/commit_stage.sv:411`). So a CDERIVE child's rounded bounds hold from the next instruction on, as the switch
  models. The encoder there is the single scheme with R-54's step (`core/include/ariane_pkg.sv:777`; the cursorless
  branch is gone).
- **The codec against the RTL lane's `bounds-model.py` `cderive_child`** (capstone-ariane capstone-sublet b7e3287e4):
  200,018 random (cursor, size) triples, 144,150 of them inexact. 0 mismatches.
- **The same check on d35af47b163c's codec** must fire: 43,591 mismatches, every one an R-54 shape.
- **Switch off on an existing arm:** set 2's silicon-pool-sublet reads identically on the new build (4 controls, 5
  rows, 0 differences).
- **Switch on here:** the carve-sublet VM's log shows the one rounded child, case 00's group,
  `[e05f2330, e05faf30)` rounded to `[e05f2300, e05faf40)`.

## The silicon-mimic QEMU reading (switch on, GP_FABRICATE=0, firmware fw-b1-top d5c57ee765c9)

Each arm's `qemu-and-predictions.json` has every row.
- **silicon-level0:**
  - controls as declared;
  - 13 of 13 DEFECT-REPRODUCED;
  - 99 DEFECT-REPRODUCED. The crossings stay inside the block, which is the only bound this arm has.
- **silicon-carve-bounds:**
  - controls as declared;
  - 13 of 13 CAUGHT at the labelled probe;
  - 99 CAUGHT;
  - 69 regions, the largest 2,048 bytes, every one exact.
- **silicon-carve-sublet:**
  - controls as declared;
  - 13 of 13 CAUGHT at the labelled probe;
  - 99 CAUGHT (7);
  - 98 CAUGHT (24) at the probe: the re-carved region's old alias.

**One placement effect, recorded rather than corrected.** In case 00's fixed run, both regions print `base_eq=0`.
- **Why:** the group over the 35,840-byte block is inexact, so its base rounds down 48 bytes while its cursor stays on
  the block. `capstone_cap_derive` offsets a child from the parent's BASE. So each region sits 48 bytes below the case's
  own arithmetic, with its exact length (`bounds=320`).
- **Effect on the case:** the fixed run still prints VERDICT FIXED, and the buggy crossing is still CAUGHT at the
  probe.
- **On silicon:** the RTL keeps the cursor at the requested start and rounds the base down the same way, so it does the
  same.
- **Elsewhere:** every other carved block is below 4 KiB or exact.

The same mechanism is what stops memcached's patch 0006 here: `slabs_take_back` computes `gen + (ptr - base(gen))`.
That corpus is held for the next set (below).

## The board prediction

Every row as QEMU read it, by outcome and site. Causes map as the earlier sets read them on this bitstream:
- QEMU's 5 / 7 read 28 (the RTL's OUT_OF_BOUNDS);
- QEMU's 24 reads 25 (ISSUES Q-11): only 98 and the uaf control on the Sublet arm.

**New on silicon, and what a failure would say:**
- **CDERIVE nested three levels below level0's group** (the block's own child, the driver's group over it, each
  region), with a CREVOKE of the lowest level while the others stay live (control 98). The node accounting is the RTL lane's: CDERIVE +1, CREVOKE frees the child. This arm peaks at a few
  dozen nodes.
  - If 98 reads RETURNED, CREVOKE of a grandchild did not end its alias.
  - If 98 reads 29, the child was not a live direct child of the group the driver kept.
- **An inexact group derive on silicon** (case 00, as above). If case 00's fixed run fails on the board where QEMU read
  FIXED, the RTL's rounding differs from the encoder model for that block.

## Held for the next set: memcached's nested allocators and wireshark's wmem

Both were built and run on the same QEMU. Both need a level0 change first.
- **memcached, silicon-mc-sublet (patch 0006):**
  - **Shifted chunk pointers.** 02, 03, 04, 06, 07 and control 90 fault 5 in `do_slabs_free`. A slab page's generation
    (chunk size x perslab over a 1 MiB page at a 16-byte placement) rounds its base below its cursor, so
    `slabs_take_back`'s pointer is off by the rounding.
  - **CREVOKE on an untagged operand.** The fixed runs of 00 and 08 fault 24 in CREVOKE. Both use cache.c's read
    buffers, 16 KiB, which are inexact at a 16-byte placement. The likely cause is the same base-relative arithmetic,
    reading the lifetime slot through a shifted pointer. That is inferred, not shown: the next set's QEMU reading on
    the changed level0 settles it.
  - 05's fixed run faults 24 there too. It also failed on the virtual arm (NO-READING), so it is not new.
- **memcached, silicon-mc-stock:** the stock-ledger driver takes 64 MiB of payload and 16 MiB of metadata up front, so
  it needs the sub-object set's 84 MiB arena (CONTROL-FAILED 702 at 1 MiB).
- **wmem:** its patch relies on `free()` revoking a freed block's lifetime. level0 gives a 2 MiB block no lifetime at a
  16-byte placement (the inexact fallback), so wmem's jumbo control could not fault.

**The level0 change:** a request of 4 KiB or more is placed on its granule and derived at its padded size, so only a
top rounds and base == cursor everywhere.
- **Checked on this QEMU** with a two-sided control:
  - before: the 4 KiB+ requests fall back six times, and a stale read of a freed 2 MiB block reads 7;
  - after: 0 fallbacks, and the same read faults.
- **The default build is byte-identical** (b0-hello 0042544c7ec7e2cf).
- It lands with that set, not this one, which is built on the level0 of lane 1ef381aaa812.
