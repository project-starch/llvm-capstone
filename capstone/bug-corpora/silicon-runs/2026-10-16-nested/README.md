# memcached's nested allocators and FFmpeg's plane corpus on silicon: pre-registration (2026-10-16, before the boot)

This is the fourth board set of `docs/plans/2026-10-15-bug-corpus-on-fpga.md`. It holds what set 3 held back
(`../2026-10-16-carved`, "Held for the next set"). The four arms run in one boot on `caplifive_sublet_d93a8d66b`:
- private image f869aa4b3f5c474a, firmware 8c1a74aec8105d2d;
- the same line as sets 2 and 3: the exact-regions supervised-call monitor (wrapper 1269c77, monitor ebe0a25) and the
  127-slot process-ABI module.

Board order: plane-silicon-level0, plane-silicon-carve-bounds, memcached-silicon-mc-stock, memcached-silicon-mc-sublet.
Each arm has its own rung, and its keys are prefixed with the arm.

## The level0 change this set is built on

Under `CAPSTONE_LEVEL0_LIFETIMES`, a request of 4 KiB or more is now placed on the granule of its padded size and
derived at that padded size. The padded size is the request rounded up to its granule, 2^(hb(n) - 9).
- Its child is exact, and base == cursor.
- Up to one granule minus one byte above the request is inside the child, which is the outward rounding the RTL would
  apply anyway.
- A smaller request is placed as before. The default build is unchanged.

The change exists because set 3's QEMU stage showed memcached's patch 0006 depending on cursor == base. The patch's
`slabs_take_back` computes `gen + (ptr - base(gen))`, and on silicon an inexact generation's base rounds below its
cursor.

**Checks before this boot:**
- **Default build:** byte-identical (b0-hello 0042544c7ec7e2cf).
- **Two-sided control** on the silicon-mimic QEMU, one program allocating, freeing and re-reading blocks of 100 B to
  2 MiB + 7:
  - before: six inexact fallbacks, and a stale read of a freed 2 MiB block reads 7;
  - after: no fallbacks, and the same read faults.
- **The plain corpora's lifetimes arm**, rebuilt on the change: 72 of 72 as predicted on QEMU, controls ok.

## The arms

| arm | corpus | allocator | build |
|---|---|---|---|
| memcached-silicon-mc-sublet | memcached/allocator-repros (9) + control 90 | slabs.c and cache.c with the application's patch 0006 (every chunk and cache object a CDERIVE child, CREVOKEd at slabs_free / cache_free), over the lifetimes heap; `shared/driver-virtual.c` | `-DMC_CARVE_BOUNDS -DMC_CAPSTONE_SUBLET`, 16 MiB arena |
| memcached-silicon-mc-stock | the same | slabs.c and cache.c stock (patches 0001 + 0002) on the stock-malloc ledger (`MCP_ADAPTER_BACKING`, `src/cheribsd/malloc-leases.c`), over the lifetimes heap; `shared/driver.c` | 84 MiB arena (the driver takes 80 MiB up front) |
| plane-silicon-level0 | ffmpeg/plane-repros (1) | `av_frame_get_buffer`'s one buffer, planes by pointer | level0, bounds only |
| plane-silicon-carve-bounds | the same | each plane narrowed by SHRINK (`FFP_CARVE_BOUNDS`) | level0, bounds only |

- **memcached** sources are the port's own `cmake/prepare-source.py` output from the pinned 1.6.45 archive: the
  protected variant for the Sublet arm and the ported variant for the stock arm. They are built at -O1, as
  `shared/build-cases.sh`, with `-fno-inline` so the probes stay attributable under full LTO.
- **plane** links 14 units of FFmpeg 9.0.1's libavutil (frame, buffer, mem, imgutils, pixdesc, log, error, dict,
  bprint, channel_layout, samplefmt, refstruct, time, side_data). They come from the tree its recorded
  capstone-carve-bounds run used, with that run's config.h. Built at -O0, as every case.c corpus here.
- Control 90 (a slab chunk freed and read through its old pointer) is a row that gates its arm. It must fault on the
  Sublet arm and complete on the stock arm.

## The silicon-mimic QEMU reading

Instrument: capstone-qemu silicon/cderive-round ce1be0c642 with `CAPSTONE_CDERIVE_ROUND=1`, GP_FABRICATE=0, firmware
fw-b1-top d5c57ee765c9 (set 3's README has its checks). Every arm's tool controls are as declared.

**memcached-silicon-mc-sublet:**
- 00-04 (temporal): CAUGHT 24 at `read_probe`.
- 06, 07 (spatial, through MC_CARVE_BOUNDS's narrowed regions): CAUGHT 7 and 5 at the labelled probes.
- control 90: CAUGHT 24 at `read_probe`.
- **05:** the fixed run faults 24 in CREVOKE (`capstone_cap_revoke_child`), so the row reads FIXED-ARM-FAILED. The
  virtual arm recorded the same (NO-READING, control-failed), so it is not new to silicon.
- **08: the case refuses its own premise (exit 75, CONTROL-FAILED 785).** Its scan of an all-spaces 16 KiB read buffer
  must run past the object. Under patch 0006 the read buffer is a cache.c child over the slot and the object, 16,400
  bytes. That size is inexact at 32-byte granules, so the child's top rounds up 16 bytes. The scan stops at the first
  non-space byte in that slack, before it reaches the bound, and the case's CHECK refuses the run.
  - The RTL rounds this child the same way with or without level0's padding.
  - This is a predicted spatial difference: decision (a), accepted, not padded away.

**memcached-silicon-mc-stock:**
- 00-07: DEFECT-REPRODUCED. The stock allocators reissue the storage, and nothing faults.
- 08: CAUGHT 5 in `mc_case_body`, the case's declared site. Here each read buffer is its own 16 KiB malloc, exact under
  the aligned placement, so the scan faults at its end.
- control 90 completes.

**plane:**
- level0: the case reproduces (DEFECT-REPRODUCED), because the crossing stays inside the frame's one buffer.
- carve-bounds: CAUGHT at the labelled probe.

## The board prediction

Every row as QEMU read it, by outcome and site. Causes map as the earlier sets read them on this bitstream: QEMU's
5 / 7 read 28, and 24 reads 25 (ISSUES Q-11). For 05's fixed run that gives cause 25 in `capstone_cap_revoke_child`.
For 08 there is no fault: the run exits 75 with CONTROL-FAILED 785.

**New on silicon, and what a failure would say:**
- **The aligned placement.** A 1 MiB slab page is placed on a 2 KiB boundary inside a 16 MiB arena. If its CDERIVE is
  not exact on the RTL, the generation's base and cursor part, and slab frees fault 28 in `do_slabs_free` (set 3's
  QEMU signature).
- **88 MiB domains** for the stock arm, as set 2's sub-object images (which launched).
- **libavutil's frame allocator on silicon** (plane), for the first time.

## Held for the next set: wireshark's wmem

Both wmem arms were built and run on the same QEMU:
- the stock arm read 23 of 23 as predicted;
- the Sublet arm read 22 of 23. Its case 22 (a double free) faults in `capstone_cap_revoke_child`, not in its
  declared `wmem_block_sublet_free` / `_release`.

That is this build's `-fno-inline` making a `static inline` helper a function of its own, not a silicon difference.
wmem's probes are `noinline` with asm labels, so both arms are being rebuilt with the port's own Release flags. They
are pre-registered separately.
