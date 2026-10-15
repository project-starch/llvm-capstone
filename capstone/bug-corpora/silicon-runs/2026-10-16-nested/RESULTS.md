# memcached's nested allocators and FFmpeg's plane corpus on silicon: results (2026-10-16, caplifive_sublet_d93a8d66b)

Pre-registered in README.md (lane 02105a6f4c96, before the boot). One boot, 07:11-07:43:
- firmware 8c1a74aec8105d2d, Image f869aa4b3f5c474a, on the line of sets 2 and 3;
- four arms, 60 runs, every one ended by itself (each arm's rung returned 0).

**Every outcome and every site is as pre-registered,** and so is every control. One cause differs, in a row whose
verdict does not change (05, below). Every image is the pre-registered one (sha256), and every fixed run that was to
read FIXED did.

| arm | rows | control 90 | tool controls (clean / oob / uaf / subobj) |
|---|---|---|---|
| memcached-silicon-mc-stock | 00-07 MISSED; **08 CAUGHT 28** in mc_case_body (declared) | completes | RETURNED / CAUGHT 28 / CAUGHT 25 / RETURNED |
| memcached-silicon-mc-sublet | **00-04 CAUGHT 25, 06-07 CAUGHT 28** at the labelled probe; 05 no reading (its fixed run fails); 08 no reading (CONTROL-FAILED 785) | **CAUGHT 25** at read_probe | RETURNED / CAUGHT 28 / CAUGHT 25 / RETURNED |
| plane-silicon-level0 | 00 MISSED | - | RETURNED / CAUGHT 28 / RETURNED / RETURNED |
| plane-silicon-carve-bounds | **00 CAUGHT 28** at the labelled probe | - | RETURNED / CAUGHT 28 / RETURNED / RETURNED |

**What it shows:**
- **memcached's slab and cache allocators reissue a freed chunk with nothing to stop a stale reader** on the stock
  arm: 00-04 MISSED.
- **Patch 0006 catches each of those five at the stale read on silicon,** cause 25. Every chunk and cache object is
  a CDERIVE child that slabs_free / cache_free revokes. Control 90 (a chunk freed and read back) faults the same way.
  - This needs set 4's level0 placement. With blocks of 4 KiB and up on their granule, the slab page's generation has
    base == cursor, and patch 0006's `slabs_take_back` arithmetic lands on the chunk.
  - On set 3's level0, the same patch faulted in `do_slabs_free` (QEMU, set 3's README).
- **The spatial cases 06 and 07** are caught through MC_CARVE_BOUNDS's narrowed key and suffix regions, as on the
  virtual arm.
- **08 is caught only on the stock arm, which has no nested protection.** There each read buffer is its own 16 KiB
  malloc, an exact level0 child, and the all-spaces scan faults at its end. Under patch 0006 the buffer is a cache.c
  child over slot and object, 16,400 bytes, which rounds up 16 bytes at 32-byte granules. The scan stops in that slack
  before the bound, and the case refuses its own premise (exit 75), as pre-registered: decision (a)'s accepted spatial
  difference.
- **libavutil's frame allocator on silicon:** a read one row past a plane stays inside the frame's single buffer
  (MISSED), and narrowing each plane at `av_frame_get_buffer`'s carve catches it.

**The one cause that differs: memcached 05's fixed run, CAUGHT 24 in CREVOKE where 25 was predicted.** The row reads
no reading either way, as pre-registered and as on the virtual arm.
- **The prediction** mapped QEMU's 24 to the RTL's 25 by rule (ISSUES Q-11: QEMU untags a revoked capability that the
  RTL keeps tagged).
- **Here the operand is not a capability at all,** on silicon too.
  - The case plants a sentinel with `memset(next, 0x5a, 1)` on the next chunk's first byte.
  - Under patch 0006 that byte is the start of `sublet_parent`, the generation the patch stores first in every item.
  - The write clears its tag, and the next chunk's `slabs_free` hands CREVOKE an untagged generation (24).
  - Patch 0006 says as much: "Overwriting the slot clears its tag and the next free faults".
- **For the corpus:** case 05's setup and patch 0006's item layout conflict, on every Sublet build.
- **For the predictions:** a QEMU 24 maps to 25 only where Q-11's mechanism holds, a revoked capability. It does not
  hold for a value that was never a capability.

## Per case

| case | memcached-silicon-mc-stock | memcached-silicon-mc-sublet |
|---|---|---|
| 00_7af02b0c87_rbuf_copied_after_cache_free | MISSED | CAUGHT 25 |
| 01_0ad4de66ae_io_walk_reads_freed_link | MISSED | CAUGHT 25 |
| 02_59bd02ce29_tail_repair_frees_referenced_item | MISSED | CAUGHT 25 |
| 03_a8c4a82787_refcount_overflow_frees_linked_item | MISSED | CAUGHT 25 |
| 04_152ddb68f7_unlocked_refcount_drift | MISSED | CAUGHT 25 |
| 05_2d61f18_item_data_one_past | MISSED | no reading (fixed run: 24 in CREVOKE) |
| 06_78eb770_suffix_write_no_space | MISSED | CAUGHT 28 |
| 07_ecdb011_unterminated_key_read | MISSED | CAUGHT 28 |
| 08_e8364b5_ascii_all_spaces_scan_past_rbuf | CAUGHT 28 in mc_case_body | no reading (CONTROL-FAILED 785) |
| 90_ctl_chunk_freed_to_slab (control) | completes | CAUGHT 25 |

| case | plane-silicon-level0 | plane-silicon-carve-bounds |
|---|---|---|
| 00_b7946098b1_alphablend_row_past_plane | MISSED | CAUGHT 28 |

A CAUGHT with no site is at the case's labelled probe. **Judged by** `tools/run-silicon-cases.py board`, then compared
with the pre-registered outcome, cause and site, keyed by image hash.
