# wireshark's wmem corpus on silicon: pre-registration (2026-10-16, before the boot)

This is the fifth board set of `docs/plans/2026-10-15-bug-corpus-on-fpga.md`. It holds wireshark/wmem-repros (23
cases) and its three controls, on two arms, in one boot on `caplifive_sublet_d93a8d66b`:
- private image bded55b35f656a8b, firmware 6cbb3a4494fcdc73;
- the same exact-regions supervised-call line as sets 2-4.

Board order: wmem-silicon-wm-stock, then wmem-silicon-wm-sublet. Each arm has its own rung, with keys prefixed by the
arm.

## The arms

| arm | wmem | heap | the buggy run |
|---|---|---|---|
| wmem-silicon-wm-stock | as released, the port's `reference` source | level0 + `CAPSTONE_LEVEL0_LIFETIMES` | the fix differential, `0 N buggy`. Its temporal cases assert that the stale chunk was reoccupied before the read |
| wmem-silicon-wm-sublet | the port's patch 0001: every block and block_fast object a CDERIVE child of its block, CREVOKEd at free, realloc and the pool reset | the same | the plain sequence, `0 N` |

These are virtual-malloc's and virtual-nested-pools' configurations on the silicon ABI.
- **The fixed run** is `0 N fixed` on both arms. It must print VERDICT FIXED and exit 0.
- **The three controls** (00 uaf-chunk, 01 uaf-jumbo, 02 bounds-chunk) run once each as `0 N`, never as the
  differential, and gate the arm:
  - Sublet arm: all three must fault at the probe;
  - stock arm: 01 must fault, and 00 and 02 must complete, as `controls/README.md` declares per arm.
- **The tool's four controls** run on each arm too.

**The build:**
- Sources: the port's `cmake/prepare-source.py` output from the pinned 4.6.8 archive. That is the six wmem units the
  port compiles, plus its `system.c`, `compat.c` and `scopes.c`.
- Flags: the port's Release flags, `-O3 -DNDEBUG`, gnu11, through the B0 pipeline. The data region is 40 MiB, with a
  32 MiB level0 arena.
- **Not `-fno-inline`, unlike memcached and the pool corpus.** A first build with it attributed case 22's fault (the
  second free of a double free) to the `static inline` CREVOKE helper instead of its declared `wmem_block_sublet_free`.
  wmem's probes are `noinline` functions with asm labels and do not need it.
- The level0 is set 4's (lane 02105a6f4c96): blocks of 4 KiB and up are placed on their granule and are exact
  children.
  - That is what lets the jumbo control fault. The jumbo is a block wmem hands to `g_free`, and its level0 lifetime is
    revoked there.
  - Before the change, a 2 MiB block at a 16-byte placement had no lifetime.

## The silicon-mimic QEMU reading

Instrument: capstone-qemu silicon/cderive-round ce1be0c642 with `CAPSTONE_CDERIVE_ROUND=1`, GP_FABRICATE=0, firmware
fw-b1-top d5c57ee765c9. Both arms' tool controls are as declared.

**wmem-silicon-wm-sublet:**
- 22 cases CAUGHT at the case's own probe (`wm_probe` or `wm_write_probe`): cause 24 for the stale reads and writes,
  5 / 7 for the crossings.
- Case 22 CAUGHT 24 in `wmem_block_sublet_free`, its declared site.
- Controls 00, 01 and 02 CAUGHT at `wm_probe` (24, 24, 5).

**wmem-silicon-wm-stock:**
- 23 DEFECT-REPRODUCED: the differential's VERDICT, with the stale chunk reoccupied.
- Control 01 CAUGHT 24 at `wm_probe`; controls 00 and 02 complete.

Every row matches the recorded virtual verdicts: virtual-malloc MISSED and virtual-nested-pools CAUGHT for all 23.

## The board prediction

Every row as QEMU read it, by outcome and site. Causes map as the earlier sets read them on this bitstream: QEMU's
5 / 7 read 28, and 24 reads 25 (ISSUES Q-11).

**New on silicon, and what a failure would say:**
- **wmem's pool reset at scale.** A reset revokes a block's generation and derives a new one, with every packet-scope
  object a child below it. The node pool holds 65,532 mints; a case that exhausted it would fault 30, not wedge.
- **A jumbo freed through g_free while its alias is still held** (control 01 on both arms). If 01 completes on the
  board, level0's free did not end the block's lifetime on silicon. Set 4's memcached rows depend on the same
  placement.
- **40 MiB domains,** within what sets 2 and 4 launched.
