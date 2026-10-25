# The nested-lifetime controls on silicon: results (2026-10-16, caplifive_sublet_d93a8d66b)

Pre-registered in README.md (lane 75ce44a1ceeb, before the boot). One boot, 08:26-08:52:
- firmware a43fcda381219db6, Image b90a6b8de26be0df, on the line of sets 2-5;
- six arms, 63 runs, every one ended by itself (each arm's rung returned 0).

**Every run is as pre-registered,** by outcome, cause and site, on the pre-registered image (sha256): 39 control runs
and the 24 tool controls.

| arm | permitted runs (parent, sibling, successor) | the successor's stale read |
|---|---|---|
| lt-silicon-pool-stock (FFmpeg pools stock) | 91, 92, 93 complete | completes |
| lt-silicon-pool-sublet (patch 0003) | 91, 92, 93 complete | **CAUGHT 25** at stale_read_probe |
| lt-silicon-mc-stock (memcached slabs and cache stock) | 91 page, 92, 93, 94, 95, 96 complete | 93 and 96 complete |
| lt-silicon-mc-sublet (patch 0006) | 91 page, **91 generation**, 92, 93, 94, 95, 96 complete | 93 and 96 **CAUGHT 25** at read_probe |
| lt-silicon-wm-stock (wmem as released) | block 00-02, block_fast 00-01 complete | block 03 and block_fast 02 complete |
| lt-silicon-wm-sublet (patch 0001) | block 00-02, block_fast 00-01 complete | block 03 and block_fast 02 **CAUGHT 25** at wm_probe |

**What it shows:**
- **On silicon, each port's lifetime patch leaves every permitted access alone.** That covers:
  - the allocator's own reference to the parent storage, after the child was released. This includes memcached's slab
    page and the generation that patch 0006 revokes chunks through, cache.c's list head, FFmpeg's entry data
    reference, and the wmem block;
  - a sibling allocated beside the released object, read and written with its contents intact;
  - the successor allocated at the released object's address, through its new pointer.
- **Only the stale reference faults,** with cause 25, at the labelled probe. On the stock arms the same stale read
  completes and reads the successor's data. So the rejection is the patch's, not the platform's.
- This is the false-positive half of sets 2, 4 and 5, which showed the patches reject the stale accesses of FFmpeg's
  pool, memcached's slabs and cache, and wmem.

Every stale read here is through a revoked child, so QEMU's 24 read 25 on silicon, as pre-registered (ISSUES Q-11).

**Judged by** `tools/run-silicon-cases.py board --prereg` on the run-scoped transcript, then compared row by row with
the pre-registered outcome, cause and site, keyed by image hash.
