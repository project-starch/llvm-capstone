# The nested-lifetime controls on silicon: pre-registration (2026-10-16, before the boot)

The bug corpus's nested sets (2, 4, 5) show that each port's lifetime patch rejects stale accesses on silicon. These
controls ask the other half: whether the patch leaves the PERMITTED accesses alone. They are
`docs/plans/2026-10-11-nested-lifetime-controls.md`'s parent, sibling and successor controls, in the real allocators,
driven through each allocator's own API:

| family | allocator | controls | runs |
|---|---|---|---|
| memcached allocator-repros/controls/lifetime | slabs.c | 91 parent page (and its generation on the Sublet arm), 92 sibling chunk, 93 successor chunk | `N VARIANT` |
| memcached allocator-repros/controls/lifetime-cache | cache.c | 94 the cache's list head, 95 sibling object, 96 successor object | `N VARIANT` |
| ffmpeg pool-repros/controls/nested-lifetime | AVBufferPool | 91 the entry's own data reference, 92 sibling entry, 93 successor entry | `run N`, `stale N` |
| wireshark wmem-repros/controls/lifetime | wmem block | 00 parent block, 01 sibling, 02 successor, 03 the stale reference | `0 N` |
| wireshark wmem-repros/controls/lifetime-block-fast | wmem block_fast | 00 parent block, 01 successor, 02 the stale reference | `0 N` |

**The prediction, as the plan registered it:**
- every run completes on both arms of every port, except the successor's stale read;
- on the Sublet arm, the stale read faults at the family's labelled probe;
- on the stock arm, the stale read completes, reading the successor's data.

The probes are `read_probe` (memcached), `stale_read_probe` (FFmpeg) and `wm_probe` (wmem, label `wm_defect_probe`).

## The arms and the boot

Six arms, each its port's Sublet or stock arm with exactly the build its corpus set used:
- set 2 for FFmpeg's pool: -O1 -fno-inline, the protected or traced FFmpeg tree, and `-DCTL_PROTECTED=1/0` as the
  controls' header requires;
- set 4 for memcached: patch 0006 with driver-controls.c, or the stock ledger;
- set 5 for wmem: the Release flags.

All six are on set 4's level0.

One boot on `caplifive_sublet_d93a8d66b`:
- private image b90a6b8de26be0df, firmware a43fcda381219db6, on the line of sets 2-5;
- order: lt-silicon-pool-stock, lt-silicon-pool-sublet, lt-silicon-mc-stock, lt-silicon-mc-sublet,
  lt-silicon-wm-stock, lt-silicon-wm-sublet;
- each arm runs the tool's four controls first.

## The silicon-mimic QEMU reading (capstone-qemu silicon/cderive-round ce1be0c642, `CAPSTONE_CDERIVE_ROUND=1`)

Every run is as predicted, on every arm, with the tool controls as declared:

| arm | permitted runs | the stale read |
|---|---|---|
| lt-silicon-pool-stock | 91, 92, 93 complete | 93 stale completes |
| lt-silicon-pool-sublet | 91, 92, 93 complete | 93 stale CAUGHT 24 at stale_read_probe |
| lt-silicon-mc-stock | 91 page, 92, 93, 94, 95, 96 complete | 93 and 96 stale complete |
| lt-silicon-mc-sublet | 91 page, 91 generation, 92, 93, 94, 95, 96 complete | 93 and 96 stale CAUGHT 24 at read_probe |
| lt-silicon-wm-stock | lifetime 00-02 and block-fast 00-01 complete | lifetime 03 and block-fast 02 complete |
| lt-silicon-wm-sublet | lifetime 00-02 and block-fast 00-01 complete | lifetime 03 and block-fast 02 CAUGHT 24 at wm_probe |

**The board prediction:** the same, with each stale read's 24 reading 25 on silicon. That is ISSUES Q-11's mechanism
holding exactly: every one of these is a read through a revoked child, which QEMU untags and the RTL keeps tagged.

**What a failure would say:**
- **A permitted run that faults on silicon** is the finding these controls exist for: the patch on this RTL rejects an
  access it must allow. The run's fault line names the access.
- **A stale read that completes on a Sublet arm** means the release did not end the child on silicon.

## Two corrections made before this pre-registration, on the QEMU stage

- **wmem's stale probe.** The family table first named `wm_ctl_load` as its probe. The controls' own headers say the
  stale read is at `wm_probe` (label `wm_defect_probe`), and the QEMU reading put it there. The table was corrected
  and the arms re-read: 7 of 7 each.
- **Missing build inputs.** The first build of memcached's cache family and FFmpeg's family failed to compile:
  - memcached's cache family missed the corpus's `shared/` include;
  - FFmpeg's family missed `CTL_PROTECTED`.
  Both were added and the families rebuilt. The slab family and wmem's built as they were. Their family-local
  `shared/` copies are byte-identical to the corpus's.

`tools/run-silicon-cases.py board` gains `--prereg`, so a board matrix is read against the committed prediction and
not the tool's own rule. The tool's rule is kept beside it as `predicted_tool`. Set 4's memcached Sublet matrix is
regenerated with it. Its 08 was pre-registered CONTROL-FAILED, which the tool's rule, taken from the virtual arm, had
shown as a miss. Nothing else in that matrix changes.
