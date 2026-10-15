# Lifetime controls on block_fast: parent, successor (sibling not applicable)

These are not cases. They are Wireshark's row of the pre-registration
`docs/plans/2026-10-12-nested-lifetime-controls-extended.md`. They drive wmem's real `block_fast`
allocator through its own API: a fresh `wmem_allocator_new(WMEM_ALLOCATOR_BLOCK_FAST)`,
`wmem_alloc`, and `wmem_free_all`. block_fast's `wmem_free` is a no-op on both builds, so its only
release is the reset. Patch 0001 makes the reset revoke the first block's generation (CREVOKE from
`life`) and derive a new one (`wmem_sublet_block_renew`).

| program | what it does | predicted, protected | predicted, release |
|---|---|---|---|
| `00_control_parent_block` | allocates A (64 bytes) and a second object, resets, then reads and writes A's former range through the allocator's own block reference; on the protected build, also through the new `generation` and through `life` | completes; the read returns what the allocator left | completes |
| `01_control_successor_reuse` | allocates A and a second object, resets, allocates at A's size until wmem returns A's address as C (printed, and required to be equal), then writes and reads C | C == A; completes | C == A; completes |
| `02_control_successor_stale` | 01's sequence exactly, then reads A's whole former range through the OLD A pointer at the corpus's labelled read (`wm_probe` / `wm_defect_probe`), after the ready mark | **fault, cause 25, at `wm_defect_probe`** | completes and reads C's bytes |

There is no sibling program. The reset ends every object at once, and no release leaves a sibling
live, so the sibling control is reported as not applicable.

`blockfast.h` says which reference is "the parent" and restates block_fast's layouts. The
labelled accesses (`wm_ctl_load` / `wm_ctl_store`), the patterns, the printed lines and the build
check (`wm_ctl_arm`, exit 75 on a mismatch) come from `../lifetime/lifetime.h`, included unchanged.
Control 00 checks first what the allocator left after the reset, reading through the block:

* `pos` is rewound to the block header size;
* A's chunk header still holds A's length;
* A's 64 bytes still hold A's pattern.

## Build and run

Build exactly as `../lifetime/README.md` says, with `-DWM_CORPUS_DIR=<this directory>`.
`run-lifetime-controls.py` is that directory's runner, with the judge and the recorded
configuration changed for block_fast. It runs the three programs on one arm in one boot of the
virtual platform. The record of 2026-10-12 is
[`../../results/2026-10-12-controls-ext/`](../../results/2026-10-12-controls-ext/README.md).

This directory adds no case to the corpus. The CMake glob and `shared/run-defects.py` scan only
`NN_*` directories, so neither reaches `controls/lifetime-block-fast/`.
