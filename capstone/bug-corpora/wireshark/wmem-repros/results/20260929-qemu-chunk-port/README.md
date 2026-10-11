# The wmem corpus against the chunk port (capstone-qemu, 2026-09-29)

**Question.** On 2026-09-21 the region-granular port caught 12 of the 13 defects (`../20260921-qemu/`).
Case 12 was the recorded non-detection: a lifetime ended by an individual `wmem_free` into the block
allocator's recycler, inside a live block. The chunk port (`a34caaedb1bc`) gives every chunk a
region of its own. Does the corpus now come out 13 of 13, and is case 12's new fault the chunk
port's?

**Answer: yes, as registered in `PREDICTIONS.md` (`60491a5b4ad8`).**

| | 2026-09-21, region-granular | now, chunk port |
|---|:--:|:--:|
| **Sublet caught** | 12 / 13 | **13 / 13**, each at the labelled read probe, cause 24 |
| spatial (no protection) completes | 13 / 13 | 13 / 13 |

| case 12 | Sublet | spatial |
|---|---|---|
| **chunk build** (`WM_CHUNKS=ON`) | **FAULT** at the labelled read probe ×3 | completes |
| **control build** (`WM_CHUNKS=OFF`, one option apart) | completes ×3, the old non-detection | completes |

That pair is the attribution. It is the same source and case, one build option apart, in the same
guest, and the option decides the outcome. The option compiles the chunk port in or out, so the two
images differ.

The control build's protection is shown to be live by its own positive control, run after the
claim audit: case 0 in Sublet mode on the `WM_CHUNKS=OFF` build faults at its read probe. So case
12 completing there is the region-granular port's non-detection, not an unprotected build.

Cases 0-11 are unchanged, and were expected to be. Their pool is the packet pool (BLOCK_FAST),
which the chunk port leaves as it was.

The **negative control** writes an input record whose count disagrees with its length. **26 of 26
arms FAILED**, as they must:

- 23 by the guest-side loader refusing the record (exit 3) before any domain is created;
- 3 Sublet arms (cases 0, 6 and 8) by a boot stall (exit 75), which fails the same way.

So a run that never reaches the case cannot pass. That control does not exercise the probe-address
comparison, so the comparison was tested separately, offline, on the real fault logs:

- the 15 passing fault arms, re-judged against the WRITE or ALLOCATOR probe instead of the read
  probe, **all FAIL**;
- against the read probe they all pass.

The 2026-09-21 negative control has the same mechanism, and its README described it wrongly; that
is retracted on dev in `92e05e7e0c34`.

## Found and fixed before the run (both in `60491a5b4ad8`)

1. **The corpus did not build on dev.** The chunk port had added a `wm_probe` to the port, and the
   corpus seam's driver defines its own `wm_probe`, the read probe every oracle names. The link
   failed on the duplicate. The port's symbol is now `wm_handback_probe`.
   - The chunk port's own verification never built the corpus, which is how this went unseen from
     `a34caaedb1bc` until now.
   - The labelled fault site `wm_widen_probe` is unchanged.
   - The chunk port's recorded results stand for the images they hashed. A rebuild after the rename
     hashes differently, because symbol names change.
2. **The runner judged every build against the region-granular oracle.**
   - Each `case.json` now has a `sublet-chunks` oracle beside `sublet`.
   - `shared/run-defects.py` picks between them from the domain build's `WM_CHUNKS`, so a new
     detection cannot read as a failure, nor a lost one as a pass.
   - A build with no `CMakeCache.txt` is refused. A cache without `WM_CHUNKS` is judged as
     region-granular, since the option did not exist before the chunk port.
   - Every protected row of `matrix.tsv` names the oracle that judged it (`oracle_arm`).
   - Tested on the four build states. The contract checker accepts the new arm and still rejects
     its four corruptions.

## How it ran

- **Builds.** Presets `capstone-domain` and `linux-guest` from `60491a5b4ad8`, compiler `3979abd8`,
  `-DWM_CORPUS_DIR` pointing at this corpus. The control is the same tree with `WM_CHUNKS=OFF`.
- **Guests.** They boot a private copy of the rootfs, repaired with `e2fsck`, because the shared
  one carries ext4 errors. Kernel and firmware are the shared ones.
- **Native control.** The native, unprotected arm with the chunk port was run before the
  registration: all 13 complete. That established that case 12's own check, that the next
  allocation lands on the freed chunk's address, holds under the chunk port.
- **Infrastructure.** Boot stalls (runner exit 75, serial logs ending in the firmware banner or at
  init):
  - three Sublet boots of the first pass (cases 3, 6 and 11), re-run, all three as predicted;
  - three negative-control Sublet boots (cases 0, 6 and 8), not re-run, since they fail as a
    refusal does.
  - Every row is kept in `matrix.tsv`.
- **N.** 1 per cell, as the corpus's runs of record, and 3 for case 12 on both builds. The emulator
  is not run under `-icount`, so repeats reproduce the outcome but are not bit-identical: some
  boots stall on identical inputs.

## What this does not establish

1. **QEMU only** (Q-11): on silicon a stale access retires.
2. **The cases are reductions.** The consumers are reduced; the allocator is wmem's own, unmodified
   but for the port's hooks.
3. **None of the 13 defects is live at the 4.6.8 pin.** Each `case.json`'s `live_proof` quotes the
   fixed code. The cases are the upstream defects' shapes, run against the pinned allocator.

## Files

- `PREDICTIONS.md`: the registration.
- `matrix.tsv`: one row per arm, from every run in the order run, with the run named.
- `inputs.json`: per run, the image and tool hashes.
