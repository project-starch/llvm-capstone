# FFmpeg app safety results, hardened: N = 3 per cell, the missing control, a second workload (QEMU)

**Why this exists.** The first safety results (`../2026-09-23-qemu-safety/`,
`../2026-09-24-qemu-pool-safety/`) had three weak spots, named by their audits:
- every fixture cell was N = 1;
- the pool arms' "one byte below a refstruct object faults" had no stock-build control, so it
  showed exact bounds, not where the header lives;
- there was one 1-second workload.

This folder closes all three. It is QEMU only, with the caveats of those folders (ISSUES Q-11;
fabricated `gp`).

**Verdict.**

1. **Every fixture cell is N = 3, and every counted run is as pre-registered.**
   - **Cells:** 47 in all:
     - fixtures 1–10 on level0, shrink and sublet;
     - fixture 16 on those three arms;
     - fixtures 11–17 on pool0 and pool2.
   - **Runs:** 141 counted runs, 0 disagreements. Each cell's three runs are distinct boots on one
     fixture image and one host binary (hash sidecars).
   - **Predictions:** unchanged since they were pushed. `456fa82` holds the pool arms'
     predictions and `411732f` fixture 16's on the heap arms, each pushed before those runs.
     Fixtures 1–10's predictions were first committed AFTER their first counted runs; for those
     30 first runs the pre-registration rests on the session record
     (`../2026-09-23-qemu-safety/`). Every other run came after pushed predictions: all runs of
     this round, every pool run, and every fixture-16 run.
2. **The refstruct header contrast is now measured.**
   - **shrink and sublet:** stock refstruct allocates `[RefCount | object]` as one `av_malloc`
     block. Fixture 16's pointer keeps that block's bounds, starting **64 bytes below the
     object** (e.g. `[…840, …8c0)` with the cursor at `…880`). So `p[-1]`, the last byte of the
     64-byte `RefCount` (its `free` field: `refstruct.o` stores at +0, +0x10, +0x20 and +0x30,
     and returns buf+0x40), is readable. It returned 0x00, 3/3 on each arm.
   - **level0:** the pointer carries the whole arena, so its 3/3 returns say nothing about
     layout.
   - **pool0 and pool2:** the same read faults on bounds, 3/3 on each. That shows the header is
     not under the object's capability. That it lives OUT OF BAND is known from how patch 0002
     is built; this fault does not measure it.
3. **A second workload holds.** A 5-second clip at 320x180 (150 frames) gives **150/150 frames
   bit-identical to native, with the flipped control firing**, on level0, on the revoking
   `sublet` heap, and with FFmpeg's pools leased (`pool2`). The flipped input changes frames
   79–83. On every arm the domain's flipped output also equals NATIVE's flipped output, hash for
   hash (150/150), so the whole corrupted clip was decoded the same way. The pools reuse their blocks: the
   payload stays at 315,072 bytes, the same as for 1 second, while the leases grow with the work
   (1,846 taken and revoked).
4. **The revocation-node spend scales with the work.** A pool2 decode of 150 frames spends
   **6,236 nodes** in the domain's heap and pool primitives:
   - heap: split 753 + mrev 3,398;
   - pools: split 43 + mrev 2,042.

   That is about 41.6 per frame on average, and about 36 per extra frame (the 30-frame decode
   spent 1,900).
   - **Non-reclaiming silicon** (bitstreams before 2026-09-17: 65,532 per boot, ISSUES R-12):
     about 1,600–1,800 frames, roughly a minute of this clip, before counting the host's and the
     monitor's own node use.
   - **The resident bitstream (`054cea69b`)** reclaims the nodes a revoke walk invalidates, but
     never a handle's own node (R-12). So its ceiling for this workload depends on a leak
     fraction that has NOT been measured.
   - QEMU reuses nodes and shows neither case.

## Method

- **Repeats.** Two more full repeats (a, b) of every cell on the current images, plus a third
  run (c) of fixture 16 on the heap arms.
  - **Earlier counts:** the counted run of each cell from the earlier folders is the first of
    the three.
  - **Boots:** one predicted fault per boot, last. Images load from the guest's `/tmp`.
  - **Logs:** every attempt is kept. 17 fixture sections, from 9 boots, came from attempts that
    stalled before any fixture ran:
    - 7 before the guest's login;
    - 1 at login;
    - 1 in the copy from the 9p share, after login.

    Each is listed as "no section" in `result-lines.txt` and was retried; no stalled attempt is
    counted.
- **Images.** The repeats ran on byte-identical images, checked against the earlier boots'
  hash sidecars. Fixture 7's first counted run is round 2's, because its image was rebuilt after
  round 1; the repeats use that round-2 image on every arm.
  The FFmpeg libraries and runtime are unchanged: every default arm rebuilds byte-identical, and
  the level0 run of record verifies 7/7.
- **The second workload** is `FFAPP_CLIP_SECONDS=5` (`host/build-native.sh`,
  `host/build-domain.sh`, `host/run-qemu.sh`).
  - **Separate files:** it gets its own input, reference and image directories
    (`input-5s.mkv`, `stock-5s.framemd5`, `domain…-5s`), so the 1-second input, reference and
    images are untouched. Shared scratch files, such as `domain-m5.out` and the share directory,
    are overwritten by whichever run came last.
  - **Native check:** the native build of the same decode core matched stock on it (150/150),
    and the flip control changed 5 hashes, before the domain runs.
  - **Guest budget:** raised (`CAPSTONE_GUEST_COMMAND_TIMEOUT=3000`) for the TCG decode.

## What this still does not establish

1. **The board.** Temporal faults are QEMU's (Q-11), and `gp` is fabricated. On the resident
   bitstream the node ceiling depends on an unmeasured leak fraction (verdict item 4).
2. **Performance.** QEMU timing means nothing.
3. **Other codecs, resolutions or containers.** Two clips of one kind; 640x360 does not fit
   the domain's block.

## Files

- `result-lines.txt`: every repeat attempt's verdict; the fixture-16 control readings; the
  5-second M5 counters and oracle lines.
- `SHA256SUMS`: the 5-second reference, input and flipped input; the images and guest hosts of
  all three 5-second arms; and the heap arms' fixture-16 images. The other images are hashed in
  the earlier folders and were verified unchanged.
- **Only one 5-second arm's booted copy was hash-checked.** The 5-second boots wrote no per-boot
  hash sidecar, and only pool2's booted share copy survives; it matches 7/7.
