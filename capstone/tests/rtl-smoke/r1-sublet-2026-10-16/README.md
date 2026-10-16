# R1 warm and cold on the Sublet bitstream: pre-registration (2026-10-16, before the boots)

The board campaign's wave 3, item 6 (`docs/plans/2026-10-16-board-campaign-all.md`, R1 warm 8 boots and cold 5
boots). It repeats R1's two silicon campaigns on `caplifive_sublet_d93a8d66b.bit`, each with its own image and list
unchanged:
- **R1 warm**, the release-to-reuse cost (measurements doc "R1 on silicon", boots sw8x-r1-b1…b8, 2026-09-21);
- **R1 cold**, the fit range (measurements doc "R1's fit range", 5 boots, 2026-09-22).
Both originals ran on `caplifive_m1_054cea69b`.

## What runs

`drivers/board-r1e4.sh`, as `../h1-sublet-2026-10-16`'s private copy (tree root at the main clone; each boot keeps
its `fw_payload.bin`):
- the classic line: d04bd83 / df0e9a8 / 93440d5, the #3 module, `REFUSAL_RECORD=1`, `PREFLIGHT_DTS_NODES=65536`;
- the readback host 2c9e82d101b48160 and the stock k800 control first and last.

| campaign | image (entry 0x410000) | list | boots | emulator record |
|---|---|---|---|---|
| warm | `1b7a04fe237e1580` | `drivers/lists/r1-invocations.txt`, 12 lines per boot (90 in all) | 8 | the R1 campaign's QEMU sweep, 84 `R1 s=` lines; pass record `qemu-pass/1b7a04fe…` |
| cold | `46f99c7b5bf2556e` | `drivers/lists/b3-cold-coefficient.txt`, one line per boot, reps 2-5 | 4 | `r1-b3b/qemu`, 12 `R1 s=wide` lines; pass record `qemu-pass/46f99c7b…` |

- **Cold rep 1** ran on this bitstream on the classic line already: the Sublet acceptance's a3, boot s10a3. So reps
  2-5 complete the five. Each repetition gets its own boot because one invocation mints 41,280 nodes (the list's
  README). Rep 1 ran with the relinked k800 589ceee3; reps 2-5 use the stock one. The control does not enter
  the measurement.
- **Warm boot 1** has not run on the classic line here. Thursday's a2 ran it on the exact-regions firmware. So all
  eight warm boots run.

## Pre-registered readings

**Per boot:**
- k800 returns 4 twice;
- every invocation ends with its `R1 end` line and `released pool rc=1`;
- every point record reads `ok=1 bad=0`;
- no refusal and no cause 25.

**The readings are the slopes, against 054cea69b's:**
- **warm:** REVOKE against nodes destroyed, 22.91 cycles/node shared and 15.92 combined there;
  - the four nulls (bytes, heap, depth, object size), with the fill-vs-bytes positive control at R² ≈ 1, as there;
- **cold:** the per-node constant across the node table's capacity, 19.66 at nd=64 rising to 28.15 at nd=16,384
  there.

**What this bitstream is expected to change:**
- H1's nodes regression on this bitstream, boot h1s16r1, already reads REVOKE lower than on R-43 v2 at every n:
  71 / 145 / 623 / 2,369 / 9,692 cycles for n = 1…256, against 145 / 193 / 724 / 2,880 / 11,274.
- So the warm slope is expected below 22.91. The size of the change is the result. The node record's parent link
  and CREVOKE in R-45's flush are this bitstream's revocation path.
- The cold constant and the ramp's shape are not predicted beyond that. The two campaigns together give the warm
  and cold per-node cost on this silicon.

**Not established here:** one bitstream; the same images as 2026-09-21/22, so the comparison is across silicon
only.
