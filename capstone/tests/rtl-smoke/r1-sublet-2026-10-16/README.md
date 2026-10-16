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
- So the warm slope is expected below 22.91. Amended before the first R1 boot: the prediction is taken from this
  bitstream's own point, not from a cause.
  - H1's nodes regression is R1 warm's shared series at nd = 512, on the same image family.
  - Its three boots read rv = 9,354-9,692 there, so 18.3-18.9 cycles per destroyed node.
  - The predicted warm shared slope is therefore about 18.5 cycles/node, against 22.91 on 054cea69b (whose own H1-shape
    point was 22.0 per node on v2).
  - The combined and cold slopes are reported, not predicted. The size of the change is the result. The node record's parent link
  and CREVOKE in R-45's flush are this bitstream's revocation path.
- The cold constant and the ramp's shape are not predicted beyond that. The two campaigns together give the warm
  and cold per-node cost on this silicon.

**Not established here:** one bitstream; the same images as 2026-09-21/22, so the comparison is across silicon
only.

## Amended after warm boot 1 (2026-10-16 22:40): at most six invocations per boot on this line

**Warm boot 1 (r1ws16-b1, 22:22-22:35) completed invocations 1-7 and stopped at the 8th.**
- What the 8th read: `SQ: D/mapped`, then `create_region failed -- above 4 MiB this needs a CMA area`, obs=65536, and
  the driver's HARD STOP.
- Invocations 1-7 read as pre-registered: 35 point lines, all ok=1 bad=0; 7 `R1 end` and 7 `released pool rc=1`;
  no refusal.
- **This is ISSUES R-55's region-table row, on the line this campaign uses.** Monitor 93440d5 decides head or tail
  on decoded bounds, so each invocation leaves about 11 slots instead of 6, and "R-43 a2 fills the 96-slot table at
  invocation 8 of 12". R-43 a2 is this list's lines 1-12. The exact-regions monitor completes them 12 of 12. This
  pre-registration should have counted that; the SLT one, written the same evening, did.
- The run stopped there, so warm boots 2-8 and cold reps 2-5 have not run.

**The rest runs in boots of at most six invocations, list order unchanged:**
- lines 8-90 in sixes from line 8: r1ws16s-01 is lines 8-13, r1ws16s-02 lines 14-19, …, r1ws16s-14 lines 86-90;
- then cold reps 2-5 as above (one invocation per boot).
- Each invocation is a fresh domain and a fresh arena, so lines 1-7 of boot 1 stand.
- What changes is the boot composition: six invocations before a point instead of up to eleven. The slopes are
  per invocation, and the composition is recorded beside them.
- The readings are as pre-registered above.

## Amended again (2026-10-16 23:20): the September images mis-carve on this bitstream; everything reruns on the fixed-runtime build

**The old images.** Every R1-harness image this folder used, 1b7a04fe (warm, M2) and 46f99c7b (cold), predates the
sublet-runtime fix a07a11194fa0. Neither contains SHRINK, the instruction that fix adds to `sublet_split`; fixed builds
carry it. Under this bitstream's full bounds scheme their carves alias (ISSUES R-55, the runtime half).
- M2 boots m2s16-b1 and -b2 wedged at their first chase invocation, arms D and S. Both read mcause 6, a store to
  address 1, at `sd a1,0(a2)` in `run_chase`'s identity loop: the second carve (`lookup`) lands on the first (`region`),
  and record 1's write turns `lookup[4]` into the integer 1.
- So R1 warm boot 1's seven invocations and the Sublet acceptance's cold rep 1 (a3) ran mis-carved images. They are
  not results here. H1's REVOKE finding was retracted for the same reason (dev 783fb0f93f7e).

**The image now: `66764b0e62075e57`.** It is the R1 harness rebuilt from this lane with the fix, with the default
defines, 0x410000 and -O1 (`../h1f-sublet-2026-10-16`), and 14 SHRINK words.
- Its emulator records (`/tmp/capstone/fixq-1016`): every distinct R1 warm invocation once (84 point lines, as the
  campaign's sweep), the five cold reps (48 points) and M2's nine (arm, seed) pairs (45 points, every `chk_ok=1` and
  `cycle_ok=1`).
- Seed 1's checksums equal the ones pre-registered above.

**The boots, at most six invocations each, list order unchanged:**
- M2: m2f16-01..08 (45 invocations);
- R1 warm: r1wf16-01..15 (90);
- R1 cold: r1cf16r1..r5, all five reps, because rep 1 ran an unfixed image.
They run after SLT, P1b and H1F, before the size-100 boots.

**What changes in the readings:**
- the image differs from 054cea69b's and r30r31's, so every comparison crosses tree, toolchain and silicon;
- R1's warm slope is reported, not predicted from H1, whose REVOKE reading is retracted;
- M2's two outcomes, (a) S/P above 1.00 at N >= 1024 and (b) S/P = 1.000, stand as written.
