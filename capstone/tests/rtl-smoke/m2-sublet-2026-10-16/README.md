# M2's dependent chase on the Sublet bitstream: pre-registration (2026-10-16, before the boots)

The board campaign's wave 3, item 8 (`docs/plans/2026-10-16-board-campaign-all.md`, M2 dependent chase). It repeats
measurements doc §7y (F5, boots sw8x-r1f5-b1…b4, 2026-09-15, `caplifive_r30r31_1bfff7776`) on
`caplifive_sublet_d93a8d66b.bit`.

## What runs

- `drivers/board-r1e4.sh` over `drivers/lists/f5-chase.txt`: 45 invocations in 4 boots (12 / 12 / 12 / 9), one
  (arm, seed) each. At most four Sublet invocations per boot, for the node budget.
- Arms:
  - custom-spatial P: one alias per record, 6 live nodes;
  - custom-sublet S: a leaf and an alias per record, 2N+4 live nodes;
  - data-only D: integer chase on one base, no lookup.
- Working sets of 16 / 64 / 256 / 1024 / 4096 records; 100,000 timed dependent accesses per point after two warm-ups.
- **The image is the R1 harness 1b7a04fe237e1580** (entry 0x410000), the R1 warm campaign's, which carries the chase
  series. §7y's build9 (fdd3029ff0f96680) is no longer on disk, and its manifest records no hash.
- The private driver copy as `../h1-sublet-2026-10-16` (tree root at the main clone, `fw_payload.bin` kept), the stock
  k800, host 2c9e82d101b48160, the classic line, `REFUSAL_RECORD=1`, `PREFLIGHT_DTS_NODES=65536`.

## The emulator's oracle (main clone's capstone-qemu deb7d75756, `qemu/`, `qemu-chase-combined.log`)

One run per (arm, seed), all five working sets, every point `chk_ok=1` and every cold traversal `cycle_ok=1`:
- checksum per working set, seed 1: 750,000 / 3,150,039 / 12,750,873 / 51,154,965 / 204,826,544, the same in all three
  arms;
- seeds 2 and 3 have their own values (in the log);
- minted nodes: D and P 6; S 2N+4 (36 … 8,196).

## Pre-registered readings

**Per boot:**
- k800 returns 4 twice;
- every invocation ends `R1_RC=0`;
- every point `chk_ok=1` with the emulator's checksum for its (seed, N), every cold traversal `cycle_ok=1`;
- minted as above;
- no refusal and no cause 25.

**The reading is cycles per access, per arm and N, median over the runs.** On r30r31 the result was:
- P and S equal at every N: 18.00 at N ≤ 256, 61.37 / 61.39 at 1024, 88.45 / 88.44 at 4096;
- S/P = 1.0000-1.0003. That held because the DYN unit queried only the lookup capability's node (Q-11), and the plain
  load's LSU check was dropped (R-34);
- D: 10.00, then 41.84 and 49.25.
The rise at N ≥ 1024 is the data cache's, common to all three arms.

**This bitstream decides between two outcomes:**
- **(a) S/P rises above 1.00 at N ≥ 1024.** P and D rise there too, with the data cache, as on r30r31.
  - This is what happens if each plain load through a record capability now queries that record's node.
  - The Sublet bare floor read every load through a revoked capability refused with 25 (2026-10-16), which implies the
    check is no longer dropped.
  - S then has 2N+4 live nodes, and from N = 1024 (2,052 nodes) that exceeds the node cache's 2,048-node capacity. P's
    record capabilities share 6 nodes.
- **(b) S/P = 1.000 at every N.** The access path still queries one node per access, as on r30r31.
- Either reading is a result. D is the control for the plain chase without a lookup: about 10 cycles per access at
  N ≤ 256, and r30r31's 41.84 / 49.25 beyond, are expected.

**Not established:** the node cache's capacity on this bitstream is taken from §7t's 2,048. The knee's position, if any,
is the measurement.

## Amended while boot 1 ran (2026-10-16 22:40): what a boot cannot reach runs in boots of at most six

R1 warm boot 1, on the same image, host and line, stopped at its 8th invocation (ISSUES R-55's region table; see
`../r1-sublet-2026-10-16`). So each 12-invocation boot here is expected to end the same way.
- The boots run as listed.
- Every invocation past a boot's last `R1 end` line reruns afterwards, in boots m2s16t-01… of at most six, in list
  order.
- The readings are as pre-registered above, and the boot composition is recorded beside them.

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
