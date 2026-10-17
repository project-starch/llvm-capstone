# M2's dependent chase on the Sublet bitstream: results (2026-10-17 03:22-05:08)

These are the readings for the plan in README.md: lane 7fc8751dc0e5, amended at 4ad24e19f5ce and f8b4bfde7f6a, all
before the boots. There were eight cold boots, m2f16-01 to -08, on `caplifive_sublet_d93a8d66b.bit` on the classic
line (d04bd83 / df0e9a8 / 93440d5):
- the image is the fixed-runtime R1 harness `66764b0e62075e57` (14 SHRINK words);
- `drivers/lists/f5-chase.txt` ran in list order, at most six invocations per boot;
- the emulator records are `/tmp/capstone/fixq-1016` (45 chase points per (arm, seed), every `chk_ok=1` and
  `cycle_ok=1`).
The result lines are in `results/`. The judge is `judge-m2.py`, and its output is `results/judge-m2.txt`.

**Every pre-registered check holds, in all eight boots:**
- 225 of 225 points, and 45 invocations;
- each point `chk_ok=1` with the emulator's checksum for its (arm, seed, N);
- 225 cold traversals, each `cycle_ok=1`;
- minted D and P 6, and S 2N+4;
- k800 4 twice in every boot;
- no refusal and no cause 25.
The judge was tested before the boots: it passes the emulator's own lines and fails a planted checksum, a planted
minted count and a missing control.

**Cycles per access, the median of 15 runs per point (5 repetitions × 3 seeds):**

| N | D | P | S | S − P | S/P | S's live nodes |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 10.003 | 18.002 | 18.002 | 0.000 | 1.0000 | 36 |
| 64 | 10.000 | 18.001 | 18.001 | 0.000 | 1.0000 | 132 |
| 256 | 10.002 | 18.002 | 29.954 | 11.952 | **1.6639** | 516 |
| 1024 | 41.812 | 61.384 | 99.101 | 37.717 | **1.6144** | 2,052 |
| 4096 | 49.252 | 88.454 | 123.518 | 35.064 | **1.3964** | 8,196 |

The spread (max − min over the 15 runs) is at most 0.003 cycles at N ≤ 256 for D and P, and 0.12 for S there. At
N ≥ 1024 it is up to 0.23 (D), 1.14 (P) and 2.35 (S). The minima and maxima are in `results/judge-m2.txt`.

**The reading is pre-registered outcome (a): S/P rises above 1.00.**
- **D and P read as on r30r31.** D is 10.00, then 41.84 / 49.25 there, against 41.81 / 49.25 here. P is 18.00 at
  N ≤ 256, then 61.37 / 88.45 there, against 61.38 / 88.45 here. So the data cache's rise at N ≥ 1024 is unchanged,
  and the chase itself costs the same on both bitstreams.
- **S departs from P, where r30r31 read S/P = 1.0000-1.0003.** On r30r31 the DYN unit queried only the lookup
  capability's node (Q-11), and the plain load's LSU check was dropped (R-34). Here each access through a record
  capability costs about 12 more cycles at N = 256 and about 35-38 more at N ≥ 1024.
- **The rise starts at N = 256, below the 2,048-node capacity the pre-registration took from §7t.** At N = 256, S has
  516 live nodes. The pre-registration expected the rise at N ≥ 1024, where S's 2,052 nodes first exceed 2,048. It
  also said the knee's position is the measurement.
- So on this bitstream something between 132 and 516 live nodes, or their data-cache footprint beside the records,
  already costs S per access. Which of the two it is, is not separated here.

**Where the knee comes from (added 2026-10-17, after the boots):**
- The LSU's revocation validity cache on this bitstream is 64 sets × 4 ways, 256 ids. Its set index is the node id's
  low 6 bits (capstone-ariane d93a8d66b, `core/load_store_unit.sv:258-260`: `RVC_SETS = 64`, `RVC_WAYS = 4`,
  `RVC_IDX_BITS = 6`). The 2,048 the pre-registration used was §7t's figure, not this cache's.
- S's 132 live ids at N = 64 fit, about two per set. Its 516 at N = 256 do not. That is where S/P leaves 1.000.
- The RTL lane's account of the cost: a hit is a same-cycle compare, so S equals P while the ids are resident. A miss
  is R-43's query-on-miss probe to the rev-node unit, which reads the node's 16-byte record through the D-cache: the
  +12 cycles at N = 256. At N ≥ 1024 that record read itself starts missing the D-cache. That account is the RTL
  lane's reading of the design. This run does not separate the two costs.

**Not established:**
- The knee's exact position between 132 and 516 live ids. This run has no point between N = 64 and 256.
- Why S/P falls from 1.66 to 1.40 as N grows. P's own data-cache cost rises faster than S's surcharge, so the ratio
  falls while S − P stays at 35-38 cycles.
- The comparison crosses the image, tree and toolchain as well as the silicon. r30r31's build9 is gone.
