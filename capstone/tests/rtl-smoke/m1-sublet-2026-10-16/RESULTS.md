# M1's series on the Sublet bitstream: results (2026-10-17)

The pre-registration and its amendment are in `README.md`. Every boot ran on `caplifive_sublet_d93a8d66b.bit`, on the
classic line (buildroot d04bd83, monitor 93440d5) with `REFUSAL_RECORD=1`, on the fixed-runtime images the amendment
names. The result lines come from `../m1v2-2026-10-01/extract.py`, the 2026-10-01 campaign's extractor, unchanged.
`judge-m1f.py` computes the per-snapshot checks and the per-operation costs (`take_cyc/n`, `give_cyc/n`).

## m1s16f-1: drop and ring to 10C, image `66764b0e` (09:15-09:26)

- **k800** read 4 twice.
- **Drop and ring both reached `stop=target alloc=655320`**, against the 65,532-index pool, with no exhaustion:
  - drop: minted 655,351, revoked 655,320;
  - ring: the same, retained 16.
- **159 snapshots per arm, `minted - revoked = 31` on every one.**
- No wedge: the run returned.

## m1s16f-5: pressure to the buffer bound, then release, image `74263f41` (09:26-09:36)

- **k800** read 4 twice.
- **Pressure:** `stop=buffer alloc=43297`, the pre-registered clean null; retained 43,296.
- **Release** entered phase 2 at 43,290 (`released arm=release alloc=43290`) and ended at `stop=target alloc=51948`.
- **201 snapshots** (pressure 10, release 191), `minted - revoked = 31` on every one.

## Cost, against v2's readings (reported, not predicted, after the amendment)

| arm | take per op, this run | v2 | give per op, this run | v2 |
|---|---|---|---|---|
| drop | 94.347 [94.264, 95.010] | 72.029 | 106.046 [105.961, 106.472] | 107.057 |
| ring | 88.000 [87.978, 88.001] | — | 75.056 [75.033, 75.832] | — (+4.4 on v2, unproven at N = 1) |
| pressure | 88.000 [88.000, 88.001] | 72.010 | 75.053 [75.019, 76.613] | 75.066 |
| release, all snapshots | 88.000 [88.000, 130.015] | — | 75.579 [75.339, 97.185] | 75.782 (phase 1) |

Medians over the arm's snapshots, [min, max].

- **Take rose 16 cycles on pressure and 22 on drop; give is unchanged to about 1 cycle.**
- **The image changed:** the fixed-runtime images carry a07a11194fa0's `sublet_split` SHRINK and
  cursor-based take. The bitstream changed as well. These boots do not separate the two.
- For reference, v2's bridge boots read take unchanged from 054cea69b to v2 on identical images (72.029 to 72.029).
  This rise's attribution is open.

## The stale-operation rows, image `b77b35e5` (probe build)

Each row ran three times. The first two were refused before booting by `board-r1e4-main.sh`'s live-runner check, at
09:36:41 and 09:50:42: `pgrep -f 'python3 -m fpga_driver'` matched my own background launchers, whose command lines
carried a run script's text containing that pattern. No board time was spent. The third launch, bare, ran.

### m1s16f-2: the four Design arms, then ONE stale READ at age 0, LAST (09:54-10:14)

- **The four Design arms**, the same as boots 1 and 5:
  - drop and ring to `stop=target alloc=655320`;
  - pressure to `stop=buffer alloc=43297`;
  - release ending at 51,948 after phase 2 at 43,290.
- **Probes through live aliases succeed before the last arm:**
  - `probe read ok live=1 is_live_data=1` on ring, and on pressure at age 0 and at age 2;
  - `probe write ok live=1 via_live_alias=165 wrote=165`.
- **The stale READ at age 0** is refused at the probe's own instruction:
  - mcause 25, mepc − DBAS = **image+0x4938**, as pre-registered;
  - no `probe read ok live=0` line.
- **The refusal record reads LATCHED:** revnode id 0x065800e7, generation 1624 at index 231. Its parity is ok.
- **k800** read 4 once, before the stale arm, which ends the domain by design.

### m1s16f-3: the four Design arms, then ONE stale WRITE at age 0, LAST (10:14-10:34)

- **The four Design arms and the live-alias probes:** as m1s16f-2, line for line. That includes the live write
  `via_live_alias=165 wrote=165`.
- **The stale WRITE at age 0** is refused at the probe's own instruction:
  - mcause 25, mepc − DBAS = **image+0x499c**, as pre-registered;
  - no `probe write ok live=0` line.
- **The refusal record reads LATCHED,** with the same id as m1s16f-2: 0x065800e7, generation 1624 at index 231. The two
  boots run the same deterministic history up to their last arm.
- **k800** read 4 once, before the stale arm.

### m1s16f-4: the four Design arms, then ONE stale READ at age 2, LAST (10:35-10:55)

- **The four Design arms and the live-alias probes:** as m1s16f-2.
- **The stale READ at age 2** is refused at the probe's own instruction:
  - mcause 25, mepc − DBAS = **image+0x4938**, as pre-registered;
  - tval[9:0] = 0x3c0, slot 15 of the 64-byte leaves;
  - no `probe read ok live=0` line.
- **The refusal record reads LATCHED:** revnode id 0x05480101, **generation 1352 at index 257**. Head is 258 and
  serving 242.
- **Observed, not predicted:**
  - the age-2 row's refused reference sits at index = head − 1, with serving = head − 16, at generation 1,352;
  - those are the same generation and the same two relations as R-43 v2's m1v2s-4 (generation 1,352 at index 136,
    head 137);
  - only B differs: head 258 here, 137 there.
  - The turnover witness was deferred because this bitstream's node ids are not modelled, so this is a reading. It
    is not a confirmation of a model.
- **k800** read 4 once, before the stale arm.

## Summary

**Every pre-registered reading holds on all five rows:**
- permitted reuse to 10C with no exhaustion;
- `minted − revoked = 31` on every snapshot;
- the three stale accesses refused with cause 25 at the predicted instruction, with live aliases working and the
  record LATCHED;
- pressure's clean null and the release phases at 43,290 and 51,948.

**Take rose 16-22 cycles per operation against v2, with give unchanged.** The image and the bitstream both changed,
and these boots do not separate the two.

**The probe image's costs** (`results/judge.txt`) are identical across its three boots to within the band:
- take per operation: drop 96.0, ring 89.0, pressure 89.0, release 89.0;
- give per operation: drop 165.3, ring 137.7 / 124.8 / 124.8, pressure 139.1, release 139.1-139.3.

v2's probe image (eb3ed1e3) recorded a "mint cost per invocation" of drop 96.0, ring 73.0, pressure 84.0, release 73.0.
Whether that bracket is this take bracket is not established here, so no comparison is drawn. The probe build's
brackets are larger than the bridge images' (give 165 against 106 on drop), which is consistent with its
instrumentation (`M1_PROBE_RT`).

`results/` holds each boot's result lines, the snapshots excepted (`judge.txt` reduces those), and `judge-m1f.py`.
