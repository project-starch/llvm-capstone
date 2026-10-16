# P1 cells 5 and 6 beside the native -O2 baseline on the Sublet bitstream: results (2026-10-17 02:55-03:22)

These are the readings for the plan in README.md: lane 83ce50f019b4 and the meminfo amendment 5131d25ce2eb, both
pushed before the boots. There were two cold boots on `caplifive_sublet_d93a8d66b.bit`, on the classic line
d04bd83 / df0e9a8 / 93440d5:
- p1s16b-c5, firmware `9c49881855f19f6f`;
- p1s16b-c6, firmware `3bcbf9a7ad94655b`.

The result lines are in `results/`. The driver's summary says "cell 5" in both boots, but that is its label text: the
image hashes are the cells'.

**Every pre-registered identity reading holds in both boots:**
- k800 returns 4 twice;
- both arms read `112006 38bb59fd`;
- HEAP 2097152 on cell 5 and HEAP 1344064 on cell 6, each with DROPPED 0;
- lookasides 25,010 on the cell and 25,122 on the native;
- cell 6 prints `sublet: split=5568 mrev=37970 delin=32569 revoke=37970 init=5401`, and cell 5 prints no sublet line;
- each boot has one banner, and neither shows a HARD STOP.

| arm | image | cycles | predicted | CPI against its QEMU count |
|---|---|---:|---:|---:|
| cell 5 | `21f4b2659074511c` | 1,213,784,631 | 1,167,686,616 (the estimate) | 3.636 (333,794,832) |
| native -O2 warm, c5 boot | `9a80c1cd2a576ed2` | 856,244,108 | 856,160,837 ± 0.1 % | 3.557 (240,714,705) |
| cell 6 | `4bfe3b0ec8494385` | 1,292,707,481 | 1,292,584,693 ± 0.5 % | 3.757 (344,110,502) |
| native -O2 warm, c6 boot | `9a80c1cd2a576ed2` | 856,627,547 | 856,160,837 ± 0.1 % | 3.559 |

- **Cell 6 is within 0.01 % of Thursday's c6pair reading, the same image.** The two natives are within 0.01 % and
  0.06 % of the prediction, and within 0.045 % of each other.
- **Cell 5 reads 3.95 % above its estimate.** That estimate was the one number not taken from the same image. It
  multiplied today's instruction count by the CPI of the 09-30 cell 5 image `5f4b44257c933477`, which was 3.498.
  Today's image runs at CPI 3.636. It has the same CPDOMREQ but different code (1,443,880 bytes against 1,378,344),
  from another tree and toolchain. Its cycles are 4.9 % above that image's 1,156,968,951 on the same bitstream, for
  0.9 % more instructions.

**The ratios, each from one boot:**
- **cell 5 / native -O2 = 1.4176**, against the pre-registered ≈ 1.364. The difference is cell 5's CPI, above.
- **cell 6 / native -O2 = 1.5091**, against ≈ 1.510.
- **cell 6 / cell 5 = 1.0650** across the two boots, against ≈ 1.107, again because of cell 5's CPI. Normalised by each
  boot's own native, it is 1.0645.

**The board's idle memory (MemFree from `/proc/meminfo`, the virtual-profile plan's figure):**
- after Linux: 940,924 kB and 941,680 kB;
- after the native, at the end of the boot: 938,032 kB and 934,332 kB.

`meminfo` exits 1, because its loop's last test is false. The driver reads the BASELINE-PROBE line, not the status.

**Not established:**
- N = 1 per cell, at size 1 only, with none of P1's other arms;
- why today's cell 5 runs at a higher CPI than the 09-30 image. The code differs, and nothing here separates the
  code from placement.
