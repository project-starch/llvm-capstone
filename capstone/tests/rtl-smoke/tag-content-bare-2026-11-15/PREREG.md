# tag-content-bare-2026-11-15: pre-registration (written before any board run)

Image `images/tag-content.bin`, 32,680 bytes, sha256 273e77e9da57f4d5c96d0fde4fbc762af58b493ee1e10a80d55edc55232d192a,
board_rec 0x80007000, SR count 7. One boot per bitstream, the image alone, on whichever of these is resident:
F (a948e4134), F2 (eeb654104), G (9fd9d2530).

## The vector (image sims, sim13, dirty=no, `.sup/img{G,F,F2,PC}`, board-vector.py on the trace)

| pos | reading | expected on F, F2, G | mutant TAG_AUDIT_PC (2702f75d4; a store skips on any TWC hit) |
|---|---|---|---|
| 1 | tag bytes equal to expected | 0x400 (1,024) | 0x200 |
| 2 | tag bytes wrong | 0x0 | 0x200 |
| 3 | first wrong granule index | 0xffff (none) | 0x0 |
| 4 | its byte | 0x0 | 0x1 |
| 5 | ones read correctly | 0x101 (257) | 0x101 |
| 6 | page fills during the test (ev27) | 0x4 | 0x4 |
| 7 | traps | 0x0 | 0x0 |

All three trees read the expected column exactly, and the mutant reads 512 wrong. The same source reads identically
in the testlist runs, on G (overlap 4), at overlap 1 and under TB_MEM_PIPELINED. There the sim-only audit, a TWC-
independent shadow of the adapter's store stream checked against DRAM at test end, also reads 0 mismatches over 771
granules, against 512 under the mutant (sim13 `.sup/tc4`, `tc1`, `tc4p`, `tcpc`, `tcb*`).

## Grading (on the bitstream named by the boot's read-back)

- **VOID** if: the image hash differs, the SR count is not 7, the traps reading is not 0, or reading 5 is not 0x101
  (the raw path did not see tags as written).
- **P3 ungraded** (the fill arm did not create its condition) if reading 6 is not 0x4: the page was already marked at
  boot. Readings 1-3 are then reported, and the P3 granules' share of any wrong count is named from reading 3.
- **PASS** if readings 1-4 equal the expected column.
- **FAIL** on any wrong byte (reading 2 > 0). On silicon that is a stale shadow tag after a plain overwrite: a
  forgeable capability. It is surfaced at once, with the granule from readings 3-4 and its arm by index
  (0-255 P0, 256-511 P1, 512-767 P2, 768-1023 P3).

## What it cannot see

One shape set, one boot, and the TWC's in-flight windows as silicon's timing makes them. A PASS says these
overwrite shapes leave correct DRAM tags on the bitstream read. It says nothing about every program's interleavings.
The sim audit covers the 95-sweep and the tag test set on G and the F2-equivalent (results land in the twc-reach plan).
