# H1 on the Sublet bitstream with the fixed Sublet runtime: results (2026-10-17 02:12-02:53)

Pre-registered in README.md (lane 3e021fae4070, before the boots). Three cold boots, h1fs16r1..r3, on
`caplifive_sublet_d93a8d66b.bit`. The image is `66764b0e62075e57`, with 14 SHRINK words; it ran H1's list on the
classic line. Result lines: `results/`.

**Outcome (b): REVOKE reads as H1 read it on the old image, and not as R-43 v2. The drop does not come from the
unfixed Sublet carve.**

| n (nd = 2n) | H1F boots 1 / 2 / 3 | H1, old image `da9c54e5` | R-43 v2 | H1F median / v2 |
|---:|---:|---:|---:|---:|
| 1 | 68 / 92 / 64 | 71 / 71 / 71 | 144-146 | 0.47 |
| 4 | 145 / 170 / 183 | 145 / 145 / 146 | 193 | 0.88 |
| 16 | 652 / 621 / 616 | 623 / 634 / 608 | 722-725 | 0.86 |
| 64 | 2,437 / 2,434 / 2,399 | 2,369 / 2,336 / 2,370 | 2,873-2,897 | 0.84 |
| 256 | 9,523 / 9,432 / 9,460 | 9,692 / 9,354 / 9,490 | 11,106-11,293 | 0.84 |

- **At n ≥ 16, where the per-node work dominates,** the fixed image reads within the old image's spread: 0.84-0.86 of
  v2, against H1's 0.82-0.86.
- **The fixed part (n = 1) halves again.**
- **n = 4 varies more on this image** (145-183, against H1's 145-146), and lies between the two outcomes. It is
  recorded, not explained.

**Every other pre-registered reading holds in all three boots:**
- latency 9 / 9 / 40 / 48 / 48;
- `cyc_cyc=2 cyc_cyc2=2 ret_ret=1` in all nine calibration domains;
- nd = 2n with ok=1 bad=0;
- linear types 7,1,7,7,7,1,7,1,0,1,0,1;
- k800 4 twice, instret 1089;
- no fault and no refusal.
- The 64 KiB traversal spreads read 0.065, 0.044 and 0.039 cycles per load, inside the 0.1 band.

**What this settles, and what it does not:**
- The retraction on dev (783fb0f93f7e) withdrew the drop's attribution, because H1's image carried the unfixed carve.
  A fixed-runtime image now reads the same drop. So the drop is a property of this bitstream and its classic-line
  monitor, not of the image's runtime.
- Which of the two causes it is not established here:
  - against R-43 v2 the bitstream differs (Sublet, with the node record's parent link and CREVOKE in R-45's flush), and
    so does the monitor (93440d5 against 2dcd3a5);
  - the RTL lane's diff of the rev-node unit finds the same three queries per node.
- The tree and toolchain differ from H1's image too (dev147 on today's lane against the #119 build). The REVOKE
  bracket is the instruction between two `mcycle` reads, so they reach it only through placement.

**Not established:** N = 3, as H1.
