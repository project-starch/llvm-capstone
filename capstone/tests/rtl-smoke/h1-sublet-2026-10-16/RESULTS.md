# H1 calibration on the Sublet bitstream: results (2026-10-16)

Pre-registered in README.md (lane 21e38145c053, before the boots). Three cold boots on `caplifive_sublet_d93a8d66b.bit`,
classic line, each running the whole `h1-v2-calibration.txt` list on image da9c54e578845735:
- h1s16r1, 19:36-19:49;
- h1s16r2, 19:50-20:03;
- h1s16r3, 20:04-20:17.
The result lines are in `results/`. Each boot kept its `fw_payload.bin`.

**Every pre-registered reading holds, with one exception, a jitter figure in boot 3.**
- k800 returned 4 twice in every boot, with instret 1089.
- There were no faults and no refusals.

| reading | 2026-09-29, R-43 v2 (3 boots) | d93a8d66b (3 boots) |
|---|---|---|
| latency per load, 4 / 16 / 64 / 256 / 1024 KiB | 9 / 9 / 40 / 48 / 48 | **9 / 9 / 40 / 48 / 48**, every boot |
| timer `cyc_cyc` / `cyc_cyc2` / `ret_ret` | 2 / 2 / 1 | **2 / 2 / 1**, all nine domains |
| nodes regression | nd = 2n, ok=1 bad=0 | **nd = 2n, ok=1 bad=0**, every n, every boot |
| linear types, arms 0-11 | 7,1,7,7,7,1,7,1,0,1,0,1 | **7,1,7,7,7,1,7,1,0,1,0,1**, every boot |

**The exception: traversal agreement at 64 KiB in boot 3.**
- Its three traversals read 2,674,716 / 2,671,550 / 2,679,231 cycles, a spread of 0.117 cycles per load against the
  pre-registered 0.1.
- Boots 1 and 2 spread 0.062 and 0.008 there, and every other size in every boot stays at or below 0.068.
- The per-load value is 40 in all three traversals, as in the other boots.
- So it is a wider band at the one size that straddles the cache, not a moved figure. Reported as a new reading,
  as pre-registered.

## The finding: revocation is cheaper on this bitstream, at every node count

REVOKE (`rv`) in the nodes regression, cycles:

| n (nd = 2n) | R-43 v2, 2026-09-29 | d93a8d66b, boots 1 / 2 / 3 | median ratio |
|---:|---:|---:|---:|
| 1 | 144-146 | 71 / 71 / 71 | 0.49 |
| 4 | 193 | 145 / 145 / 146 | 0.75 |
| 16 | 722-725 | 623 / 634 / 608 | 0.86 |
| 64 | 2,873-2,897 | 2,369 / 2,336 / 2,370 | 0.82 |
| 256 | 11,106-11,293 | 9,692 / 9,354 / 9,490 | 0.84 |

- At n = 256, 512 nodes destroyed, that is about 18.5 cycles per node against about 22.0 on v2.
- The fixed part halves: n = 1 drops from about 145 to 71.
- The three boots agree within 4.3 % at every n (n = 16: 608 to 634).
- This bitstream's revocation path carries the Sublet node record's parent link and CREVOKE in R-45's flush. What
  changed it is not established here. The R1 warm and cold campaigns pre-registered next
  (`../r1-sublet-2026-10-16`) measure the per-node slope across the node table's range.

**Not established:** N = 3, as 2026-09-29.
