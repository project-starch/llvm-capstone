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

## REVOKE read lower than on R-43 v2, at every node count (attribution RETRACTED)

> **RETRACTED (2026-10-16 23:10): the attribution of this drop to the bitstream.**
> - **The image predates the fix.** H1's image `da9c54e578845735` was built on 2026-09-29, before the sublet-runtime fix
>   a07a11194fa0. It contains no SHRINK, the instruction that fix adds to `sublet_split`; fixed builds of the same
>   runtime carry four.
> - **So its carves are wrong here.** Its `sublet_split` gives the lower piece its parent's decoded start, and under
>   this bitstream's full bounds scheme that rounds below the nominal base (ISSUES R-55, the runtime half).
> - **A sibling image showed it.** `1b7a04fe237e1580`, same harness and vintage, faulted at M2's first chase on this
>   bitstream (boot m2s16-b1): mcause 6 storing to address 1, the address the carves' aliasing predicts.
> - **What is withdrawn:** "revocation is cheaper on this bitstream". The REVOKE figures below are readings of this
>   image on this bitstream, and they stand as readings; the drop's attribution waits for a fixed-runtime image.
> - **What stays, as readings of the same image:** latency, timer, node counts and linear types.


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

**Boot history and pieces, compared with 2026-09-29:**
- The session composition is the same: a fresh boot, then control, list lines 1-12 of the same list (six
  invocations, same order), then control.
- The image, host 2c9e82d101b48160, lpc 3b93a2b6e2adfa36 and buildroot d04bd83 are the same.
- One firmware piece differs. The monitor was 2dcd3a5 then and is 93440d5 now: 2dcd3a5 plus the R-55 carve fixes a
  full-format bitstream needs.
- The `rv` bracket is the REVOKE instruction between two `mcycle` reads inside the domain
  (`sublet/r1/r1_slots_pools.c`), so the monitor is not inside it. It could act only indirectly, through where nodes
  and regions sit.
- The RTL lane's diff of `capstone_rev_node.anvil` (8f6a0af98 against d93a8d66b) finds the same three queries per
  node in the REVOKE walk. So nothing in the RTL is yet shown to explain the drop, and it stays an open finding.

**Not established:** N = 3, as 2026-09-29.
