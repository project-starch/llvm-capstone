# S1 on the Sublet bitstream, the SQLite half: results (2026-10-16)

Pre-registered in README.md (lane 930bd30d1060, before the boots). Three boots on `caplifive_sublet_d93a8d66b.bit`,
classic line, images `41f9b021371b4b66` (memsys5, 0x810000) and `445d93b331cd3ba2` (Sublet, 0xc10000):
- s1sql-b1s16, 20:17-20:44;
- s1sql-b2s16, 20:44-21:12;
- s1sql-b3s16, 21:12-21:40.

The result lines are in `results/`. Each one is the judge's reading of one arm, taken from that arm's own segment of
the run-scoped transcript, against `cells.tsv`. The judge (`/tmp` script, not committed) was negative-tested: a
`cells.tsv` copy that expects cause 28 for sbp2 reads MISMATCH on boot 2. Each boot kept its `fw_payload.bin`.

**Every pre-registered reading holds in all three boots.** The Sublet port refuses both use-after-free reads at the access, with cause 25, and the spatial read past a block with cause 28. The unprotected memsys5 build reads freed memory in all three boots.

| case | memsys5 (3 boots) | Sublet |
|---|---|---|
| p1, lookaside slot read after `sqlite3_finalize` | reads the freed slot | **cause 25 at the probe's `lbu`** (+0x1e46c), record LATCHED hit-dead, boot 1 |
| p2, 4,096 B memsys5 block read after `sqlite3_free` | reads the freed block | **cause 25 at the probe's `lbu`** (+0x1df18), record LATCHED hit-dead, boot 2 |
| p3 / p4, a live sibling block / slot | read | read, 3 boots |
| p5, last byte of a 128 B block | read | read, 3 boots |
| p6, the byte past it | read past the block (BOUNDS_NOTRAP) | **cause 28 (out of bounds) at the probe's `lbu`** (+0x1e9b0), tval 0xace00080, record EMPTY, boot 3 |

- **k800** returned 4 in every boot.
- **Every returning cell** printed the emulator's marker line and returned.
- **The freed bytes** that m5p1 and m5p2 read are non-ASCII. The capture stores each as U+FFFD, so their values are
  not recoverable from it.

**The refusal ids** are recorded, not predicted. They decode as `{generation id[29:16], index id[15:0]}`
(`../m1v2-2026-10-01/extract.py:7`):
- boot 1: 0x00010182, generation 1 at index 386;
- boot 2: 0x000300e2, generation 3 at index 226.

Both lie inside the 65,532-index pool. As on 10-01 they are not attributed to the probe: ten invocations ran first.

**Against 2026-10-01 (R-43 v2):** the same verdict in every case, on rebuilt images (README "Why rebuilt"). This
bitstream adds the Sublet port's lend API on the full format with the sublet-runtime fix, and the node record's
parent link. Neither changed a verdict.

**Not established:** the fault cells are N = 1; before-reuse only; survivors ran before the fault, not after it.
