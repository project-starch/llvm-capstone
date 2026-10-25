# S1 on the Sublet bitstream, the SQLite half: pre-registration (2026-10-16, before the boots)

The board campaign's wave 1 item 4 (`docs/plans/2026-10-16-board-campaign-all.md`, S1-SQLite). It repeats
`../s1-sqlite-silicon-2026-10-01` (boots s1sql-b1a…b3a on R-43 v2, measurements doc "S1 on R-43 v2") on
`caplifive_sublet_d93a8d66b.bit`.
- A1's six probes run in the measurement harness, one image per arm, the probe chosen at run time by
  `--s1-probe n`.
- The arms are SQLite's memsys5 and lookaside unprotected, and the same allocators under the Sublet port.
- The audited 2026-10-01 pre-registration lives in the nested-allocators paper bundle; this folder is the board
  repeat, and the bundle is not edited.

## The images, rebuilt

`../s1-sqlite-silicon-2026-10-01/build-images.sh`, unchanged, on lane b0-silicon-runtime (dev's sqlite port and
Sublet runtime, which carries the sublet-runtime fix a07a11194fa0):
- compiler `clang version 22.0.0git (llvm-capstone 612b3ec514c0)`;
- the P1 cells' geometry: -O2, lookaside 1200,40, heap 2 MiB, stack 385,024. The Sublet arm is lent a 2 MiB arena
  and 1,750,285 B of tables.

| arm | image | entry | 2026-10-01's |
|---|---|---|---|
| memsys5, unprotected | `41f9b021371b4b66` | 0x810000 | `611df066a69668b7` |
| Sublet | `445d93b331cd3ba2` | 0xc10000 | `483f764c8d691b3c` |

**Why rebuilt, and what differs:**
- The Sublet image must carry the sublet-runtime fix. On this full-format bitstream the unfixed runtime wedges at
  `revoke` (the P1 cell 6 matched pair, 2026-10-16).
- Both images differ from 2026-10-01's. The tree moved since then (dev's sqlite port and runtime), and so did the
  toolchain: 10-01's build named only "a toolchain with #119"; today's is 612b3ec514c0.
- So this repeat compares verdicts across bitstreams, not instruction streams.

Each image earned its emulator pass record by running the benchmark (`run-speedtest1-measure.sh`, verification hash
`112006 38bb59fd…` on both). The records are `~/capstone-artifacts/qemu-pass/<sha256>`.

## The emulator reading (main clone's capstone-qemu deb7d75756, the board's host 2c9e82d101b48160, `qemu/`)

- **memsys5:**
  - p1 and p2 read the freed slot and the freed block (NOTRAP);
  - p3 and p4 read their siblings;
  - p5 reads `p[127]`;
  - p6 reads `p[128]` past the block (BOUNDS_NOTRAP).
  - All return.
- **Sublet:**
  - p3, p4 and p5 return as on memsys5;
  - p1 faults 24 at `speedtest1_probe+0x7ac` (+0x1e46c), `lbu a0, 0x0(s5)`;
  - p2 faults 24 at `speedtest1_probe+0x258` (+0x1df18), `lbu a0, 0x0(s3)`;
  - p6 faults 5 at `speedtest1_probe+0xcf0` (+0x1e9b0), `lbu a3, 0x0(a3)`.
  - Each fault is the probe's own load.

## The boots

`drivers/board-s1sql.sh` with `S1_FOLDER` pointing here (a new override; the default stays 2026-10-01's), and
`S1_DIR` holding the two images.
- The classic line: d04bd83 / df0e9a8 / 93440d5, the #3 module, `REFUSAL_RECORD=1`.
- The main clone's runner tree, as `../repro322-silicon-2026-10-16`, and `PREFLIGHT_DTS_NODES=65536`.
- Each boot keeps its own `fw_payload.bin` beside its record (new in the driver), so image membership stays
  checkable by sha256 after the restore bake.

| boot | cells, in order |
|---|---|
| 1 | k800, m5p1-m5p6, sbp3, sbp4, sbp5, **sbp1** |
| 2 | the same, then **sbp2** |
| 3 | the same, then **sbp6** |

## Pre-registered readings

- **k800** returns 4. Otherwise the boot is void.
- **Every returning cell** prints the emulator's marker lines and returns, in every boot.
- **sbp1 and sbp2** wedge with **cause 25** at their probe's `lbu`. The emulator reads 24, because it untags a
  revoked capability (Q-11); 2026-10-01's silicon read 25 for both.
- **sbp6** wedges with **cause 28** (out of bounds) at its probe's `lbu`. The emulator reads 5.
- The refusal record at each wedge:
  - LATCHED, hit-dead, for sbp1 and sbp2: a genuine revocation verdict, attributed to the boot's first cause-25
    denial and not to the probe, as 10-01 stated;
  - EMPTY for sbp6.

**New on this bitstream:**
- the Sublet port's lend API on the full format, with the sublet-runtime fix;
- the node record's parent link.

A Sublet cell that returns where 10-01 refused is the informative divergence. So is a refusal with a different
cause.

**Not established** (as 10-01): the fault cells are N = 1, and before-reuse only.
