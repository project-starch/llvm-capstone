# M1's series on the Sublet bitstream: pre-registration (2026-10-16, before the boots)

The board campaign's wave 3, item 7 (`docs/plans/2026-10-16-board-campaign-all.md`, M1 node reclamation). It repeats the
five SERIES boots of 2026-10-01's M1 campaign on R-43 v2 (measurements doc "M1 on R-43 v2", boots m1v2s-1…5) on
`caplifive_sublet_d93a8d66b.bit`, with the same images and lists.

**Not repeated here:** the seven turnover-witness boots (m1tw-1…7). Their predictions are R-43 v2's node-id arithmetic:
- generation 1,352 at index head − 1;
- retirement at generation 16,383 for the 32 indices one ring arm drives.
This bitstream's node record gains a parent link, and the RTL lane has not modelled its ids. S1's boot 1 here already
latched an id of 65,922, outside the 65,532-index range that model assumes. Those boots wait for the RTL lane's
prediction on this bitstream.

## What runs, one boot per row

`drivers/board-m1v2.sh`'s rows, run through `../h1-sublet-2026-10-16`'s private driver copy:
- tree root at the main clone, `fw_payload.bin` kept;
- the classic line, `REFUSAL_RECORD=1`, `PREFLIGHT_DTS_NODES=65536`;
- host 2c9e82d101b48160, stock k800, each image's 2026-10-01 emulator log and pass record.

| boot | image | list | what it carries |
|---|---|---|---|
| m1s16-1 | `1b7a04fe237e1580` | `m1-order9-ctl.txt` | drop and ring to 10C, the bridge image |
| m1s16-5 | `249cfda958f22f16` | `m1-maxret43296.txt` | pressure to the buffer bound, then release, the bridge image |
| m1s16-2 | `eb3ed1e3f8ac4d82` | `m1v2-rt-series-rstale0.txt` | the four Design arms, then ONE stale READ, age 0, LAST |
| m1s16-3 | `eb3ed1e3f8ac4d82` | `m1v2-rt-series-wstale0.txt` | the same, then ONE stale WRITE, age 0, LAST |
| m1s16-4 | `eb3ed1e3f8ac4d82` | `m1v2-rt-series-rstale2.txt` | the same, then ONE stale READ, age 2, LAST |

## Pre-registered readings (v2's, condition by condition)

- **k800** returns 4 twice, except where the last arm wedges by design.
- **Permitted reuse:** drop and ring reach `stop=target alloc=655320` with 159 snapshots and no exhaustion, against the
  65,532-index pool.
  - The RTL lane's diff (8f6a0af98 against d93a8d66b) finds REVOKE_NODE's change only in a kill mode that a plain
    REVOKE never enters. So reissue and retirement are expected as on v2.
- **Accounting:** `minted − revoked = 31` on every snapshot.
- **Pressure:** `stop=buffer alloc=43297`. Release enters phase 2 at 43,290 and ends at 51,948.
- **Stale operations (m1s16-2/3/4):**
  - the last arm's stale access is refused at the probe's own instruction with cause 25: READ at image+0x4868, WRITE at
    image+0x48cc;
  - the same instructions through live aliases succeed earlier in the boot (`probe read ok live=1`,
    `probe write ok live=1`);
  - the refusal record reads LATCHED. Its id is recorded, not predicted.

**The cost readings, against v2's on the same images:**
- take: 72.029 (drop) / 72.010 (pressure);
- give: 107.057 (drop) / 75.066 (pressure) / 75.782 (release phase 1);
- eb3ed1e3's mint costs per invocation: drop 96.0, ring 73.0, pressure 84.0, release 73.0.
- H1 on this bitstream reads REVOKE 14-51 % cheaper per node count. If the give bracket contains the revoke, give is
  expected to fall. By how much is the reading.
- take is expected unchanged within 0.1, as on v2.

**Not established:** N = 1 per row, as 2026-10-01; the turnover witness, as above.
