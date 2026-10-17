# M1's series on the Sublet bitstream: pre-registration (2026-10-16, before the boots)

The board campaign's wave 3, item 7 (`docs/plans/2026-10-16-board-campaign-all.md`, M1 node reclamation). It repeats the
five SERIES boots of 2026-10-01's M1 campaign on R-43 v2 (measurements doc "M1 on R-43 v2", boots m1v2s-1…5) on
`caplifive_sublet_d93a8d66b.bit`, with the same images and lists.

**Not repeated here:** the seven turnover-witness boots (m1tw-1…7). Their predictions are R-43 v2's node-id arithmetic:
- generation 1,352 at index head − 1;
- retirement at generation 16,383 for the 32 indices one ring arm drives.
This bitstream's node record gains a parent link, and the RTL lane has not modelled its ids. Those boots wait for the
RTL lane's prediction on this bitstream.

**Corrected after this file's first commit (0f3a221c015a):** that version added that S1's boot 1 "latched an id of
65,922, outside the 65,532-index range". That reading was wrong.
- It took the packed record as a bare index. The record holds `{generation, index}`, with generation in id[29:16] and
  index in id[15:0] (`../m1v2-2026-10-01/extract.py:7`).
- So 0x00010182 is generation 1 at index 386, and S1's boot 2 latched 0x000300e2, generation 3 at index 226. Both
  are inside the pool.
- The deferral rests on the unmodelled ids alone.

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

## Amended before any M1 boot (2026-10-16 23:40): the fixed-runtime images

All three 2026-10-01 images predate the sublet-runtime fix a07a11194fa0: `1b7a04fe`, `249cfda9` and `eb3ed1e3` carry
no SHRINK. Under this bitstream's full bounds scheme their carves alias. `1b7a04fe` wedged M2 here (ISSUES R-55, the
runtime half; `../m2-sublet-2026-10-16`), and H1's REVOKE finding was retracted for the same reason (dev
783fb0f93f7e). So M1 runs on rebuilds from this lane, which carries the fix, made with dev147 at 0x410000 and -O1,
with v2's defines (the paper bundle's `M1/2026-10-01-v2-series/manifest.json`).

| boot | image | was | defines | carve |
|---|---|---|---|---|
| m1s16f-1 | `66764b0e62075e57` (`../h1f-sublet-2026-10-16`) | `1b7a04fe` | default | 336,112 |
| m1s16f-5 | `74263f41e1386f69` | `249cfda9` | `-DM1_MAXRET=43296` | 963,312 |
| m1s16f-2/3/4 | `b77b35e5628bb4ae` | `eb3ed1e3` | `-DM1_MAXRET=43296 -DM1_PROBE_RT=1 -DM1_STALE_MINT=0` | 964,080, as `eb3ed1e3` |

- **Each image carries 14 SHRINK words.**
- **The carves differ from the September bridge images, and the cause is known.** The tree's fixture storage grew
  by 254,656 B between 09-19 and 09-29: the default build carved 81,456 in `1b7a04fe` and 336,112 in H1's
  `da9c54e5`. The pressure image differs by exactly that amount (708,656 + 254,656). The probe image, built on 10-01
  after the growth, reproduces `eb3ed1e3`'s carve exactly, which confirms the defines.
- **Emulator flow checks, as v2's (`--cap 64 --budget 60000`):** drop, pressure and the probe image each end
  `R1 m1 end … minted=671 revoked=640`, minted − revoked = 31. The pass records are in `qemu-pass/`.
- **The stale probes moved by 0xd0 with the rest of `run_m1`:** READ (`lbu s1,0(a1)`) is at **image+0x4938** and
  WRITE (`sb a0,0(a1)`) at **image+0x499c**. The spacing is unchanged; each instruction occurs once in each image.

**Pre-registered readings as above, with the probe offsets as just stated.** Take, give and mint are reported
against v2, not predicted from H1, whose REVOKE reading is retracted. The lists are unchanged, at most seven
invocations per boot, which is within this line's region-table limit: R1's boot read seven and failed at the eighth.
