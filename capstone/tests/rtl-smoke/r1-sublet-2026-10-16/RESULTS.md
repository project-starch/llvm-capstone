# R1 warm and cold on the Sublet bitstream: results (2026-10-17)

The pre-registration and its amendments are in `README.md`; the last, before any of these boots, is lane f8b4bfde7f6a. Everything below is the fixed-runtime image
`66764b0e62075e57`, on `caplifive_sublet_d93a8d66b.bit`, on the classic line with `REFUSAL_RECORD=1`.
- Each boot logs buildroot d04bd83 and monitor 93440d5 (`fw_payload <hash> (monitor ..., buildroot ...)`). The
  opensbi wrapper's pin is not logged per boot.
- Each boot's own `fw_payload.bin` carries the image: its initramfs holds `r1_slots_pools.dom` = `66764b0e`, checked
  per boot by the audit below. The same check reads `1b7a04fe` on the unfixed boot r1ws16-b1. The judge is `judge-r1.sh`. It bundles each boot with r1-bundle and fits with the 054cea69b
campaigns' own `analyze.py`, copied unchanged, so the readings are computed exactly as the originals were.

**Every comparison below crosses image, tree, toolchain and silicon.** The originals ran images `1b7a04fe` (warm) and
`46f99c7b` (cold) on `caplifive_m1_054cea69b`. Their images are not usable here: they mis-carve on this bitstream
(README, second amendment).

## R1 warm: 15 boots (r1wf16-01..15), 90 invocations, 420 records

**Every per-boot reading is as pre-registered:**
- k800 read 4 twice in all 15 boots;
- each boot ran 6 invocations, each ending with its `R1 end` line and `released pool rc=1`;
- each boot printed one banner;
- all 420 records read `ok=1 bad=0`, completed, with no wrong survivor;
- the bundler gave 0 warnings, and neither stop condition fired.
- **No refusal and no fault.** The runner counted `refused: 0` in every boot. Every invocation printed its `R1 end`
  line, which a trapping access prevents: M2's two unfixed boots, which wedged at their first chase, show `R1 end
  lines: 0`.

| axis varied | held fixed | this bitstream | 054cea69b | reading |
|---|---|---|---|---|
| affected nodes, shared | bytes, depth, heap | **17.97 cycles/node**, R² 0.99977 | 22.91, R² 0.99966 | proportional, **-21.6 %** |
| affected nodes, combined | bytes, depth, heap | **15.37 cycles/node**, R² 0.99979 | 15.92, R² 0.99967 | proportional, -3.5 % |
| released bytes, 4 KiB to 1 MiB | nodes at nd = 32 | -1.0e-05 /byte, R² 0.0072 | 3.6e-05, R² 0.069 | no dependence |
| unrelated heap, 0 to 4 MiB | the released set | 2.3e-06 /byte, R² 0.0070 | 7.7e-06, R² 0.027 | no dependence |
| object size, 16 B to 4 KiB | one released leaf | -0.0004 /byte, R² 0.044 | -0.0009, R² 0.080 | no dependence |
| delegation depth, chains 1/2/4/8 | node count 47 at every level | -18.0 /level (R² 0.35) shared, +11.8 (R² 0.23) combined | -0.73 (R² 0.003) shared, -5.63 (R² 0.18) combined | **no trend**; chain 1 is confounded with the first point (below) |
| fill against bytes, the POSITIVE CONTROL | none | **1.8125 cycles/byte, R² 0.9999999983** | 1.8125, R² 0.9999999993 | proportional |

- **The withdrawn prediction.** About 18.5 cycles/node was pre-registered from H1's n = 256 point (lane
  4933017702bd). It was withdrawn before these boots, when H1's reading was retracted (f8b4bfde7f6a). The reading,
  17.97, lies close to it. H1's REVOKE drop has since been re-established on this same image
  (`../h1f-sublet-2026-10-16`). The prediction was not in force, so this is consistency, not a confirmation.
- **Bookkeeping is node-linear:** 28.76 cycles/node shared and 28.59 combined. 054cea69b read 28.54 and 28.79. So the
  move in the shared slope is in revocation, not in the API work around it.
- **The spatial arm** has `rv = 0` in all 210 of its records, by construction, as there.
- **Repeatability:** across the five fresh-domain repetitions of each of 42 Sublet cases, the median spread is
  **1.5 %** and the maximum 25.4 %. 054cea69b read a 7.8 % median and a 39.5 % maximum.
  - Every case sits at the same position in every repetition, so these spreads cannot see position effects. The same
    was true on 054cea69b.

**Depth: no trend, and what looks like level structure is the first point of each invocation.** The node count is
47 at every level, so depth varies with nodes held fixed, as the pre-registration requires.
- REVOKE medians, cycles, five repetitions each:

  | chain | 1 | 2 | 4 | 8 |
  |---|---|---|---|---|
  | shared | 1,004 | 813 | 827 | 813 |
  | combined | 765 | 897 | 954 | 881 |

- **The harness sweeps depth in a fixed order,** `d_chain[4] = {1, 2, 4, 8}` (`capstone/sublet/r1/r1_slots_pools.c:316`).
  So chain 1 is always the invocation's first point.
- **The first point reads off by about the same factor on axes where depth is not varied.** REVOKE at point 0,
  divided by the median of the later points, five repetitions each:

  | | bytes | heap | depth |
  |---|---|---|---|
  | shared | 1.22 | 1.05 | 1.23 |
  | combined | 0.83 | 0.79 | 0.85 |

  The first point is also inflated in bookkeeping, reissue and teardown, outside the nodes series (point 0 divided by
  the median of the rest, 35 sweeps each): 1.24, 1.20 and 1.34 on the Sublet arm, and 1.54, 1.24 and 1.70 on the
  spatial arm, which revokes nothing.
- **The point-0 excess is timing, not extra work.** The instruction count is the same at all four chain levels (82,275
  on the board), and the emulator reads the same counts at every level and position.
- **Without point 0:**
  - the shared levels read 813, 827 and 813, a 1.7 % range;
  - the combined levels read 897, 954 and 881, about 8 %.
- **So depth shows no trend in chain length,** and its chain-1 offset is the first-point effect, which the design
  cannot separate from depth.
- **054cea69b was not flat either:**
  - shared 975 / 899 / 941 / 951 and combined 788 / 745 / 736 / 725;
  - its combined fit was -5.63 /level (R² 0.18), against this run's +11.8 (R² 0.23).
  A linear R² does not test whether levels are equal, in either campaign.
- **The discriminating run, not made:** one boot whose depth invocation sweeps {8, 4, 2, 1}, and one with a
  throwaway point before chain 1. The emulator cannot discriminate, because its counts are flat.

> **Corrected before this file was committed** (the audit, 2026-10-17): the draft read the chain-1 offset as real
> level structure ("on this bitstream depth is not a free variable at the 20 % level") and called 054cea69b's levels
> "equal within noise". The fixed sweep order confounds chain level with point position, and 054cea69b's combined
> fit was not flat.

## R1 cold: 5 boots (r1cf16r1..r5), one 12-point `wide` series each, 60 records

**Every per-boot reading is as pre-registered:**
- k800 read 4 twice in each boot, with one invocation and `released pool rc=1`;
- each boot printed one banner;
- all 60 records read `ok=1 bad=0`, completed, with no wrong survivor;
- 0 warnings.

Rep 1 is this run's, on the fixed image. The Sublet acceptance's s10a3 ran an unfixed image and is not used.

`rv`/`nd`, the median of five repetitions, in cycles per destroyed node:

| `nd` | this bitstream | spread | 054cea69b | difference | step | share of the rise |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | **14.844** | 2.5 % | 19.66 | -24.5 % | — | — |
| 256 | 16.184 | 5.5 % | 20.86 | -22.4 % | +1.340 | 10.5 % |
| 512 | 16.998 | 2.2 % | 22.05 | -22.9 % | +0.814 | 6.4 % |
| 1,024 | 19.005 | 1.9 % | 23.62 | -19.5 % | +2.007 | 15.7 % |
| 1,536 | 21.063 | 1.1 % | 23.81 | -11.5 % | **+2.058** | **16.1 %** |
| 1,792 | 21.935 | 1.5 % | 24.09 | -8.9 % | +0.872 | 6.8 % |
| 2,048 | 22.739 | 1.6 % | 25.09 | -9.4 % | +0.804 | 6.3 % |
| 2,304 | 23.536 | 0.9 % | 25.81 | -8.8 % | +0.796 | 6.2 % |
| 3,072 | 24.855 | 0.9 % | 26.88 | -7.5 % | +1.319 | 10.3 % |
| 4,096 | 25.785 | 1.0 % | 27.74 | -7.0 % | +0.930 | 7.3 % |
| 8,192 | 27.057 | 0.1 % | 28.14 | -3.8 % | +1.272 | 10.0 % |
| 16,384 | **27.624** | 0.1 % | 28.15 | -1.9 % | +0.567 | 4.4 % |

**Four readings:**
1. **The warm end is about a quarter cheaper; the cold end is nearly unchanged.**
   - nd = 64 is the invocation's first point, which carries the first-point timing seen in the warm series
     (r1cf16r1: bookkeeping 34.6 per node at nd = 64, against about 29.1 at every other rung). So its -24.5 %, and
     the 1.86x below, include a position component.
   - nd = 256 and 512 still read -22 %, so "about a quarter cheaper" does not rest on point 0. At nd = 64 this bitstream reads
   24.5 % below 054cea69b; at nd = 16,384, 1.9 % below. The difference shrinks with nd, though not strictly
   monotonically: about -23 % to nd = 512, -19.5 % at 1,024, between -7 % and -12 % from 1,536 to 4,096, then -3.8 %
   and -1.9 %.
2. **Cold over warm is 1.86x** (27.62 / 14.84), against 1.43x on 054cea69b. The rise is 12.78 cycles/node, against
   8.49.
3. **The constant has not saturated by nd = 16,384,** by analyze.py's own test: the top two rungs within 1 %.
   8,192 to 16,384 moves it by 2.1 %; 054cea69b moved 0.034 % over the same doubling. The last step is the smallest
   share of the rise, 4.4 %, so the approach is flattening.
4. **The approach is still a RAMP.** The largest single interval holds 16.1 % of the rise, at nd = 1,536, below the
   2,048-node capacity. nd = 1,024's +2.007 is within 0.05 of it, so the location is not robust; "below capacity"
   holds either way. 054cea69b's largest held 18.5 %, at nd = 1,024.

**The positive control held:** bookkeeping and fill rise 214.7x and 263.0x across the same records.

**Against this bitstream's own warm slope:**
- the cold constant at nd = 16,384 is **+53.7 %** over the warm shared slope (27.62 against 17.97);
- 054cea69b's was +22.9 % (28.15 against 22.91).

## What these runs do NOT establish

- **Why the warm end moved and the cold end did not.** The bitstream's revocation path differs from 054cea69b's in
  the node record's parent link and in R-45's CREVOKE flush (README). The image differs as well. These runs do not
  separate the two.
- **Bitstream or monitor.** These boots ran monitor 93440d5; 054cea69b's ran 2dcd3a5. H1's REVOKE drop on the same
  image leaves the same question open (`../h1f-sublet-2026-10-16`).
- **Where the cold constant saturates on this bitstream.** It needs a rung above 16,384, and the series stops there.
- **Depth apart from the first-point effect** (warm, above). It needs the reversed-order boot described there.
- **Timing closure.** As on 054cea69b, no cycle count here separates a design property from an artefact of the build.
- **The invalid-access companion of R1 step 1.** It was not part of this campaign.

## The audit (claim-auditor, 2026-10-17, before commit)

**Confirmed:**
- **Record selection, per boot from each boot's own `fw_payload.bin`.** All 20 boots carry `66764b0e`. The positive
  control: the same extraction reads `1b7a04fe` on r1ws16-b1.
- **The bitstream:** every `driver.log` reports the same flash epoch for caplifive_sublet_d93a8d66b, so there was no
  reflash in between.
- **The bundle:** 420 + 60 records equal each boot's own R1 lines, field by field.
- **Every table value** recomputes from `runs.jsonl`.
- **The 054cea69b column** matches the measurements doc and that bundle's `results.json`.
- **The analyzers** are byte-identical to the 054cea69b bundles'.

**Refuted:** the draft's depth paragraph, corrected above.
