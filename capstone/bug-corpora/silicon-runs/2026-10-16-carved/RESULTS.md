# FFmpeg's carved corpus on silicon: results (2026-10-16, caplifive_sublet_d93a8d66b)

Pre-registered in README.md (lane e4f15d4b7a32, before the boot). One boot, 06:46-07:01:
- firmware 124238b7d5b41abd, Image 0bba51e9baec9816, on set 2's exact-regions supervised-call line;
- three arms, 98 runs, every one ended by itself (each arm's rung returned 0).

**Every row and every control is as pre-registered,** by outcome, cause and site, on the pre-registered image
(sha256):
- the 13 cases on each arm;
- carve control 99 on each arm, and 98 on the Sublet carve;
- the four tool controls on each arm;
- every fixed run printed VERDICT FIXED and exited 0.

| arm | 13 carved cases | carve controls | tool controls (clean / oob / uaf / subobj) |
|---|---|---|---|
| silicon-level0 (pointer carve) | 13 MISSED | 99 completes | RETURNED / CAUGHT 28 / RETURNED / RETURNED |
| silicon-carve-bounds (SHRINK at the carve) | **13 CAUGHT 28** at the labelled probe | 99 CAUGHT 28 | RETURNED / CAUGHT 28 / RETURNED / RETURNED |
| silicon-carve-sublet (a CDERIVE child per region, CREVOKE at a re-carve) | **13 CAUGHT 28** at the labelled probe | 99 CAUGHT 28; **98 CAUGHT 25** | RETURNED / CAUGHT 28 / CAUGHT 25 / RETURNED |

**What it shows:**
- **A crossing between regions carved from one allocation is invisible to per-object bounds:** 13 MISSED on level0,
  with control 99 completing in the same boot.
- **Bounding each region at the carving code catches all 13 on silicon,** whether by SHRINK or by the Sublet carve's
  CDERIVE children.
- **The Sublet carve's lifetimes hold on silicon:** a re-carved region's old alias faults 25 while its sibling and
  successor work (98). Here CDERIVE nests three levels below level0's group.
- **The inexact group derive behaves on silicon as modelled.** Case 00's fixed run prints `base_eq=0` on the board, as
  on the rounding QEMU: its group over a 35,840-byte block rounds its base down, so its regions sit 48 bytes below the
  case's arithmetic. The run still reads FIXED, and the buggy crossing is still CAUGHT at the probe.

## Per case

| case | silicon-level0 | silicon-carve-bounds | silicon-carve-sublet |
|---|---|---|---|
| 00_85407c7e63_mpegvideo_edge_emu_u_block_one_row_short | MISSED | CAUGHT 28 | CAUGHT 28 |
| 01_699341d647_apedec_array_0000_writes_64_into_channel_1 | MISSED | CAUGHT 28 | CAUGHT 28 |
| 02_cd7524fdd1_apedec_filter_3800_reads_past_short_channel | MISSED | CAUGHT 28 | CAUGHT 28 |
| 03_55937bb4a7_alsdec_ra_block_runs_into_next_channel_prefix | MISSED | CAUGHT 28 | CAUGHT 28 |
| 04_cd09284924_alsdec_mcc_chan_data_one_slot_per_channel | MISSED | CAUGHT 28 | CAUGHT 28 |
| 05_9d3032b960_alsdec_opt_order_past_max_order | MISSED | CAUGHT 28 | CAUGHT 28 |
| 06_2d0bea4719_vp9_uv_nnz_ctx_sized_for_420 | MISSED | CAUGHT 28 | CAUGHT 28 |
| 07_2563a33856_vp9_intra_pred_carved_for_old_bpp | MISSED | CAUGHT 28 | CAUGHT 28 |
| 08_b5ff61695f_swscale_v_line_at_pre_doubling_stride | MISSED | CAUGHT 28 | CAUGHT 28 |
| 09_043bcdcdb0_svq1enc_inter_block_runs_into_source | MISSED | CAUGHT 28 | CAUGHT 28 |
| 10_d2213b6493_rv34_b_block_carved_at_old_linesize | MISSED | CAUGHT 28 | CAUGHT 28 |
| 11_68226ed9ec_vorbis_type1_residue_end_spans_channels | MISSED | CAUGHT 28 | CAUGHT 28 |
| 12_ac59fc542f_thumbnail_hbd_tail_index_past_plane_slice | MISSED | CAUGHT 28 | CAUGHT 28 |
| control 99 (shared/carve-control.c) | completes | CAUGHT 28 | CAUGHT 28 |
| control 98 (shared/carve-recarve-control.c) | - | - | CAUGHT 25 |

Every CAUGHT is at the case's labelled probe (`ffc_read_probe_*` / `ffc_write_probe_*`).

**Judged by** `tools/run-silicon-cases.py board` on the run-scoped transcript. Its carve controls gate their arm. The
pre-registered cause and site were then compared row by row, keyed by image hash.

**Not in this set:** the plane corpus (1 case), whose image needs FFmpeg's libavutil on the silicon ABI. It has since
been built from 14 libavutil units and is in the next set.
