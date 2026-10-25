# The sub-object and pool corpora on silicon: results (2026-10-16, caplifive_sublet_d93a8d66b)

Pre-registered in README.md (lane 1ef381aaa812, before the boot). One boot, 05:51-06:46:
- firmware 37ae782f7718b018, Image f44e75a27d067401;
- the exact-regions supervised-call monitor (wrapper 1269c77, monitor ebe0a25) and the 127-slot process-ABI module;
- five arms, 94 runs, every one ended by itself (each arm's rung returned 0).

**Every row and every control is as pre-registered,** by outcome, cause and site, on the pre-registered image (sha256):
- 37 rows: 9 + 9 + 9 sub-object, and 5 + 5 pool;
- 20 controls;
- every fixed run printed VERDICT FIXED and exited 0.

| arm | rows | controls (clean / oob / uaf / subobj) |
|---|---|---|
| silicon-subobject | 7 CAUGHT 28 at the probe or a declared site; 04 and 06 MISSED | RETURNED / CAUGHT 28 / RETURNED / **CAUGHT 28** |
| silicon-level0 | 9 MISSED | RETURNED / CAUGHT 28 / RETURNED / RETURNED |
| silicon-lifetimes | 9 MISSED | RETURNED / CAUGHT 28 / CAUGHT 25 / RETURNED |
| silicon-pool-stock | 4 MISSED; control 90 completes | RETURNED / CAUGHT 28 / CAUGHT 25 / RETURNED |
| silicon-pool-sublet | **4 CAUGHT 25** in ff2_case_run (a declared fault site); control 90 CAUGHT 25 at read_probe | RETURNED / CAUGHT 28 / CAUGHT 25 / RETURNED |

**What it shows:**
- **Field bounds work on silicon** (`-Xclang -fcapstone-subobject-bounds`). They catch the seven crossings QEMU's
  capstone-subobject arm caught, and miss the same two (04, 06), whose images are byte-identical to the level0 arm's.
  - A crossing that stays inside one allocation is invisible to per-object bounds and per-object lifetimes alike: 9
    MISSED on both.
  - The 88 MiB domains these images need launched without a CMA failure. One launch's diagnostics read 1.66 s at
    12.5 MHz.
- **FFmpeg's buffer pools on silicon:**
  - **Stock:** every case reuses a returned entry through a stale reference, and nothing faults. Control 90 completes,
    reading the reissued buffer.
  - **Patch 0003** (each hand-out a CDERIVE child of the entry's lifetime, CREVOKEd at release) catches all four, cause
    25.
  - This is the first application allocator run on silicon with CDERIVE nested inside the lifetimes heap: the pool's
    child, inside the entry's lifetime, inside level0's object.

**The revised level0** (free by address, through the child recorded in the block's header) was re-run as promised.
On the silicon-mimic QEMU, the plain corpora's lifetimes arm reads 72 of 72 as predicted, with controls ok: the
revision changes no plain verdict.

## Per case

| case | silicon-subobject | silicon-level0 | silicon-lifetimes |
|---|---|---|---|
| 00_8864fd0aec_cbs_h265_pic_timing_member | CAUGHT 28 in ff2_case_run | MISSED | MISSED |
| 01_68845e26f7_vulkan_hevc_refpicset_member | CAUGHT 28 in ff2_case_run | MISSED | MISSED |
| 02_e058af88ab_vulkan_hevc_dpb_member | CAUGHT 28 in ff2_case_run | MISSED | MISSED |
| 03_89de2f0de1_aac_arith_last_member | CAUGHT 28 in ff2_case_run | MISSED | MISSED |
| 04_1a00ea51cb_rtsp_control_url_underflow | MISSED | MISSED | MISSED |
| 05_d29ff88422_vulkan_av1_tile_sizes | CAUGHT 28 | MISSED | MISSED |
| 06_a809a784ec_vvc_entry_point_start_ctu | MISSED | MISSED | MISSED |
| 07_275e217b10_hlsenc_key_uri_strlcpy_size | CAUGHT 28 in memcpy | MISSED | MISSED |
| 08_fb862976df_cbs_h266_col_width_val | CAUGHT 28 | MISSED | MISSED |

| case | silicon-pool-stock | silicon-pool-sublet |
|---|---|---|
| 00_461fb22053_af_join_dedup_bound | MISSED | CAUGHT 25 in ff2_case_run |
| 01_1886c3269d_h264_refs_partial_clear | MISSED | CAUGHT 25 in ff2_case_run |
| 02_316531e61c_vidstab_parked_plane_pointer | MISSED | CAUGHT 25 in ff2_case_run |
| 03_5c66a3ab51_vvc_nonref_output_releases_tabs | MISSED | CAUGHT 25 in ff2_case_run |
| 90_ctl_buffer_returned_to_pool (control) | completes | CAUGHT 25 |

A CAUGHT with no site is at the case's labelled probe. A named site is one the case declared before any run
(case.json `fault_sites`), as on QEMU.

**Judged by** `tools/run-silicon-cases.py board` on the run-scoped transcript, with each arm's prefixed keys. The
pre-registered cause and site were then compared row by row, with the image hash as the key. The comparer flags all
46 spatial rows of set 1's lifetimes record, whose cause field carried QEMU's 5 / 7, so it can fire.
