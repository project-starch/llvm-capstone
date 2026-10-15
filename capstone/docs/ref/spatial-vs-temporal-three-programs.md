# Spatial vs temporal, nested vs plain — memcached, FFmpeg, tshark

**Scope: the three programs the paper's target evaluation uses.** Other corpora in this tree
(cpython, httpd, PostgreSQL, sqlite, mruby, the cross-language set) are deliberately out of scope
here; the generated `capstone/bug-corpora/INDEX.md` is the whole-tree inventory, and
`three-columns-all-programs.md` puts the other six programs on section 0's three columns.

**Read this first if you came here grepping for "spatial".** `spatial` is also the name of a
**measurement arm** — bounds-only Capstone, revocation disabled — and it appears in dozens of
`case.json` files and in the paper as `Capstone spatial 0/57`. That zero is a *configuration*
failing to catch *temporal* bugs. It is not a count of spatial bugs, and it is not evidence that
anything is broken. A grep for the word finds arms, not defects.

## A. Audit of these tables (2026-10-10)

Every cell the tables below are computed from -- 1,103 (case, arm) cells over the 130 cases -- was traced to
the run record that produced it, by four independent read-only audits (FFmpeg nested, FFmpeg plain, tshark,
memcached), each told to attack named gaps and to quote a record for every claim. The plan and the
pre-registrations of every run the audit needed are `docs/history/10-10-2026_17-20-00_bug-corpus-audit-three-programs.md`
(`bbc7db0bd3d5`) and carved case 12's own `case.json` (`26cdee4e4d8d`). The Capstone carved arms started
before the plan was pushed, so their predictions were the readings already on `dev`. **Every pre-registered prediction held.**

**No cell contradicted its record.** What the audit found instead:

| finding | cells | fix |
|---|---:|---|
| The generator read an arm with no verdict by its oracle's opening words, so a PREDICTION ("fault at the labelled read probe") counted as a catch | 334 holes once strict | strict generator; 149 cells now cite the record the audit traced them to, the rest closed by the runs below |
| ASan cells with no per-case record: plain-heap corpora run when they held 4, 1 and 1 cases | 40 | re-run, 46 cases two-sided, every report at the labelled probe (`results/2026-10-10-native-plain-heap`) |
| ASan and CheriBSD arms that could read only one value: memcached's and wmem's hosted harnesses carved everything out of one arena (wmem also with the chunk port compiled in), so nothing reached a redzone or `free()` | 22 + 9 + 22 | upstream's own backing (`MCP_STOCK_MALLOC`, `WM_LIBC_SYSTEM`) with wmem's own control 90 in the same run; memcached 8 now reported by ASan; wmem on stock CheriBSD reads "missed (reused)" with each case's own reuse assertion holding |
| A reduction defect: memcached allocator 08's fixed arm read one byte past its object | 1 | the fixed arm probes where its bounded scan stopped |
| One defect counted twice: FFmpeg plain-heap 13 is the release/9.0 backport of case 02's fix (`git patch-id` equal) | 1 case | `duplicate_of`; every table leaves it out -- **129 distinct defects** |
| A carve filed as a struct member: subobject 09 (vf_thumbnail) crosses a slice carved out of one `av_calloc`, which carved-repros files NESTED by the rule these tables use | 1 case | moved to carved-repros as case 12 and measured on every arm |
| "Capstone bounds" on nested rows is the port's per-chunk bound, not malloc's (wmem `wm_narrow`, memcached mode 0) | 13 spatial | the column is relabelled in 0b; the stock allocators on a per-allocation bound are column 2 |
| Unbacked Capstone cells: memcached allocator 06 and 07 on `spatial` and `sublet` (prose about a run, no record) | 4 | measured, as predicted (complete), negative control firing on all four |
| Mechanism mis-attributed: memcached plain-heap 06-08 recorded in the write probe, wmem 16/17/18/20/21 "not the probe" | 22 arms | llvm-nm on each run's own binary: `mch_read_probe+0x10` and the `wm_defect_write` label; runners now resolve each case's own probe |
| CheriBSD/PoisonCap records carried no binary or platform hash, and the bounds control only as a marker printed before its faulting read | 47 records | `provenance.tsv` from each run's surviving summary: every bounds control exited 162 |

Not changed by the audit, and still true of the tables: the two races are serialised into one thread
(0.1); the field-bounds arm on the plain-heap corpora produced images byte-identical to the spatial arm's,
in all three programs (every one of their 46 field-bounds images hashes equal to a spatial image), so its "no false
positive" there is vacuous rather than a test; the FFmpeg subobject harness gives a struct
no allocation bound of its own on the `spatial`, `sublet`, CHERI and PoisonCap arms (the one bounded reading is
the `sublet-full` arm, column 3, which puts `av_malloc` on the Sublet heap: 0/9); and several CheriBSD predictions were committed with their results rather than
before them (plain-temporal's NOT-REISSUED, carved ASan, the pool `sublet-malloc` relabel). The tshark
plain-heap cases 10 and 11 are caller-dependent -- their in-file callers take an emem `ep_alloc` chunk, other
callers a direct allocation -- and stay plain with the callers recorded.

**What moved in the published numbers:** 130 cases became 129 defects; spatial nested/plain went from
26/56 to 27/54 (case 12 in, case 13 out); FFmpeg's carved row went from 12 to 13 cases and its struct row
from 10 to 9; ASan on memcached's nested spatial row went from 0/4 to 1/4. Every reading of every other
cell, and every headline -- 22/22 nested temporal caught only through each allocator's port or adapter,
0/22 by CHERI's quarantine and by Sublet in malloc -- is unchanged, and now cites a record.

## A2. Second audit and one new case (2026-10-10/11)

The whole-corpus audit (`docs/history/10-10-2026_23-30-00_whole-corpus-audit-cross-program.md`, and the
cross-program comparison `docs/ref/bug-corpora-cross-program-comparison.md`) changed these tables in
three ways, each run pre-registered:
- **Attribution, no verdict change:**
  - memcached allocator 08's CHERI catch is at the defective scan (`mc_case_body+0x1e8`);
  - the subobject arm no longer accepts a fault anywhere, and all seven catches stand at their sites;
  - the supervised CheriBSD runs name their sites;
  - wmem's PoisonCap runner resolves each case's label.
- **A new case, wmem-repros 22** (`c702b44a01`, a fix reversal): the corpus's first double free (S2)
  into a nested allocator, which corrupts wmem's in-band free list (S4).
  - Missed by CHERI, Sublet in malloc, ASan, the region-granular port and PoisonCap mode 0.
  - Caught by the chunk port at the allocator's handback, and by PoisonCap mode 1.
  - The counts below include it: 130 defects, 49 temporal, 23 nested temporal.
- **Virtual-Capstone columns** for the three programs, beside the physical ones: section 0c.


## 0. Three columns per bug: CHERI, Sublet in malloc, Sublet in nested — all 130 defects (after the audits of 2026-10-10 and the case added on 2026-10-11)

Generated by `capstone/bug-corpora/tools/catch-tables.py --board`, which reads only `case.json`
and exits 1, naming each hole, while any of a bug's three cells is not a reading. Since the audit
(section A) a cell is a reading only if its arm says it was MEASURED and cites the record; a bare
oracle is a prediction and counts as a hole. 130 defects: 131 cases, of which FFmpeg plain-heap 13
is a backport of case 02's fix and is left out (`duplicate_of`). A cell reads **caught / measured**.

1. **CHERI** — stock CheriBSD purecap, libc revocation on. A freed chunk **held in quarantine**
   (the stale pointer was followed, the chunk was never reissued; a revocation control faulted in
   the same boot) counts as caught, by the rule asked for, and is shown apart: "(n held)". It is
   the unported platform: CHERI's carve and field bounds are in section 0b.1.
2. **Sublet in malloc** — Sublet ONLY as the system allocator, the program's nested allocator
   stock. On a plain case that is the `sublet` arm. On a nested one it is the new `sublet-malloc`
   arm: wmem as released (`WM_VARIANT=reference`: none of the port's patches, every `g_malloc` a
   region, every `g_free` a revoke; positive control `controls/sublet-malloc/90`), memcached's
   ledger in mode 2 (a chunk carries its whole page's bound, an object its own, nothing revoked
   until a page or object is given back; the page-bound check negative-tested; mode 1 on the same
   images faults on all five temporal cases), and FFmpeg's pools unported (`poolstock`, the
   application port's 2026-10-04 readings; its build refuses a pool arm without the Sublet heap,
   `ports/ffmpeg/app/host/build-domain.sh`).
3. **Sublet in nested** — the Sublet port of the **innermost** allocator that made the object:
   FFmpeg's pool port (`sublet-port`), wmem's chunk port (`sublet-chunks`), the slab/cache port
   (`sublet`), and where code carves the object out of a block, that carve's port (`sublet-carve`:
   the carved corpus, av_frame_get_buffer's planes, and memcached 6 and 7, whose key and suffix
   ITEM_key/ITEM_suffix carve out of a slab item). A **plain** case, with no nested allocator on its
   path, runs with its program's port linked and live (`sublet-full`, `tools/full-config/`): a
   non-interference check, every run required to print the port's `FULLCONFIG ... live` line.

**Temporal (49)**

| program | allocator layer | axis | n | CHERI (quarantine = caught) | Sublet in malloc | Sublet in nested |
|---|---|---|---:|---:|---:|---:|
| FFmpeg | AVBufferPool | nested | 4 | 0 / 4 | 0 / 4 | 4 / 4 |
| FFmpeg | direct malloc | plain | 13 | 13 / 13 (13 held) | 13 / 13 | 13 / 13 |
| tshark | wmem | nested | 14 | 0 / 14 | 0 / 14 | 14 / 14 |
| tshark | direct g_malloc | plain | 10 | 10 / 10 (10 held) | 10 / 10 | 10 / 10 |
| memcached | slabs.c / cache.c | nested | 5 | 0 / 5 | 0 / 5 | 5 / 5 |
| memcached | direct malloc | plain | 3 | 3 / 3 (3 held) | 3 / 3 | 3 / 3 |
| **Total** |  | 23 n · 26 p | 49 | 26 / 49 (26 held) | 26 / 49 | 49 / 49 |

**Spatial (81)**

| program | allocator layer | axis | n | CHERI (quarantine = caught) | Sublet in malloc | Sublet in nested |
|---|---|---|---:|---:|---:|---:|
| FFmpeg | carved buffer | nested | 13 | 0 / 13 | 0 / 13 | 13 / 13 |
| FFmpeg | frame-pool plane | nested | 1 | 0 / 1 | 0 / 1 | 0 / 1 |
| FFmpeg | direct malloc | plain | 24 | 21 / 24 | 24 / 24 | 24 / 24 |
| FFmpeg | inside one struct | plain | 9 | 0 / 9 | 0 / 9 | 0 / 9 |
| tshark | wmem | nested | 9 | 0 / 9 | 0 / 9 | 9 / 9 |
| tshark | direct g_malloc | plain | 12 | 11 / 12 | 12 / 12 | 12 / 12 |
| memcached | slabs.c / cache.c | nested | 4 | 1 / 4 | 1 / 4 | 4 / 4 |
| memcached | direct malloc | plain | 9 | 8 / 9 | 9 / 9 | 9 / 9 |
| **Total** |  | 27 n · 54 p | 81 | 41 / 81 | 46 / 81 | 71 / 81 |

How the three columns read:

- **Temporal, CHERI 26 of 49 — all 26 held, none faulted.** Every plain temporal case reaches
  `free()` and the quarantine withholds the chunk; in the 23 nested sequences the object is reused
  inside its allocator and never reaches `free()`, so the quarantine never sees it.
- **Temporal, Sublet in malloc 26 of 49 — the same split, with a revoke instead of a hold.** In each
  of the 23 nested sequences the lifetime ends on an allocator free list: wmem's packet pool keeps
  its first block on reset (12), wmem's BLOCK allocator takes an individual free (case 12) or the
  same chunk freed twice (case 22, the double free added on 2026-10-11), memcached's chunks go back to their slab class and its objects to
  cache.c's list (5), FFmpeg's pool keeps its buffers (4). That is a property of these sequences,
  not of the programs: wmem does `g_free` every block after the first and every jumbo on reset
  (control 90 is that case), and cache.c frees above a configured limit (default 0, none).
- **Temporal, Sublet in nested 49 of 49.** The port revokes on the allocator's own release, so the
  23 fault -- case 22 at the allocator's own handback probe, where its second free hands back a revoked chunk;
  the 26 plain ones read as in column 2.
- **Spatial, Sublet in malloc 46 of 81** is the Sublet heap's per-object bound: all 45 plain
  cases that leave their allocation, and memcached 8 (its rbuf object is its own allocation); nothing inside a block.
- **Spatial, Sublet in nested 71 of 81.** The chunk and slab ports bound each chunk (wmem 9/9, slab
  5 and 8); the carve ports bound each carved region (carved 13/13, memcached 6 and 7). Those carve
  catches are the port's BOUND -- the same shrink as carve bounds, which CHERI's carve bounds match
  13/13 -- not a revocation: no buggy sequence re-carves, and the re-carve control alone shows the
  revocation. Column 3's 13/13 against column 1's 0/13 compares a ported carve with an unported
  platform, not Sublet with CHERI.
- **Plain rows, columns 2 and 3 agree on all 80** (71 caught). That is the expected
  non-interference: the port is off these bugs' path. For tshark and memcached the image carries
  the port's Sublet layer, driven by the constructor, not wmem or slabs.c themselves; the SDK's heap
  size differs from the `sublet` arm's too (each corpus's README says which).

### 0.1 "Sublet misses races": what it misses, and the fix

**Two of the 130 are races**, both memcached and both nested: allocator 01 (an IO object walked
across its free by two threads) and 04 (an unlocked refcount decrement loses a concurrent get). The
corpus says so itself (`memcached/allocator-repros/shared/corpus.h`: "two of its defects are
races") and serialises each interleaving into one thread. Two more nested lifetimes sit beside them
and are NOT races: FFmpeg pool 03 (its PROVENANCE excludes the frame-threading race) and memcached
02 (the allocator deliberately overwrites a held item's refcount).

| case | kind | CHERI (quarantine = caught) | Sublet in malloc | Sublet in nested |
|---|---|---|---|---|
| memcached 01 | race | missed (reused) | missed | caught, cause 24 |
| memcached 04 | race | missed (reused) | missed | caught, cause 24 |
| memcached 02 | premature release into a slab | missed (reused) | missed | caught, cause 24 |
| FFmpeg pool 03 | premature release into a pool | missed (reused) | missed | caught, cause 24 |

**The miss comes from nesting, not from racing.** In each sequence the object goes back to the
allocator's free list, not to `free()`, so a heap is never told its lifetime ended and no change to
the heap alone can catch it; a race on an object straight from malloc would be caught by column 2,
as all 26 plain temporal cases are. The fix is column 3: the allocator's Sublet port, which revokes
on the allocator's own release -- all 23 nested lifetimes, the two races included. On the porting
deck these are the memcached rows where the `sublet` arm "returns": that arm is column 2. What is
not measured: revocation racing a real access on another hart -- every reduction here is one
thread.

### 0.2 What no column catches

10 spatial cases: the 9 struct-member crossings, which no code carves -- only the compiler's
field bounds narrow a member (7/9 on Capstone, 8/9 on CHERI, section 0b.1) -- and the frame plane,
whose read stays inside the 1024 bytes FFmpeg allocates for the alpha plane: the Sublet port of the
frame carve issues exactly that extent (measured, NOT CAUGHT; its bound and free controls fault),
and only a bound tighter than FFmpeg's own allocation would catch it.

### 0.3 Every bug

| program | corpus | case | axis | nested | CHERI (quarantine = caught) | Sublet in malloc | Sublet in nested |
|---|---|---|---|---|---|---|---|
| FFmpeg | carved-repros | 00_85407c7e63_mpegvideo_edge_emu_u_block_one_row_short | spatial | yes | missed | missed | caught |
| FFmpeg | carved-repros | 01_699341d647_apedec_array_0000_writes_64_into_channel_1 | spatial | yes | missed | missed | caught |
| FFmpeg | carved-repros | 02_cd7524fdd1_apedec_filter_3800_reads_past_short_channel | spatial | yes | missed | missed | caught |
| FFmpeg | carved-repros | 03_55937bb4a7_alsdec_ra_block_runs_into_next_channel_prefix | spatial | yes | missed | missed | caught |
| FFmpeg | carved-repros | 04_cd09284924_alsdec_mcc_chan_data_one_slot_per_channel | spatial | yes | missed | missed | caught |
| FFmpeg | carved-repros | 05_9d3032b960_alsdec_opt_order_past_max_order | spatial | yes | missed | missed | caught |
| FFmpeg | carved-repros | 06_2d0bea4719_vp9_uv_nnz_ctx_sized_for_420 | spatial | yes | missed | missed | caught |
| FFmpeg | carved-repros | 07_2563a33856_vp9_intra_pred_carved_for_old_bpp | spatial | yes | missed | missed | caught |
| FFmpeg | carved-repros | 08_b5ff61695f_swscale_v_line_at_pre_doubling_stride | spatial | yes | missed | missed | caught |
| FFmpeg | carved-repros | 09_043bcdcdb0_svq1enc_inter_block_runs_into_source | spatial | yes | missed | missed | caught |
| FFmpeg | carved-repros | 10_d2213b6493_rv34_b_block_carved_at_old_linesize | spatial | yes | missed | missed | caught |
| FFmpeg | carved-repros | 11_68226ed9ec_vorbis_type1_residue_end_spans_channels | spatial | yes | missed | missed | caught |
| FFmpeg | carved-repros | 12_ac59fc542f_thumbnail_hbd_tail_index_past_plane_slice | spatial | yes | missed | missed | caught |
| FFmpeg | plain-heap-repros | 00_d133b4a231_showcwt_kernel_scan_past_array | spatial | no | caught | caught | caught |
| FFmpeg | plain-heap-repros | 01_bcbf3a5630_vf_scale_format_list_compaction | spatial | no | missed | caught | caught |
| FFmpeg | plain-heap-repros | 02_56309e476a_vf_vif_mirror_below_base | spatial | no | caught | caught | caught |
| FFmpeg | plain-heap-repros | 03_495b402f27_diracdec_edge_emu_buffer_undersized | spatial | no | caught | caught | caught |
| FFmpeg | plain-heap-repros | 04_b3c7ebc1ed_swaprect_temp_sized_for_plane0 | spatial | no | missed | caught | caught |
| FFmpeg | plain-heap-repros | 05_8553e6ef57_cbs_av1_t35_payload_unpadded | spatial | no | missed | caught | caught |
| FFmpeg | plain-heap-repros | 06_041d4f010e_prores_raw_header_len_unchecked | spatial | no | caught | caught | caught |
| FFmpeg | plain-heap-repros | 07_8880a174d0_librist_read_ignores_caller_size | spatial | no | caught | caught | caught |
| FFmpeg | plain-heap-repros | 08_b2df2f4f22_mpegenc_system_header_fixed_128 | spatial | no | caught | caught | caught |
| FFmpeg | plain-heap-repros | 09_16b2049d4d_cfhd_transform2_wider_than_plane | spatial | no | caught | caught | caught |
| FFmpeg | plain-heap-repros | 10_989444060d5f_lut3d_size2_computed_before_directive | spatial | no | caught | caught | caught |
| FFmpeg | plain-heap-repros | 11_76645e096fab_exif_string_clone_drops_terminator | spatial | no | caught | caught | caught |
| FFmpeg | plain-heap-repros | 12_ad956ff076ea_drawtext_bbox_buffer_ignores_separators | spatial | no | caught | caught | caught |
| FFmpeg | plain-heap-repros | 14_789d7b1b1dff_ffv1dec_fltmap_index_unmasked | spatial | no | caught | caught | caught |
| FFmpeg | plain-heap-repros | 15_e9e6fb879835_tdsc_raw_tile_read_by_geometry_not_size | spatial | no | caught | caught | caught |
| FFmpeg | plain-heap-repros | 16_ca1c1f29ce47_img2enc_split_planes_read_by_geometry | spatial | no | caught | caught | caught |
| FFmpeg | plain-heap-repros | 17_18761f9fb55c_rtpdec_av1_output_cursor_advanced_for_skipped_obu | spatial | no | caught | caught | caught |
| FFmpeg | plain-heap-repros | 18_e8031e5b9ad2_showcwt_du_scroll_reads_row_past_plane | spatial | no | caught | caught | caught |
| FFmpeg | plain-heap-repros | 19_905a4324030e_showcwt_position_initialised_to_sono_size | spatial | no | caught | caught | caught |
| FFmpeg | plain-heap-repros | 20_bbdce45fda1e_diracdec_mctmp_rows_below_worst_case | spatial | no | caught | caught | caught |
| FFmpeg | plain-heap-repros | 21_3fbb9560821b_afir_crossfade_reads_beyond_input_frame | spatial | no | caught | caught | caught |
| FFmpeg | plain-heap-repros | 22_242ff799c75f_tdsc_cursor_advanced_by_a_stride_the_loop_already_added | spatial | no | caught | caught | caught |
| FFmpeg | plain-heap-repros | 23_f45da79b2c33_exr_dwa_block_written_full_eight_by_eight | spatial | no | caught | caught | caught |
| FFmpeg | plain-heap-repros | 24_162f75b5e679_exr_pxr24_uint_branch_strides_by_the_data_window | spatial | no | caught | caught | caught |
| FFmpeg | plane-repros | 00_b7946098b1_alphablend_row_past_plane | spatial | yes | missed | missed | missed |
| FFmpeg | subobject-repros | 00_8864fd0aec_cbs_h265_pic_timing_member | spatial | no | missed | missed | missed |
| FFmpeg | subobject-repros | 01_68845e26f7_vulkan_hevc_refpicset_member | spatial | no | missed | missed | missed |
| FFmpeg | subobject-repros | 02_e058af88ab_vulkan_hevc_dpb_member | spatial | no | missed | missed | missed |
| FFmpeg | subobject-repros | 03_89de2f0de1_aac_arith_last_member | spatial | no | missed | missed | missed |
| FFmpeg | subobject-repros | 04_1a00ea51cb_rtsp_control_url_underflow | spatial | no | missed | missed | missed |
| FFmpeg | subobject-repros | 05_d29ff88422_vulkan_av1_tile_sizes | spatial | no | missed | missed | missed |
| FFmpeg | subobject-repros | 06_a809a784ec_vvc_entry_point_start_ctu | spatial | no | missed | missed | missed |
| FFmpeg | subobject-repros | 07_275e217b10_hlsenc_key_uri_strlcpy_size | spatial | no | missed | missed | missed |
| FFmpeg | subobject-repros | 08_fb862976df_cbs_h266_col_width_val | spatial | no | missed | missed | missed |
| FFmpeg | plain-temporal-repros | 00_716d2a47c565_ops_dispatch_interior_alias_after_free | temporal | no | caught (held) | caught | caught |
| FFmpeg | plain-temporal-repros | 01_c98810ab47fa_hw_base_encode_list_walk_reads_freed_link | temporal | no | caught (held) | caught | caught |
| FFmpeg | plain-temporal-repros | 02_43de8b328b62_lzf_write_cursor_stale_after_realloc | temporal | no | caught (held) | caught | caught |
| FFmpeg | plain-temporal-repros | 03_dc87758775e2_aviobuf_checksum_ptr_stale_after_realloc | temporal | no | caught (held) | caught | caught |
| FFmpeg | plain-temporal-repros | 04_4b2248594c7f_jpeg2000_packed_headers_stream_stale_base | temporal | no | caught (held) | caught | caught |
| FFmpeg | plain-temporal-repros | 05_d6458f6a8bf1_aacdec_tag_che_map_keeps_freed_element | temporal | no | caught (held) | caught | caught |
| FFmpeg | plain-temporal-repros | 06_e8714f6f93d1_h264_delayed_pic_holds_interior_pointers | temporal | no | caught (held) | caught | caught |
| FFmpeg | plain-temporal-repros | 07_a43e9cdd442b_isom_extradata_freed_before_callee_frees_it | temporal | no | caught (held) | caught | caught |
| FFmpeg | plain-temporal-repros | 08_8a4ea9644833_diracdec_realloc_on_the_wrong_field | temporal | no | caught (held) | caught | caught |
| FFmpeg | plain-temporal-repros | 09_265731f201f1_tx_subcontext_field_left_dangling_on_failure | temporal | no | caught (held) | caught | caught |
| FFmpeg | plain-temporal-repros | 10_e7a65142b972_aacpsy_clears_the_local_not_the_field | temporal | no | caught (held) | caught | caught |
| FFmpeg | plain-temporal-repros | 11_ba28222a14ab_ratecontrol_expr_field_not_cleared | temporal | no | caught (held) | caught | caught |
| FFmpeg | plain-temporal-repros | 12_2e04d35c69e6_vlc_buf_freed_by_callee_then_read | temporal | no | caught (held) | caught | caught |
| FFmpeg | pool-repros | 00_461fb22053_af_join_dedup_bound | temporal | yes | missed | missed | caught |
| FFmpeg | pool-repros | 01_1886c3269d_h264_refs_partial_clear | temporal | yes | missed | missed | caught |
| FFmpeg | pool-repros | 02_316531e61c_vidstab_parked_plane_pointer | temporal | yes | missed | missed | caught |
| FFmpeg | pool-repros | 03_5c66a3ab51_vvc_nonref_output_releases_tabs | temporal | yes | missed | missed | caught |
| tshark | plain-heap-repros | 00_19c51d27b9_netscaler_record_past_page | spatial | no | caught | caught | caught |
| tshark | plain-heap-repros | 01_373504f7c9_dfvm_error_message_wrong_index | spatial | no | caught | caught | caught |
| tshark | plain-heap-repros | 02_381681583b_pcapng_nrb_custom_string_over_copy | spatial | no | caught | caught | caught |
| tshark | plain-heap-repros | 03_c556b648aa_strptime_reads_past_null_timezone | spatial | no | missed | caught | caught |
| tshark | plain-heap-repros | 04_87803328179_blf_apptext_sized_without_terminator | spatial | no | caught | caught | caught |
| tshark | plain-heap-repros | 05_140aad08e081_nettrace_packet_buf_scanned_by_strstr | spatial | no | caught | caught | caught |
| tshark | plain-heap-repros | 06_3aad1ef236e6_t61_accent_branch_reads_the_following_byte | spatial | no | caught | caught | caught |
| tshark | plain-heap-repros | 07_e2ca71beaed2_uat_unbinstring_leaves_no_terminator | spatial | no | caught | caught | caught |
| tshark | plain-heap-repros | 08_0cae98570ebc_cl_find_custom_memcmp_reads_the_prefix_length | spatial | no | caught | caught | caught |
| tshark | plain-heap-repros | 09_bf123efe154d_uat_oid_empty_field_underflows_the_index | spatial | no | caught | caught | caught |
| tshark | plain-heap-repros | 10_e9b933473e8f_address_to_str_buf_clamps_to_size_not_last_index | spatial | no | caught | caught | caught |
| tshark | plain-heap-repros | 11_4b15bf76a7f7_oid_to_str_buf_tail_reservation_one_short | spatial | no | caught | caught | caught |
| tshark | wmem-repros | 13_0261fd7da6_http_range_cursor_past_chunk | spatial | yes | missed | missed | caught |
| tshark | wmem-repros | 14_1d8acb21ab_solaredge_payload_six_past | spatial | yes | missed | missed | caught |
| tshark | wmem-repros | 15_d24613c461_opcua_padding_below_chunk | spatial | yes | missed | missed | caught |
| tshark | wmem-repros | 16_e8ef9df09d_dcp_etsi_rs_parity_write | spatial | yes | missed | missed | caught |
| tshark | wmem-repros | 17_5a560f3f6a_dns_one_byte_write | spatial | yes | missed | missed | caught |
| tshark | wmem-repros | 18_716a200295_rtps_batch_sample_info_unguarded | spatial | yes | missed | missed | caught |
| tshark | wmem-repros | 19_4a4871a831_ntlmssp_blob_length_before_check | spatial | yes | missed | missed | caught |
| tshark | wmem-repros | 20_ed20250c13_proto_undecoded_bitmap_unbounded | spatial | yes | missed | missed | caught |
| tshark | wmem-repros | 21_69dac89280_tcp_flags_str_sixteen_bytes | spatial | yes | missed | missed | caught |
| tshark | plain-temporal-repros | 00_f3c2e6087e7b_k12_callee_frees_its_own_argument | temporal | no | caught (held) | caught | caught |
| tshark | plain-temporal-repros | 01_7dcf69480de8_peak_trc_callee_frees_state_struct | temporal | no | caught (held) | caught | caught |
| tshark | plain-temporal-repros | 02_0fc7f3781351_wspstat_container_freed_before_its_contents | temporal | no | caught (held) | caught | caught |
| tshark | plain-temporal-repros | 03_012a179785ab_filesystem_alias_named_copy_is_not_a_copy | temporal | no | caught (held) | caught | caught |
| tshark | plain-temporal-repros | 04_07ffcf90426b_extcap_one_help_string_stored_in_many_owners | temporal | no | caught (held) | caught | caught |
| tshark | plain-temporal-repros | 05_fb46cda19602_wtap_close_inner_loop_rewinds_outer_cursor | temporal | no | caught (held) | caught | caught |
| tshark | plain-temporal-repros | 06_d3e3c00fbbe2_prefs_static_filter_label_not_cleared | temporal | no | caught (held) | caught | caught |
| tshark | plain-temporal-repros | 07_8dc7d164dcdb_prefs_reset_not_idempotent | temporal | no | caught (held) | caught | caught |
| tshark | plain-temporal-repros | 08_48a00fd55671_ftype_string_freed_twice_by_caller_and_callee | temporal | no | caught (held) | caught | caught |
| tshark | plain-temporal-repros | 09_cfc15838bdec_capture_ifinfo_out_parameter_left_unassigned | temporal | no | caught (held) | caught | caught |
| tshark | wmem-repros | 00_3c8be14c82_rpcrdma_write_offsets_global | temporal | yes | missed | missed | caught |
| tshark | wmem-repros | 01_c14d731e45_cms_oid_global | temporal | yes | missed | missed | caught |
| tshark | wmem-repros | 02_99da8c2cdc_mdb_address_column | temporal | yes | missed | missed | caught |
| tshark | wmem-repros | 03_6eab9f83ab_cola_info_column | temporal | yes | missed | missed | caught |
| tshark | wmem-repros | 04_b48759e4a4_qnet6_col_set_str | temporal | yes | missed | missed | caught |
| tshark | wmem-repros | 05_5a109265a6_usbll_address_struct | temporal | yes | missed | missed | caught |
| tshark | wmem-repros | 06_a8b16d74e1_x509if_last_dn_static | temporal | yes | missed | missed | caught |
| tshark | wmem-repros | 07_fb504bc76c_mysql_auth_method | temporal | yes | missed | missed | caught |
| tshark | wmem-repros | 08_31ab1a0a17_sip_cseq_method | temporal | yes | missed | missed | caught |
| tshark | wmem-repros | 09_693dc40936_geonw_proto_data_tvb | temporal | yes | missed | missed | caught |
| tshark | wmem-repros | 10_6fd3af5e99_t38_reassembly_buffer | temporal | yes | missed | missed | caught |
| tshark | wmem-repros | 11_3a5f82dfb5_http_header_map | temporal | yes | missed | missed | caught |
| tshark | wmem-repros | 12_90bb3a5c9e_xml_root_name_recycled | temporal | yes | missed | missed | caught |
| tshark | wmem-repros | 22_c702b44a01_usbhid_output_usages_freed_twice | temporal | yes | missed | missed | caught |
| memcached | allocator-repros | 05_2d61f18_item_data_one_past | spatial | yes | missed | missed | caught |
| memcached | allocator-repros | 06_78eb770_suffix_write_no_space | spatial | yes | missed | missed | caught |
| memcached | allocator-repros | 07_ecdb011_unterminated_key_read | spatial | yes | missed | missed | caught |
| memcached | allocator-repros | 08_e8364b5_ascii_all_spaces_scan_past_rbuf | spatial | yes | caught | caught | caught |
| memcached | plain-heap-repros | 00_ddee3e2_authfile_scan_past_calloc | spatial | no | missed | caught | caught |
| memcached | plain-heap-repros | 01_d5d9ff0_cachedump_end_marker_off_by_one | spatial | no | caught | caught | caught |
| memcached | plain-heap-repros | 02_391f2e4762bf_freesuffix_realloc_sized_in_bytes | spatial | no | caught | caught | caught |
| memcached | plain-heap-repros | 03_16a809e2a062_cache_create_freelist_sized_by_object | spatial | no | caught | caught | caught |
| memcached | plain-heap-repros | 04_40aff8b0f113_stats_end_marker_past_exact_buffer | spatial | no | caught | caught | caught |
| memcached | plain-heap-repros | 05_49f3b0ca9b57_out_string_crlf_copied_with_its_nul | spatial | no | caught | caught | caught |
| memcached | plain-heap-repros | 06_212c3820c7bb_key_hash_filter_tag_length_underflow | spatial | no | caught | caught | caught |
| memcached | plain-heap-repros | 07_0f605245cf3f_bin_delete_logs_unterminated_key | spatial | no | caught | caught | caught |
| memcached | plain-heap-repros | 08_fa51ad8452d5_slab_list_shuffle_reads_one_past | spatial | no | caught | caught | caught |
| memcached | allocator-repros | 00_7af02b0c87_rbuf_copied_after_cache_free | temporal | yes | missed | missed | caught |
| memcached | allocator-repros | 01_0ad4de66ae_io_walk_reads_freed_link | temporal | yes | missed | missed | caught |
| memcached | allocator-repros | 02_59bd02ce29_tail_repair_frees_referenced_item | temporal | yes | missed | missed | caught |
| memcached | allocator-repros | 03_a8c4a82787_refcount_overflow_frees_linked_item | temporal | yes | missed | missed | caught |
| memcached | allocator-repros | 04_152ddb68f7_unlocked_refcount_drift | temporal | yes | missed | missed | caught |
| memcached | plain-temporal-repros | 00_0d4901071c74_restart_line_freed_at_entry_but_not_cleared | temporal | no | caught (held) | caught | caught |
| memcached | plain-temporal-repros | 01_e7793811f8c8_logger_write_to_watcher_freed_by_the_poll | temporal | no | caught (held) | caught | caught |
| memcached | plain-temporal-repros | 02_3bc58f6ea55a_logger_loop_condition_still_reads_the_freed_watcher | temporal | no | caught (held) | caught | caught |

## 0b. Every arm, all 130 defects (after the audits of 2026-10-10 and the case added on 2026-10-11)

Generated by `capstone/bug-corpora/tools/catch-tables.py --markdown`, which reads only `case.json`:
an arm's explicit `verdict`, or the opening words of a `MEASURED …:` oracle; since the audit a bare
oracle is a prediction and a hole. It exits 1 — printing each hole by name — while any cell is not a
reading; it printed 334 when made strict, and none now. A cell reads **caught / measured**.

The columns are one mechanism each. **ASan**: the native build under AddressSanitizer, with
positive controls reporting in the same run. **CheriBSD**: stock purecap, libc revocation on.
**Capstone
bounds**: on plain rows the level0 heap (malloc narrowed to the request); on NESTED rows the
allocator's port in mode 0 -- each chunk its own bound (wmem's `wm_narrow`, memcached's ledger), no
revocation -- so those cells are a port's per-chunk bound, not malloc's (corrected at the audit: this
sentence used to say level0 for every row). The stock allocators on a per-allocation bound are
column 2 of section 0: wmem 0/9 and memcached nested spatial 1/4. **Capstone + Sublet**: the Sublet
heap, and on the nested corpora the allocator's Sublet port.

**Temporal (49)**

| program | allocator layer | axis | n | ASan | CheriBSD | Capstone bounds | Capstone + Sublet |
|---|---|---|---:|---:|---:|---:|---:|
| FFmpeg | AVBufferPool | nested | 4 | 0 / 4 | 0 / 4 | 0 / 4 | 4 / 4 |
| FFmpeg | direct malloc | plain | 13 | 13 / 13 | 0 / 13 | 0 / 13 | 13 / 13 |
| tshark | wmem | nested | 14 | 0 / 14 | 0 / 14 | 0 / 14 | 14 / 14 |
| tshark | direct g_malloc | plain | 10 | 10 / 10 | 0 / 10 | 0 / 10 | 10 / 10 |
| memcached | slabs.c / cache.c | nested | 5 | 0 / 5 | 0 / 5 | 0 / 5 | 5 / 5 |
| memcached | direct malloc | plain | 3 | 3 / 3 | 0 / 3 | 0 / 3 | 3 / 3 |
| **Total** |  | 23 n · 26 p | 49 | 26 / 49 | 0 / 49 | 0 / 49 | 49 / 49 |

**Spatial (81)**

| program | allocator layer | axis | n | ASan | CheriBSD | Capstone bounds | Capstone + Sublet |
|---|---|---|---:|---:|---:|---:|---:|
| FFmpeg | carved buffer | nested | 13 | 0 / 13 | 0 / 13 | 0 / 13 | 0 / 13 |
| FFmpeg | frame-pool plane | nested | 1 | 0 / 1 | 0 / 1 | 0 / 1 | 0 / 1 |
| FFmpeg | direct malloc | plain | 24 | 24 / 24 | 21 / 24 | 24 / 24 | 24 / 24 |
| FFmpeg | inside one struct | plain | 9 | 0 / 9 | 0 / 9 | 0 / 9 | 0 / 9 |
| tshark | wmem | nested | 9 | 0 / 9 | 0 / 9 | 9 / 9 | 9 / 9 |
| tshark | direct g_malloc | plain | 12 | 12 / 12 | 11 / 12 | 12 / 12 | 12 / 12 |
| memcached | slabs.c / cache.c | nested | 4 | 1 / 4 | 1 / 4 | 2 / 4 | 2 / 4 |
| memcached | direct malloc | plain | 9 | 9 / 9 | 8 / 9 | 9 / 9 | 9 / 9 |
| **Total** |  | 27 n · 54 p | 81 | 46 / 81 | 41 / 81 | 56 / 81 | 56 / 81 |

How each reads, and why it misses what it misses:

- **ASan sees exactly the accesses that leave a malloc'd block** (since the audit, on each program's own
  backing: memcached's pages and cache.c objects each their own malloc, wmem's `g_malloc` the host's
  `malloc` -- the earlier arms carved everything out of one arena and could read nothing but silence, and
  memcached 8, a crossing past a cache.c object, now reports): plain spatial 45/54 (the 9
  struct-member crossings stay inside one block), plain temporal 26/26, and 1/27 nested spatial with
  0/23 nested temporal -- memcached 8 is the only nested crossing that leaves a malloc'd block, because
  upstream mallocs each cache.c object on its own; an arena, a pool or a carve is one block to ASan. Every silent row ran with controls that
  reported a read past, and a read after free of, a block the size of that corpus's arena.
- **CheriBSD misses five plain spatial rows to size-class slack**: libc rounds requests to size
  classes (17 and 24 bytes to 32), so those crossings stay inside the capability. On temporal, its
  quarantine withholds the freed chunk (no fault, no reuse).
- **The two PoisonCap columns** (mode 0 and mode 1 of the published PoisonCap platform, through
  each allocator's PoisonCap adapter on the nested rows) were removed on 2026-10-10 with the
  adapters. Their readings stay in each corpus's `results/`.
- **Capstone bounds and Capstone + Sublet** are unchanged in kind: exact per-object bounds catch
  every plain spatial row, including the five CheriBSD slack misses; Sublet adds the temporal
  column (49/49), and nothing spatial, because a spatial bug frees nothing.

### 0b.1 Inside one allocation: why the Sublet heap misses, and what catches it

Twenty-five spatial cases never leave the malloc'd object, so no allocation-granular mechanism in
the tables above sees them, Sublet included. Two remedies narrow below the allocation:
**field bounds**, set by the compiler at a struct member (`-Xclang -fcapstone-subobject-bounds`,
C1 v1, non-last array fields; CHERI clang `-cheri-bounds=subobject-safe`), and **carve bounds**,
set by the code that cuts one buffer into regions (`ffc_carve`, `mc_carve`, `ffp_carve`, each the
identity unless its switch is defined, so every other arm builds unchanged).

| inside one allocation | n | Capstone + Sublet | field bounds, Capstone | field bounds, CHERI | carve bounds, Capstone | carve bounds, CHERI | **Sublet carve** |
|---|---:|---:|---:|---:|---:|---:|---:|
| struct member (FFmpeg subobject-repros) | 9 | 0 / 9 | 7 / 9 | 8 / 9 | — | — | — |
| carved buffer (FFmpeg carved-repros) | 13 | 0 / 13 | 0 / 13 | 0 / 13 | 13 / 13 | 13 / 13 | 13 / 13 |
| slab item key / suffix (memcached allocator 06, 07) | 2 | 0 / 2 | 0 / 2 | 0 / 2 | 2 / 2 | 2 / 2 | 2 / 2 |
| frame plane (FFmpeg plane-repros) | 1 | 0 / 1 | 0 / 1 | 0 / 1 | 1 / 1 | 1 / 1 | 0 / 1 |

- **Field bounds catch member crossings and nothing else.** Struct members: 7/9 on Capstone, 8/9
  on CHERI. Still missed: 04 (a read one byte before a member — predicted caught on CHERI, refuted)
  on both, and 06 on Capstone only, because C1 v1 does not narrow a struct's last array member.
  (09, an index inside one int array, was a slice carved out of one `av_calloc`; since the audit it is
  carved case 12, which carve bounds catch.) On carved regions, slab items and the frame plane they catch
  nothing: those are pointer arithmetic on a block, not fields.
- **The Sublet carve** (column 3 for these rows; `sublet-carve`) is the carving code ported to Sublet: the
  block lent LINEAR by the Sublet heap and split into one Sublet region per carve, each issued as an
  alias bounded to the carve. It catches the 13 carved crossings and, as the slab port plus the
  key/suffix carve, memcached 6 and 7 -- **by the bound it sets, the mechanism of carve bounds**; no
  buggy sequence re-carves, and its revocation is shown by the re-carve control alone. It does NOT
  catch the frame plane: a faithful port of av_frame_get_buffer issues each plane its allocated
  extent (padded_height rows, 1024 bytes for the alpha plane), and the read at offset 160 is inside
  it. The 160-byte `ffp_carve` above is tighter than FFmpeg's own allocation and faults legitimate
  padded reads; it is a remedy for this read, not a port. Struct members are not carved by any code,
  so only the compiler's field bounds reach them (dash).
- **Carve bounds catch every carved crossing: 16 of 16 on both platforms**, at the labelled probe,
  with exact granted bounds on CHERI. Where the runners execute the fixed arm (the carved and plane
  corpora) it ran under the same narrowing and stayed FIXED. memcached case 6's region is empty;
  CHERI takes a zero-length bound, and Capstone — whose shrink needs base < end — the byte before it.

The pre-registrations: `2852edcdd747` (carved corpus, PoisonCap), `49498f856e87` and
`886abf4cc336` (field bounds), `83dcfa032845` (carve bounds for slab and plane). Refuted and
recorded in their cases: FFmpeg plain-temporal 12 and tshark plain-temporal 04 on PoisonCap mode 1, FFmpeg subobject 04 under
CHERI field bounds, and wmem 13's PoisonCap rows (the wmem adapter bounds each chunk).

## 0c. The same three columns on virtual Capstone (2026-10-10/11)

Generated by `capstone/bug-corpora/tools/catch-tables.py --board-virtual`. Column 1 is the physical CHERI
column, unchanged. Columns 2 and 3 are the virtual profile (`capstone/runtime/virtual`, the exact-bounds
QEMU): the program's allocator stock on virtual mallocng (`virtual-malloc`), and the nested allocator's
Sublet port over a block the virtual heap lends linear (`virtual-nested-pools`). A plain case has no
nested allocator, so its column 3 is the same run as column 2. Every cell comes from a derived
`verdicts.py` bundle, with the configuration's controls in the same boot. `(+N ?)` cells are not
readings: FFmpeg's pool and plane corpora have no virtual recipe yet (`build-virtual.sh` refuses the
app's pool variants, and plane needs a virtual libavutil).

**Temporal, virtual Capstone**

| program | allocator layer | axis | n | CHERI (quarantine = caught) | virtual malloc | virtual nested |
|---|---|---|---:|---:|---:|---:|
| FFmpeg | AVBufferPool | nested | 4 | 0 / 4 | 0 / 0 (+4 ?) | 0 / 0 (+4 ?) |
| FFmpeg | direct malloc | plain | 13 | 13 / 13 (13 held) | 13 / 13 | 13 / 13 |
| tshark | wmem | nested | 14 | 0 / 14 | 0 / 14 | 14 / 14 |
| tshark | direct g_malloc | plain | 10 | 10 / 10 (10 held) | 10 / 10 | 10 / 10 |
| memcached | slabs.c / cache.c | nested | 5 | 0 / 5 | 0 / 5 | 5 / 5 |
| memcached | direct malloc | plain | 3 | 3 / 3 (3 held) | 3 / 3 | 3 / 3 |
| **Total** |  |  | 49 | 26 / 49 (26 held) | 26 / 45 (+4 ?) | 45 / 45 (+4 ?) |

**Spatial, virtual Capstone**

| program | allocator layer | axis | n | CHERI (quarantine = caught) | virtual malloc | virtual nested |
|---|---|---|---:|---:|---:|---:|
| FFmpeg | carved buffer | nested | 13 | 0 / 13 | 0 / 13 | 13 / 13 |
| FFmpeg | frame-pool plane | nested | 1 | 0 / 1 | 0 / 0 (+1 ?) | 0 / 0 (+1 ?) |
| FFmpeg | direct malloc | plain | 24 | 21 / 24 | 24 / 24 | 24 / 24 |
| FFmpeg | inside one struct | plain | 9 | 0 / 9 | 0 / 9 | 0 / 9 |
| tshark | wmem | nested | 9 | 0 / 9 | 0 / 9 | 9 / 9 |
| tshark | direct g_malloc | plain | 12 | 11 / 12 | 12 / 12 | 12 / 12 |
| memcached | slabs.c / cache.c | nested | 4 | 1 / 4 | 1 / 4 | 4 / 4 |
| memcached | direct malloc | plain | 9 | 8 / 9 | 9 / 9 | 9 / 9 |
| **Total** |  |  | 81 | 41 / 81 | 46 / 80 (+1 ?) | 71 / 80 (+1 ?) |

virtual cells that are not a reading yet: 10

The virtual columns agree cell for cell with the physical columns 2 and 3 wherever both exist:
- wmem: 0/23, then 23/23;
- memcached allocator: 1/9, then 9/9;
- FFmpeg carved: 0/13, then 13/13;
- every plain case: CAUGHT.

The one place the platforms differ in kind is the bound. Virtual mallocng bounds each object exactly,
so the five size-class slack cases CheriBSD misses are caught on both Capstone platforms.

### Earlier on 2026-10-09: the Capstone columns closing, 118 cases (history; counts below predate the audit of 2026-10-10)


Counted from `case.json` by one classifier that reads an arm's explicit `verdict`, or the opening
words of its oracle (`fault…` = caught; `complete…` / `the sequence completes` = not caught;
CheriBSD's `MEASURED …: **CAUGHT**` / `**NOT CAUGHT**` / `**NO FAULT**`), and prints any arm it
cannot classify by name instead of counting it. Before today's runs it reproduced the known state
exactly (CheriBSD 42/70 spatial, 0/48 temporal; Capstone measured on 33 cases). It now finds no
arm unrun and none unclassified.

**Temporal (48)**

| cell | n | CheriBSD (revocation on) | Capstone, bounds only | Capstone + Sublet |
|---|---:|---:|---:|---:|
| nested | 22 | 0 | 0 | **22** |
| plain | 26 | 0 | 0 | **26** |
| **total** | **48** | **0** | **0** | **48** |

**Spatial (70)**

| cell | n | CheriBSD (revocation on) | Capstone, bounds only | Capstone + Sublet |
|---|---:|---:|---:|---:|
| nested | 14 | 1 | 11 | 11 |
| plain | 56 | 41 | 46 | 46 |
| **total** | **70** | **42** | **57** | **57** |

How each mechanism reads, and why it misses what it misses:

- **CheriBSD, temporal 0/48, for two different reasons.** Nested (22): the stale storage goes back
  on an inner allocator's free list inside a block malloc still owns, nothing reaches `free()`, and
  the revoker never sees it; the inner allocator reissues it and the stale pointer reads another
  object. Plain (26): the object does reach `free()`, and libc's quarantine **withholds** the chunk
  until a sweep — so nothing is reissued and nothing aliases (verdict NOT-REISSUED), but nothing
  faults either, because no sweep runs between the free and the access. The same boot's revocation
  control (forced sweep, then `tag_after_sweep=0` and SIGPROT) shows the revoker acts.
- **CheriBSD, spatial 42/70.** Its malloc bounds a capability to a SIZE CLASS, not to the request:
  5 plain crossings land in that slack and are not caught (ffmpeg plain-heap 01, 04, 05; tshark
  plain-heap 03; memcached plain-heap 00 — `tools/size-class-audit.py` predicts all five from the
  interposed sizes). 10 cross between members of ONE allocation (`ffmpeg/subobject-repros`). 13 of
  14 nested crossings stay inside the block malloc handed the inner allocator.
- **Capstone bounds only (`spatial` arm), temporal 0/48.** An exact bound cannot see a dead object:
  the stale capability still carries the freed object's bounds. On the plain cases the level0 heap
  reissues the same block and the stale pointer reads the new object (DEFECT-REPRODUCED, 26/26).
- **Capstone bounds only, spatial 57/70.** level0 narrows every malloc to the bytes requested, so
  the five CheriBSD slack misses are caught (cause 5 on a load, 7 on a store, at the labelled
  probe). Inner allocators that narrow per object (wmem's `wm_narrow`, the memcached slab port) are
  caught too: wmem 13-21 and memcached allocator 05 and 08. Misses: the 10 sub-object crossings
  (inside one allocation, by each case's own CHECKs), memcached allocator 06-07 (inside one slab
  chunk), and the FFmpeg plane case (inside upstream's own padded plane).
- **Capstone + Sublet, temporal 48/48.** `free` (or the inner allocator's release, on the Sublet
  ports of FFmpeg's pools, wmem and memcached's slabs) revokes the object, and the stale access
  faults with **cause 24** at the labelled probe. **QEMU only:** capstone-qemu reloads a revoked
  capability untagged (ISSUES Q-11); deployed silicon lets such a data access retire, so these are
  detections on the emulator, not a silicon claim. wmem case 12 is caught by the chunk port
  (`sublet-chunks`); the region-granular `sublet` build completes it.
- **Capstone + Sublet, spatial 57/70 — the same 57 as bounds only.** Revocation has nothing to fire
  on while the object is alive; every spatial catch on this arm is the inherited per-object bound.

Where the readings come from (2026-10-09 unless noted):

| corpus | Capstone evidence |
|---|---|
| the six plain-heap / plain-temporal corpora, `ffmpeg/subobject-repros`, `ffmpeg/plane-repros` | `results/2026-10-09-capstone/`, runner `tools/run-capstone-domain.py`, predictions pre-registered at `60f1c5be25a5` |
| `wireshark/wmem-repros` 18-21 | `results/20261009-qemu-capstone-18-21/` (00-17: earlier runs in the same folder) |
| `memcached/allocator-repros` 08 | `results/20261009-qemu-capstone-case8/` (00-07: 2026-10-05) |
| `ffmpeg/pool-repros` | 2026-09-25 / 2026-09-29 runs recorded in its cases |

Three readings did not go as written and are recorded as such in their cases:

- **memcached allocator 08** was predicted to fault at its labelled probe. It faults ONE STATEMENT
  EARLIER, in the upstream defective scan itself, on the first byte past the 16384-byte object
  (bounds and address from the emulator's own fault line). The corpus runner scores that row FAIL,
  because it requires the fault at the probe after the case marker; the FAIL is this reading.
- **wmem 18-21** were filed with an oracle that named neither probe nor cause; the runner's
  defaults (read probe, temporal causes) scored all 8 new arms FAIL on the first run. The oracles
  were completed from each `case.c` (which probe it calls; store = 7, load = 5) and the re-run on
  the same images reads 12/12 with both controls.
- **wmem's negative control** could not run from 2026-10-05 19:32: the empty-boot guard added then
  refused the boot the control is designed to produce. Fixed and negative-tested; 12/12 FAIL as
  required.

The subobject arm has one more limit, stated in its cases: in those images `av_malloc` is the
buffer-pool port's arena carve, which gives an object no bound of its own, so the run cannot
separate "an allocation-granular bound cannot see a member crossing" (which the cases' CHECKs
establish) from "nothing bounded the struct". It establishes that every case runs to its defect in
a Capstone domain on both arms with no fault anywhere.

## 1. The temporal 27 were all there was, because spatial had been filtered out — no longer true

> **HEADING CORRECTED 2026-10-07.** It read *"The 27 real upstream defects are all temporal"*, which
> was the state when this document was written and contradicted its own table below as soon as the
> spatial hunt produced anything. The spatial corpus is now **32 built and measured cases**, so the
> sentence had become false in the one place a reader looks first.

> **HISTORY -- counts as of 2026-10-09, before the audit of 2026-10-10.** The current counts are in
> section 0: 129 defects, spatial 27 nested / 54 plain, temporal 22 nested / 26 plain. FFmpeg
> plain-heap 13 is a duplicate and subobject 09 moved to carved-repros as case 12 (section A).

| | nested allocator | plain / system allocator | total |
|---|---:|---:|---:|
| **spatial**, built and measured | **26** | **56** | **82** |
| **temporal**, built and measured | **22** | **26** | **48** |
| **total** | **48** | **82** | **130** |

Per program, recomputed from each case's `nested` boolean and its `lifetime_ender`:

| program | spatial / nested | spatial / plain | temporal / nested | temporal / plain | total |
|---|---:|---:|---:|---:|---:|
| FFmpeg | 13 | 35 | 4 | 13 | **65** |
| tshark | 9 | 12 | 13 | 10 | **44** |
| memcached | 4 | 9 | 5 | 3 | **21** |

> **UPDATED 2026-10-09.** FFmpeg's nested spatial cell read **1** (`plane-repros/00`). It is 13: the
> new `ffmpeg/carved-repros` holds twelve upstream fixes where a codec carved ONE allocation into
> regions by pointer arithmetic and an access ran from one region into the next without leaving the
> allocation. The search and its rejected candidates are in `ffmpeg-spatial-defect-triage.md`
> (the 2026-10-09 section); case 11 (vorbis) is live at the pin by source reading.

> **UPDATED 2026-10-08.** The table above read `14 | 18 | 32` spatial and `22 | 5 | 27` temporal
> yesterday, with **temporal / plain = 0 for all three programs**. That zero was a property of
> which CORPORA existed, not of the upstream software: every temporal corpus in this tree sat on a
> nested allocator — FFmpeg's AVBufferPool and AVRefStructPool, Wireshark's wmem, memcached's
> slabs.c and cache.c — because that is what the temporal hunts were aimed at. All three programs
> also free direct allocations and use them afterwards, and there was nowhere to record one.
>
> Three `plain-temporal-repros` corpora were created and filled (FFmpeg 13, tshark 10, memcached 3),
> and the plain spatial row grew by 38.
>
> **The baseline, with its denominator named, because this document's own history shows how easily
> two differently-counted totals get compared.** All three figures below count CASES in these three
> programs, from `case.json` files, on the same basis as the table above:
>
> | state of the branch, by its last commit that day | FFmpeg | tshark | memcached | total |
> |---|---:|---:|---:|---:|
> | end of 2026-10-05 (`84990ea0ade5`) | 7 | 18 | 8 | **33** |
> | end of 2026-10-06 (`39f62dfb0354`) | 19 | 19 | 9 | **47** |
> | end of 2026-10-07 (`f2c70e991714`) = start of 2026-10-08 | 19 | 24 | 11 | **54** |
> | end of 2026-10-08 (`c3931286386f`) | 53 | 44 | 21 | **118** |
>
> Each row is counted from GIT -- `git ls-tree` of that commit, one row per `case.json` -- not
> from a table written at the time. So 2026-10-08's work is **54 -> 118, +64**, and the week's
> is **33 -> 118, +85**.
>
> **CORRECTED 2026-10-08.** An earlier version of this table labelled the 54 row "the inventory of
> 2026-10-06" and added an intermediate 62 row as the day's baseline, concluding "+56 today". Both
> were wrong. The 54 figure was right but its DATE was not -- it is the state at the end of
> 2026-10-07, while the end of 2026-10-06 was 47 -- and 62 was a point in the middle of
> 2026-10-08, after the day's first eight cases, so it is not a baseline for anything. The figures
> had been carried from a plan written earlier the same day instead of being read from history,
> which is exactly the mistake the paragraph above this table warns against.
>
> The counts are now COMPUTABLE rather than asserted: 22 temporal cases carried no `nested`
> boolean at all — the field postdates them — and were backfilled from each case's own
> `allocator_layer`, so no case is left in an "unclassified" bucket. A script that put 11 of 25
> spatial cases into such a bucket on 2026-10-06 reported a nesting share wrong by 16 points, which
> is why the boolean exists.

> **UPDATED 2026-10-07.** The spatial row read *"8 | 3 | 11"* until today. It is now **14 | 18 | 32**:
> FFmpeg went 4 -> 15 (2026-10-06), tshark 6 -> 11 and memcached 4 -> 6 (2026-10-07). The
> not-nested column grew most, because requiring liveness at the pin had been keeping it thin — a
> requirement no document ever asked for, and the retraction of that inference is what unblocked the
> growth. Counts are recomputed from each case's `nested` boolean, which exists so this row cannot
> drift from the tree again.
>
> **CORRECTED 2026-10-05.** The spatial row previously read *"11 built and measured | 0"*, putting
> every built case under *nested allocator*. **That is wrong, and the axis was the problem.** This
> table's axis is **who allocated the object**; §1a's A/B/C axis is **which bound the access
> crosses**. They are orthogonal, and I had collapsed them.
>
> On *this* table's axis: memcached cases 6 and 7 are **nested** — their objects are `slabs.c`
> chunks — even though the bound they cross is a sub-object one inside the chunk. FFmpeg cases 0-2
> are **plain**: `av_refstruct_alloc_ext` is called directly at `cbs_sei.c:257` and
> `decode.c:2352`, *not* from a pool, so the object is one `av_malloc` with no recycling layer.
> Hence **8 nested, 3 plain**, not 11 and 0.
>
> The sub-object distinction is kept in §1a because it is what decides *detectability*, which is a
> different question from *who allocated*.

**Every defect reduced from real upstream code in these three programs is temporal.**

> **RETRACTED 2026-10-05.** This section first continued *"This is not a measurement gap — it is
> what the defect hunt found, and it is the thesis the target evaluation rests on."* **That is
> false, and the refutation is in this tree.** The hunt **could not** have found a spatial defect,
> because spatial wording was a *disqualifier* on its first filter:
>
> - `wireshark-wmem-defect-triage.md:23` — filter 1 is *"the commit message reads as a lifetime
>   defect — use-after-free, freed, stale, dangling — **and not as an overflow**, a leak or a denial
>   of service"*.
> - `ffmpeg-live-defect-triage.md:26` — filter 1 is *"lifetime wording in the subject — 15 of 1,788
>   survive"*.
> - `bug-corpora/memcached/allocator-repros/README.md:52-57` — *"filtered on temporal-safety
>   vocabulary (55 hits)"*.
> - `wireshark-wmem-defect-triage.md:178` — **29 rows** of the 4.6.9 security tracker rejected en
>   bloc as *"spatial or availability"*.
> - `ffmpeg-pool-consumer-defects.md:51-54` — the population was *measured* to be **"genuinely
>   spatial-dominated — "overflow" 2,415, "out of array" 1,062"* — counted in aggregate, then never
>   read case by case.
>
> So the 0 is a property of the **search**, not of the software. A tree-wide census agrees: **0 of
> 78 `case.json` files in `bug-corpora/` carry a spatial shape**, across all eight programs — the
> signature of a single-axis hunt, not of eight clean codebases. A spatial-wording filter over the
> same populations yields **71 candidates on filter 1 alone** (wireshark 33 of 4,321, FFmpeg 17 of
> 1,786, memcached 21 of 2,349).
>
> The honest statement is: **spatial defects were excluded by design and never triaged
> individually.** Five were met incidentally and rejected with reasons — memcached `#1308`
> `raw_line()` (rejected on *reachability*, not class, `allocator-repros/README.md:79`), the 29
> Wireshark tracker rows, opcua `d24613c461`, vp9 `a024f8c541`, tdsc `fd3ee52fab`.
>
> **The hunt has since run. Its result is in §1a below and it is not zero.**

### 1a. What the spatial hunt found (2026-10-05)

Three new instruments, one per program, in `docs/ref/{wireshark,memcached,ffmpeg}-spatial-defect-triage.md`,
driven by `bug-corpora/tools/spatial-triage.py`. The classification that matters is **which bound
the overflow crosses**, because that is what decides whether anything of ours can see it:

| class | bound crossed | who faults |
|---|---|---|
| **A** | the `malloc` bound | `shrink`, `sublet`, CHERI alike — a tie row |
| **B** | a sub-allocation bound **inside a nested allocator's block** | only a ported nested allocator |
| **C** | a **sub-object** bound inside ONE allocation | nothing we have; needs per-member authority |

| program | population | filter 1 | class B | class C | live at pin |
|---|---:|---:|---:|---:|---|
| tshark | 96,806 | 303 | **15** | 0 | **2** (`1d8acb21ab`, `d24613c461`) |
| memcached | 2,349 | 20 | **0** | 0 | 0 — pin *is* upstream head |
| FFmpeg | 1,786 + history | 19 + shape search | 0 | **4** | **4** (`8864fd0aec`, `a809a784ec`, `68845e26f7`, `e058af88ab`) |

**So the spatial row is not empty: 19 class-B/C upstream defects, 6 of them live at their pin.**
Zero are built as corpus cases yet — see the honest comparison below.

**The three programs are blind for three different reasons**, which is the finding worth carrying:

- **tshark — unwired allocator coverage.** The live pair overflows a chunk in `pinfo->pool`, a wmem
  `BLOCK_FAST` allocator, and the chunk port's only wiring patch targets `wmem_block.c`; no patch
  against `wmem_block_fast.c` exists. **But 4 of the 15 are in `wmem_file_scope()`, a `BLOCK`
  allocator the port already narrows** — `0261fd7da6` (HTTP Range, also a whitelisted dissector),
  `d7d1686a95`, `1c090e9292`, `4a4871a831`. Those need no port work to discriminate.
- **memcached — structurally absent.** An item is sized exactly from the key and value lengths at
  `do_item_alloc` with the parser validating them first, so the length bugs land one layer out in
  the proxy's own `malloc`'d buffers. The four `slabs.c`/`items.c` candidates index **static global
  arrays**, not heap.
- **FFmpeg — sub-object granularity, not an allocator problem at all.** All four live defects cross
  a bound *between two members of one struct* inside a single `av_mallocz` or refstruct. No
  allocator adapter can help; the authority that would need narrowing is per struct member. That is
  the taxonomy's `partial²` cell, where CHERI and Capstone already share a verdict.

**Comparison with the temporal hunt, which produced 22 built cases:** this hunt has produced **19
triaged candidates and 11 BUILT, MEASURED cases** — tshark 5, memcached 3, FFmpeg 3. Eleven is not
parity with 22, and this is not the place to imply it is.

| program | built | corpus | measured how | live at pin |
|---|---:|---|---|---:|
| tshark | **5** | `wireshark/wmem-repros` cases 13-17 | QEMU, 12/12 on each of two builds, negative control 12/12 | 2 |
| memcached | **3** | `memcached/allocator-repros` cases 5-7 | **QEMU domain, 8/8, negative control 8/8**, plus native fix-differential 8/8 | 0 |
| FFmpeg | **3** | `ffmpeg/subobject-repros` cases 0-2 *(new corpus)* | **QEMU as probe cases 40-42, PASS/PASS on modes 0 and 2**, plus native 3/3 and ASan blind two-sided | 3 |

**The result is not the count — it is that the three programs divide on WHO CAN SEE these defects**,
and the division was measured rather than argued:

- **tshark's five all fault**, cause **5** on the three reads and cause **7** on the two writes, and
  they do **not** discriminate the chunk port: `wm_narrow()`
  (`ports/wireshark/wmem/src/shared/wmem-port-hooks.h:11-15`) narrows every wmem allocation on every
  arm, so that harness has no malloc-granular arm to contrast against. The contrast is the app
  port's fx12 ladder in §3.
- **FFmpeg's three are caught by NOTHING**, measured: each crosses a bound *between two members of
  one allocation*, so every per-allocation bound is in bounds for it, and ASan is blind with a
  positive control that fires. The `partial²` cell of §4.
- **memcached's three are native-only**: their Capstone readings are declared predictions pointing
  deliberately different ways, so a future reading settles something.

**The measurement was CLOSED over the ELEVEN cases this section counts.**
*(Two further spatial corpora landed on 2026-10-06 — `memcached/plain-heap-repros`
and `ffmpeg/plane-repros` — each with its own result bundle; the totals in
`spatial-and-temporal-bug-inventory.md` are the current ones.)*
Every one of the 80 arm cells across those eleven cases is
accounted for**, and the arithmetic reconciles from the case files rather than from a summary:

| | measured | unavailable | declined | n/a | total |
|---|---:|---:|---:|---:|---:|
| tshark (5 cases × 7 arms) | **15** | 15 | 5 | 0 | 35 |
| memcached (3 × 7) | **9** | 9 | 3 | 0 | 21 |
| FFmpeg (3 × 8) | **12** | 9 | 0 | 3 | 24 |
| **total** | **36** | **33** | **8** | **3** | **80** |

- **measured** — every Capstone arm of all eleven cases, plus the native fix-differentials and
  FFmpeg's ASan arm. **All eleven cases are now measured under Capstone**, not six of them.
- **unavailable** — PoisonCap and CheriBSD, 3 arms × 11 cases. Checked, not assumed:
  `ports/common/cmake/toolchains/cheribsd.cmake:4-9` requires `CHERI_SDK` and `CHERI_SYSROOT` and
  `FATAL_ERROR`s without them; both are unset even after sourcing the project environment, and no
  SDK, rootfs or PoisonCap image exists anywhere on this host.
- **declined** — the eight `native-detect` (ASan) arms, with a reason that is itself a measurement:
  the FFmpeg sub-object probe already shows the two-sided shape (silent inside the allocation, fires
  one element past it), and every crossing in those corpora stays inside one `g_malloc`'d block or
  slab page by the same mechanism.
- **n/a** — FFmpeg's `backing` arm: there is no block distinct from the object, because
  the object *is* one `av_malloc`.

### What the closed measurement shows, per program

- **tshark, 5 cases, all faulting.** Cause **5** on the three reads, cause **7** on the two writes,
  on **both** builds. They do **not** discriminate the chunk port: `wm_narrow()` narrows every wmem
  allocation on every arm, so the harness has no malloc-granular arm. 12/12 per build, negative
  control 12/12.
- **memcached, 3 cases, and the decisive one.** Case 5 **faults** (cause 7, write probe); cases 6 and
  7 **complete on both modes**. Case 6 is the result: the defect is real and corrupts the value's
  storage, the crossing stays **inside** the chunk, the slab port's bound *is* the chunk, and so
  **nothing we have detects it**. 8/8, negative control 8/8 fired.
- **FFmpeg, 3 cases, caught by nothing — now measured, not declared.** Probe cases 40-42 of the
  buffer-pool port, **PASS/PASS on modes 0 and 2**, runner exit 0 each. Each probe asserts the
  crossing happened, so a completion is a measurement and not a quiet nothing.

**Nine pre-registered predictions were refuted across this work** and each is recorded where it was
made, never silently corrected. The last four: case 13's completion predictions; the five tshark
rows' cause 5 (a store faults 7); memcached's *"0 class-B, structural"* verdict; and — the only one
that went the other way — memcached's three Capstone predictions, which **held**, including the one
that mattered.

### The two decisions that remain, and they are the lead's

1. **Wire `BLOCK_FAST` into the chunk adapter.** It would turn the two *live* tshark defects into
   app-port detections. The gap is wiring, not design: the adapter is block-generic, the only wiring
   patch targets `wmem_block.c`, and `BLOCK_FAST` is the simpler allocator.
2. **Whether any of this enters the paper.** `tab:target-security`'s rows are programs and its
   columns configurations, and "Capstone spatial" is a *configuration* scoring 0 on 57 **temporal**
   bugs — so adding spatial defects means new `Cases` and a changed `\targetCorpus`.

**Deliberately not pursued**, so nobody mistakes it for an oversight: tshark's **251** class-`?`
candidates and memcached's **152 of 157** unread item-size commits; and `a809a784ec`, whose
containment is partial.

### 1b. The 29 class-A spatial candidates, and why the "plain" column was empty

Class A is a spatial defect whose access crosses the **`malloc` bound itself**. The triage found
**29** across the three programs — tshark 21, memcached 3, FFmpeg 5 — and for a long while **none**
was built. That was a scoping decision of mine, not an absence in the software, and the reasoning
was: `shrink` catches them and CHERI catches them, so they are a *tie* and add nothing to the
project's claim.

**That reasoning was wrong for what the numbers are for.** A row where every configuration catches
is the **baseline and the denominator**: it is the evidence that the harness and the arms work at
all, and it is the only place CheriBSD can register a spatial hit. Reporting "0 not-nested spatial"
beside "22 nested temporal" invites the reading that the not-nested class does not exist here, which
is the opposite of true — it was simply not pursued.

> **RETRACTED 2026-10-05, in full.** A table stood here headed *"Verified starting points,
> allocation site opened in the pinned source"*, listing four rows. **Every one of the four is
> false.** The sites were afterwards opened, one at a time, in the source each port actually pins:
>
> | row as published | what the pinned source says |
> |---|---|
> | memcached `ddee3e2` — `authfile.c:44` `calloc(1, sb.st_size)` | **already fixed at the pin.** `authfile.c:50` reads `calloc(1, sb.st_size + 2)`, with `auth_end = auth_data + sb.st_size + 1` (56) and an `auth_end - auth_cur` clamp on the `fgets` length (60) |
> | memcached `11b5f9b` — a `realloc`'d array at `proxy_lua.c:1403` | **not heap.** The overflowed object is `char temp[KEY_MAX_LENGTH + 1]` at `proxy_lua.c:717`, a stack array; the classifier matched a `realloc` elsewhere in the same file |
> | tshark `be813ede9d` — `extcap/etl.c:1411` `Message = g_malloc(Length)` | **absent at the pin.** 4.6.8's `extcap/etl.c` is 799 lines and contains no `Message = g_malloc` |
> | tshark `f207d25f4b`, `830cf562a0` — `g_strdup` error strings | **half right, and the wrong half is retracted below.** `830cf562a0` is indeed an integer-underflow subject (*"pcap: Fix an integer underflow."*) that filter 1 excludes by its own wording. The `g_strdup` reading of `f207d25f4b` is false: its subject is *"Don't let the reported length underflow w/ phdr"* and it touches no `g_strdup` |
>
> Nothing was measured wrong and nothing was built on these rows. What was wrong was publishing a
> classifier's output under the word *verified* — the same defect, one level up, as the "29" itself.

**The 29 is a candidate count. The verified count is 0.** Seven of the 29 have now been read
against the pinned source — every one that had been called verified, plus the three tshark rows
that survived a liveness pass. None is a live class-A defect:

| candidate | disposition in the pinned source |
|---|---|
| memcached `ddee3e2` | fix present (`authfile.c:50`, `+ 2`) |
| memcached `11b5f9b` | stack array (`proxy_lua.c:717`) |
| tshark `be813ede9d` | code absent at 4.6.8 |
| tshark `f207d25f4b` | **RETRACTED 2026-10-05:** misattributed, and returned to the unread pool. The row said "`g_strdup` of a string literal" at `wiretap/libpcap.c:621`; that line is the file's only `g_strdup`, but this commit has nothing to do with it. `git show --stat` gives the subject *"wiretap: pcap[ng]: Don't let the reported length underflow w/ phdr"* over `libpcap.c` and `pcapng.c`. Written from a grep in the file instead of the commit's diff — the same defect retracted above, committed again the same day |
| tshark `830cf562a0` | integer-underflow subject, excluded by filter 1 |
| tshark `06d08c5811` | **no access leaves the allocation.** `wsutil/eax.c:150` allocates `worksize`; the loop bound *is* `worksize`. The fix moves where a one-past-the-end address is *formed* — legal C, and nothing dereferences it |
| tshark `7ffc11e38f` | fix present (`wiretap/file_access.c:1322`, `:1376`) |
| tshark `3be1c99180` | fix present (`wiretap/netscreen.c:63-66`, `:311`, `:332`). Also moot: `ws_buffer_assure_space` over-allocates, so a read past `pkt_len` stays inside the allocation — class C, not A |
| **the remaining 22**, `f207d25f4b` among them | **not individually read.** Stated as unread, not as absent. 7 dispositioned + 22 unread = 29: the retracted row is listed for its trail and counted in the 22, not twice |

**Why 0 is a result here and not a gap.** Class A is the class upstream fixes *first* — it is what
fuzzers, ASan and compiler warnings find — and these three programs are pinned at recent releases
(memcached 1.6.45, wireshark 4.6.8). Three of the eight rows above are that mechanism made visible:
the fix is already in the tree we build. The class that survives into a current release is the one
no existing tool sees: a crossing inside a nested allocator's block (class B) or inside a single
allocation (class C). That is the project's own thesis, and it now has a measured denominator
instead of an assumption behind it.

> **RETRACTED 2026-10-06.** What stood here said the not-nested spatial baseline is synthetic
> **by necessity**, and that the zero is "a result, not a gap". **The measurement stands and is
> unchanged: 0 of the candidates read is live at the pin.** What is withdrawn is the inference from
> it to an empty cell, which rested on a premise never stated and never checked — *that a case must
> be live at the pin to be built*.
>
> It is not. **29 of the 35 existing corpus cases carry `live_in_pin: false`**, and the convention is
> explicit at `bug-corpora/memcached/allocator-repros/README.md:132-135`: each fix is an ancestor of
> the pin, "so the shipped allocator is exercised by a pre-fix consumer shape the commit's own diff
> shows -- the FFmpeg corpus's tier." Liveness is a **field recorded in the case**, not a gate on
> building one. Every temporal case in this inventory is itself a fix-reversal.
>
> So "class A is fixed upstream first" remains true *about liveness* and explains why the live count
> is 0. It does not explain an empty cell, and it was wrong to present the cell as closed. The cell
> is **open**, with one verified buildable candidate (below).
>
> The check that would have caught it is the one already written down: test the single sentence the
> conclusion rests on. Here that sentence was "a case must be live", which no document asserts and
> the corpus contradicts 27 times.

**Re-dispositioned against the correct criterion** — *a reconstructible heap defect whose crossing
leaves the allocation*, with liveness recorded rather than required:

| candidate | under the correct criterion |
|---|---|
| memcached `ddee3e2` | **BUILDABLE.** Its subject is "Fix minor severity heap buffer overflow reading `--auth-file`"; before the fix `auth_data = calloc(1, sb.st_size)` is scanned by an unclamped `fgets(auth_cur, MAX_ENTRY_LEN, ...)`, so the read leaves the allocation. A fix-reversal case exactly like the 27 |
| tshark `7ffc11e38f` | **RETRACTED 2026-10-06: hardening, not a reachable defect.** The row called it buildable because the fix adds `file_type_subtype < 0`, and I read the guard's existence as proof the hole was reachable. It is not. The fix guards three functions, and every caller of all three supplies a validated or construction-valid type: `mergecap.c:257` rejects a negative `-F` before `:392` uses it; `editcap.c:1018`, `:1054` and `tshark.c:3309` pass `wtap_dump_file_type_subtype(pdh)` from an open dump; `file.c:4470` is reached only through the short-circuit `save_format == cf->cd_t &&` at `:4468`, so the value equals the capture file's own type; and the Qt dialog's calls use the format list it built. That list is the complete set of callers in the fix's PARENT tree, not only the ones at our pin. **A bounds check added upstream is not evidence that the unbounded path was reachable** -- that needs a caller, and none was found |
| tshark `f207d25f4b` | no — read at last, from the diff: it replaces `orig_size -= phdr_len` and `packet_size -= phdr_len` with checked `ckd_sub`, so the defect is an **unsigned underflow** of a reported length, the same class as `830cf562a0` that filter 1 excludes by its own wording. Its downstream consequence may be spatial; the defect is not |
| tshark `3be1c99180` | no — `ws_buffer` over-allocates, so the crossing stays inside the allocation. Class C |
| tshark `be813ede9d` | no — a fix-reversal needs the code to exist at the pin, and `etw_dump_write_ldap_event` does not |
| tshark `06d08c5811` | no — unchanged, no access leaves the allocation |
| memcached `11b5f9b` | no — unchanged, a stack array |
| tshark `830cf562a0` | no — unchanged, an integer-underflow subject |

**This does not put tshark's 21 back in play.** Most still fail on capacity-versus-length or on the
code having to exist at the pin; what changed is the criterion, not the evidence.

**One of the three empty cells has been filled; the other two are still empty.** The memcached
row is now `plain-heap-repros/00`, built and measured. The tshark candidate did not survive its
reachability check and was retracted the same day, which is why this table names what each cell
has rather than what it might:

| cell | candidate | the crossing |
|---|---|---|
| memcached, not-nested spatial | **BUILT** — `plain-heap-repros/00` (`ddee3e2`) | an unclamped `fgets` scan leaves `calloc(1, sb.st_size)`; measured two-sided on both native arms, and ASan reports it |
| tshark, not-nested spatial | **none** | `7ffc11e38f` was retracted as hardening; the cell is still empty and its candidates are the 22 unread |
| FFmpeg, **nested** spatial | candidates only — see the section above | a frame-plane crossing, ownership and padding still to be opened |

They are **candidates until built and measured**, and neither is counted in any table yet.


The synthetic probes keep their role regardless: fx2/fx3 and memcached 20/21 show the arms
discriminate at `malloc` granularity, which is a different job from counting upstream defects.

**Not verified, and marked so:** FFmpeg's five were classified by a subagent, **three from the diff
alone without opening the allocation site** (`db05df9d13`, `bde5c6acb6`, `79e10e5196`). They are
candidates, not facts. They were not pursued: FFmpeg already contributes three not-nested spatial
cases (§1a cases 0-2), so a fourth candidate adds nothing the baseline lacks. The two that were
opened are `c79dfd29e6` (h264 `color_frame`) and `8e55f4f3e9` (v210dec `custom_stride`).

**Where a class-A case belongs, and why it is not this corpus.** The corpus harnesses narrow every
allocation their ported allocator makes — `wm_narrow()` on the wireshark side, the chunk carve on
memcached's — so they have **no malloc-granular arm to contrast against**. The app ports do:
fixtures fx2 `heap_neighbour` and fx3 `heap_one_past` read `level0` **RETURN** and `shrink`
**FAULT** in all three. So a class-A upstream defect belongs there as a **port fixture**, which is
also where the five not-nested *temporal* defects already live (memcached 17/18, tshark 14/15,
FFmpeg 24).


## 2. The SYNTHETIC spatial probes — and Sublet does catch most of them

> This section's title used to read *"Spatial exists only as synthetic probes"*. **That is no longer
> true:** §1a's eleven built cases are reductions of real upstream defects. What follows is about
> the port **fixtures**, which are synthetic and remain the only place the *nested-vs-malloc*
> contrast is visible — see §3.

Every spatial probe across the three ports' `app/host/safety-expect.txt`. Fixture numbers differ
per port, so each row names them explicitly.

| probe | `level0` | `shrink` / `sublet` | nested arm |
|---|---|---|---|
| `heap_neighbour`, `heap_one_past` — fx2/fx3 in all three ports | RETURN | **FAULT `oob`** | `chunks`, `slabsublet*` FAULT; FFmpeg pool arms **not registered** for fx2/fx3 |
| `global_oob`, `stack_oob` (controls) — fx7/fx8 in tshark and memcached, **fx8/fx9 in FFmpeg** | FAULT `oob` | FAULT `oob` | FAULT `oob` |
| `global_merged` — tshark fx9, FFmpeg fx10 (memcached has none) | RETURN | RETURN | RETURN (`chunks` too) |
| **tshark fx12 `wmem_neighbour`** | RETURN `c000ee` | **RETURN `c000ee`** | `chunks` **FAULT `oob`** |
| **memcached fx9 `slab_neighbour`** | RETURN `9000ee` | **RETURN `9000ee`** | `slabsublet0/1` **FAULT `oob`** |
| **FFmpeg fx16 `rs_underflow`** | RETURN | **RETURN** | `pool0`/`pool2`/`poolsublet` **FAULT `oob`**; `poolstock` RETURN |

So `sublet` **faults on plain spatial** (fx2/fx3, where `level0` returns). It returns on four spatial
probes: the **three nested-discriminating cells** — tshark fx12, memcached fx9, FFmpeg fx16, the only
three in these programs, 6 `FAULT oob` rows, all measured — **plus `global_merged`, which no arm
catches at all.**

**`global_merged` is a spatial miss that is not an allocator problem**, which is why it sits outside
the nested story. The compiler merges two 64-byte statics into one `.L_MergedGlobals`, so the bound
derived for either covers the pair; reading the second through the first is in-bounds. No allocator
is involved, nested or otherwise, and nothing in a lease mechanism addresses it — **CHERI has the
identical hole**, for the identical reason. It is recorded here so the row is not later miscounted as
a nested-spatial gap. (tshark's own catalogue notes it was added after FFmpeg fixture 8 showed a
global's bounds covering a merged group rather than the object alone.)

**FFmpeg fx14 `pool_one_past` is the instructive non-discriminator.** It faults on `poolstock` too,
because `AVBufferPool` hands out individually `malloc`'d buffers — malloc-granular bounds already
cover one-past-the-end. fx16 is FFmpeg's only genuine **sub-object** shape: the refstruct header and
its payload are one allocation, so a bound on the allocation cannot separate them.

## 3. Why Sublet returns on those three — read off the capability length

The tshark fx12 ladder, one measured `len` per arm, each from a committed bundle:

| arm | measured `len` at fx12 | outcome | bundle |
|---|---:|---|---|
| `level0` | 41 908 912 (~40 MiB, the whole arena) | RETURN `c000ee` | `ports/wireshark/app/results/2026-09-25-qemu-safety/` |
| `shrink` | 8 388 560 (8 MiB) | RETURN `c000ee` | `ports/wireshark/app/results/2026-09-25-qemu-safety/` |
| `sublet` | 1 048 528 (the 1 MiB wmem BLOCK) | RETURN `c000ee` | `ports/wireshark/app/results/2026-09-25-qemu-safety-sublet/` |
| **`chunks`** | **64** (the allocation itself) | **FAULT `oob`** | `ports/wireshark/app/results/20260929-qemu-tshark-step2/`, `.../2026-10-03-qemu-wmem-chunks-arm/` |

**Every layer narrows, and none of them reaches the object** until the nested allocator is ported. One
`g_malloc` hands wmem a 1 MiB region and every chunk carved from it inherits the block's bounds, so
on `sublet` the neighbour `q` lies *inside* `p`'s bound and the write is legal. On `chunks` the same
write faults — `sb`, insn `00c50023`, bounds ending `a5100070` against target `a5100080`.

The chunks bundle's own two-sided check on the adapter, from
`ports/wireshark/app/results/2026-10-03-qemu-wmem-chunks-arm/README.md`:

    chunks  fx12  p  cursor=a5100030  bounds=[a5100030,a5100070)  len-from-cursor=64
    chunks  fx10  p  cursor=a5100030  bounds=[a5100000,a5200000)  len-from-cursor=1048528

fx10 is BLOCK_**FAST**, which the chunk port deliberately leaves alone, and still shows the block
bounds. So the port narrows *selectively*: if it did nothing, fx12 would show block bounds too; if it
did too much, fx10 would not.

The other two cells, same mechanism:

| cell | arms that RETURN | arms that FAULT `oob` | bundles |
|---|---|---|---|
| memcached fx9 `slab_neighbour` | `level0`, `shrink`, `sublet` (`9000ee`, N=3) | `slabsublet0`, `slabsublet1` (N=6) | `ports/memcached/app/results/2026-10-01-qemu-safety/`, `.../2026-10-01-qemu-slab-sublet/` |
| FFmpeg fx16 `rs_underflow` | `level0` (`len=1572752`), `shrink`/`sublet` (`len=64`), `poolstock` (`len=64`) | `pool0`, `pool2` (cause 5), `poolsublet` | `ports/ffmpeg/app/results/2026-09-24-qemu-hardening/`, `ports/ffmpeg/sublet/results/2026-09-29-qemu/` |

### The spatial catches on the `sublet` arm are not Sublet's own mechanism

Sublet **is** revocation, and revocation has nothing to fire on while the object is alive. fx2/fx3
are caught by `shrink`'s per-object bounds, which `sublet` inherits by construction. The three
nested RETURNs are the nested allocator making **extents** invisible to the system allocator
(`malloc`), exactly as it makes **lifetimes** invisible. Same structure, one axis over: a nested
allocator makes both the size and the lifetime of its sub-objects invisible to the system
allocator, and porting it restores both.

## 4. Why there is no spatial advantage to claim over CHERI

Two independent reasons, which must not be conflated.

**(a) Spatial is a tie, by this repo's own committed verdict.**
`table6-cheri-vs-capstone-explained.md:158-162`:

> "**Tie.** *This is the intellectually honest part of the table:* for the spatial / null /
> uninitialised rows, base CHERI is already sufficient — both systems catch them synchronously.
> Capstone claims **no** advantage here. Its advantage is confined to the *temporal* class."

The reason is structural. Spatial safety needs only **bounds**, and the paper's own discussion says
CHERI has them — *"CHERI supports monotonic, unprivileged bounds derivation, so a \nestedalloc can
bound each suballocation … Temporal invalidation requires additional machinery"*
(`sections/06-discussion-and-related-work.tex`, under `\para{From spatial bounds to temporal
authority.}` — quote anchor rather than a line number, since Overleaf moves lines). So a CHERI
nested allocator could narrow to each suballocation exactly as our `chunks` arm does. **The
nested-spatial gap is a porting gap, not a hardware gap** — both systems close it by changing the
allocator.

Temporal is the opposite case: the stale address is legitimate and in-bounds, so no bound helps. It
needs revocation, and revocation *inside* a nested allocator needs the lease mechanism, which bounds
derivation cannot supply.

Do **not** restate this as "CHERI cannot bound sub-objects". The claim is about what the deployed
stack *derives*, not what is *derivable*.

**(b) We have no real spatial defects *yet* — because none were searched for.** The three
discriminating cells are synthetic probes written to measure the adapter, not reductions of upstream
bugs. See the retraction in §1: the earlier wording of this paragraph (*"and the hunt found nothing
to put in it"*) asserted a negative result from a search that excluded the class by construction.

**And reason (a) does not generalise to the nested case, which is where this matters.** The tie is
about spatial defects that cross the `malloc` bound — there bounds alone suffice and CHERI has them.
A defect whose overflow stays **inside** a nested allocator's block is a different cell: the system
allocator sees one block, so `shrink` and `sublet` return, and only a ported nested allocator faults.
That is measured today only by synthetic probes (tshark fx12, memcached fx9, FFmpeg fx16). Whether a
*real* upstream defect of that shape exists in these three programs is **open**, and it is the
question the hunt now under way is meant to answer. If one lands, the "spatial is a tie" framing
needs revisiting — which is the lead's call, not this document's.

## 5. Citing a bundle for one of these cells

The six nested `FAULT oob` rows are measured, but a bundle path only goes into a document after
grepping that bundle for the fixture number **in the format the bundles actually use**:

    fx9: AS PREDICTED: FAULT oob  (predicted FAULT oob; signal 11, len=None)

A column-shaped pattern such as `^slabsublet0 +9 ` matches nothing and returns a clean zero for five
of the six rows. That zero was caught only because the **positive control** — tshark `chunks` fx12,
known present from a README already read — returned zero as well, which indicts the regex rather
than the tree. The verified paths are in the tables above.

Two bundle names are *not* evidence for these cells, though they look adjacent:
`ports/memcached/allocators/results/20261004-qemu-corpus-defects/` and
`ports/ffmpeg/app/results/20261004-qemu-pool-corpus-40-47/` hold **corpus** cases (memcached fx12–16,
FFmpeg fixtures 40–47) and contain no safety-fixture rows at all.

One figure from a subagent report is **not** recorded here because no bundle carries it: the
"208-byte slab carve" for memcached fx9. Re-derive it from the slab port's own source before using it.
