# The plain corpora on silicon: results (2026-10-16, caplifive_sublet_d93a8d66b)

Pre-registered in README.md (lane fa3806536fa7; level0 amended 0f1c5d7f7463, before its boot). Both boots ran the
exact-regions supervised-call monitor (wrapper 1269c77, monitor ebe0a25; landed on capstone-bootstrap) and the 127-slot
process-ABI module:
- silicon-lifetimes: firmware 0409fc297ed52e26, 04:29-04:49;
- silicon-level0: firmware f1cfbeece7760d6b, 04:51-05:09.

| arm | spatial (46) | temporal (26) | controls (clean / oob / uaf / subobj) |
|---|---|---|---|
| silicon-lifetimes | 46 CAUGHT at the labelled probe, cause 28 | **26 CAUGHT** at the labelled probe, cause 25 | RETURNED / CAUGHT 28 / CAUGHT 25 / RETURNED |
| silicon-level0 | 46 CAUGHT at the labelled probe, cause 28 | **26 MISSED** (DEFECT-REPRODUCED) | RETURNED / CAUGHT 28 / RETURNED / RETURNED |

- **Pairing:** every image is the one the QEMU stage ran (by sha256), and every fixed run printed VERDICT FIXED and
  exited 0.
- **Every run ended by itself:** 148 of 148 in each boot. A domain fault ends only its own capstone-exec; the next run
  continues.
- **Every verdict and every probe is as pre-registered,** 72 of 72 in each arm.
  - The temporal cause 25 was predicted. QEMU reads 24 there (Q-11).
  - The spatial cause 28 is the RTL's OUT_OF_BOUNDS where QEMU reports 5 / 7. The README left it open; the lifetimes
    arm's per-row cause field had carried QEMU's number.

**What it shows:**
- On the Sublet silicon, per-object bounds catch every spatial defect in these corpora.
- Per-object lifetimes (a CDERIVE child per object, CREVOKE on free) catch every temporal one at the stale access
  itself. Without the lifetimes, the same images miss all 26.
- The two arms differ only in level0's CAPSTONE_LEVEL0_LIFETIMES.

**Scope:**
- Plain corpora only. Sub-object (9) and nested (50) are not in these rows.
- The lifetimes arm is level0 plus CDERIVE children, not the virtual profile's mallocng.
- Every object in these cases is exact at its placement. An inexact request would keep plain bounds and no lifetime.

## Per case

| case | kind | silicon-lifetimes | silicon-level0 |
|---|---|---|---|
| memcached 00_ddee3e2_authfile_scan_past_calloc | spatial | CAUGHT 28 | CAUGHT 28 |
| memcached 01_d5d9ff0_cachedump_end_marker_off_by_one | spatial | CAUGHT 28 | CAUGHT 28 |
| memcached 02_391f2e4762bf_freesuffix_realloc_sized_in_bytes | spatial | CAUGHT 28 | CAUGHT 28 |
| memcached 03_16a809e2a062_cache_create_freelist_sized_by_object | spatial | CAUGHT 28 | CAUGHT 28 |
| memcached 04_40aff8b0f113_stats_end_marker_past_exact_buffer | spatial | CAUGHT 28 | CAUGHT 28 |
| memcached 05_49f3b0ca9b57_out_string_crlf_copied_with_its_nul | spatial | CAUGHT 28 | CAUGHT 28 |
| memcached 06_212c3820c7bb_key_hash_filter_tag_length_underflow | spatial | CAUGHT 28 | CAUGHT 28 |
| memcached 07_0f605245cf3f_bin_delete_logs_unterminated_key | spatial | CAUGHT 28 | CAUGHT 28 |
| memcached 08_fa51ad8452d5_slab_list_shuffle_reads_one_past | spatial | CAUGHT 28 | CAUGHT 28 |
| memcached 00_0d4901071c74_restart_line_freed_at_entry_but_not_cleared | temporal | CAUGHT 25 | MISSED |
| memcached 01_e7793811f8c8_logger_write_to_watcher_freed_by_the_poll | temporal | CAUGHT 25 | MISSED |
| memcached 02_3bc58f6ea55a_logger_loop_condition_still_reads_the_freed_watcher | temporal | CAUGHT 25 | MISSED |
| ffmpeg 00_d133b4a231_showcwt_kernel_scan_past_array | spatial | CAUGHT 28 | CAUGHT 28 |
| ffmpeg 01_bcbf3a5630_vf_scale_format_list_compaction | spatial | CAUGHT 28 | CAUGHT 28 |
| ffmpeg 02_56309e476a_vf_vif_mirror_below_base | spatial | CAUGHT 28 | CAUGHT 28 |
| ffmpeg 03_495b402f27_diracdec_edge_emu_buffer_undersized | spatial | CAUGHT 28 | CAUGHT 28 |
| ffmpeg 04_b3c7ebc1ed_swaprect_temp_sized_for_plane0 | spatial | CAUGHT 28 | CAUGHT 28 |
| ffmpeg 05_8553e6ef57_cbs_av1_t35_payload_unpadded | spatial | CAUGHT 28 | CAUGHT 28 |
| ffmpeg 06_041d4f010e_prores_raw_header_len_unchecked | spatial | CAUGHT 28 | CAUGHT 28 |
| ffmpeg 07_8880a174d0_librist_read_ignores_caller_size | spatial | CAUGHT 28 | CAUGHT 28 |
| ffmpeg 08_b2df2f4f22_mpegenc_system_header_fixed_128 | spatial | CAUGHT 28 | CAUGHT 28 |
| ffmpeg 09_16b2049d4d_cfhd_transform2_wider_than_plane | spatial | CAUGHT 28 | CAUGHT 28 |
| ffmpeg 10_989444060d5f_lut3d_size2_computed_before_directive | spatial | CAUGHT 28 | CAUGHT 28 |
| ffmpeg 11_76645e096fab_exif_string_clone_drops_terminator | spatial | CAUGHT 28 | CAUGHT 28 |
| ffmpeg 12_ad956ff076ea_drawtext_bbox_buffer_ignores_separators | spatial | CAUGHT 28 | CAUGHT 28 |
| ffmpeg 13_8af6c71d96f4_vif_single_reflection_indexes_below_base | spatial | CAUGHT 28 | CAUGHT 28 |
| ffmpeg 14_789d7b1b1dff_ffv1dec_fltmap_index_unmasked | spatial | CAUGHT 28 | CAUGHT 28 |
| ffmpeg 15_e9e6fb879835_tdsc_raw_tile_read_by_geometry_not_size | spatial | CAUGHT 28 | CAUGHT 28 |
| ffmpeg 16_ca1c1f29ce47_img2enc_split_planes_read_by_geometry | spatial | CAUGHT 28 | CAUGHT 28 |
| ffmpeg 17_18761f9fb55c_rtpdec_av1_output_cursor_advanced_for_skipped_obu | spatial | CAUGHT 28 | CAUGHT 28 |
| ffmpeg 18_e8031e5b9ad2_showcwt_du_scroll_reads_row_past_plane | spatial | CAUGHT 28 | CAUGHT 28 |
| ffmpeg 19_905a4324030e_showcwt_position_initialised_to_sono_size | spatial | CAUGHT 28 | CAUGHT 28 |
| ffmpeg 20_bbdce45fda1e_diracdec_mctmp_rows_below_worst_case | spatial | CAUGHT 28 | CAUGHT 28 |
| ffmpeg 21_3fbb9560821b_afir_crossfade_reads_beyond_input_frame | spatial | CAUGHT 28 | CAUGHT 28 |
| ffmpeg 22_242ff799c75f_tdsc_cursor_advanced_by_a_stride_the_loop_already_added | spatial | CAUGHT 28 | CAUGHT 28 |
| ffmpeg 23_f45da79b2c33_exr_dwa_block_written_full_eight_by_eight | spatial | CAUGHT 28 | CAUGHT 28 |
| ffmpeg 24_162f75b5e679_exr_pxr24_uint_branch_strides_by_the_data_window | spatial | CAUGHT 28 | CAUGHT 28 |
| ffmpeg 00_716d2a47c565_ops_dispatch_interior_alias_after_free | temporal | CAUGHT 25 | MISSED |
| ffmpeg 01_c98810ab47fa_hw_base_encode_list_walk_reads_freed_link | temporal | CAUGHT 25 | MISSED |
| ffmpeg 02_43de8b328b62_lzf_write_cursor_stale_after_realloc | temporal | CAUGHT 25 | MISSED |
| ffmpeg 03_dc87758775e2_aviobuf_checksum_ptr_stale_after_realloc | temporal | CAUGHT 25 | MISSED |
| ffmpeg 04_4b2248594c7f_jpeg2000_packed_headers_stream_stale_base | temporal | CAUGHT 25 | MISSED |
| ffmpeg 05_d6458f6a8bf1_aacdec_tag_che_map_keeps_freed_element | temporal | CAUGHT 25 | MISSED |
| ffmpeg 06_e8714f6f93d1_h264_delayed_pic_holds_interior_pointers | temporal | CAUGHT 25 | MISSED |
| ffmpeg 07_a43e9cdd442b_isom_extradata_freed_before_callee_frees_it | temporal | CAUGHT 25 | MISSED |
| ffmpeg 08_8a4ea9644833_diracdec_realloc_on_the_wrong_field | temporal | CAUGHT 25 | MISSED |
| ffmpeg 09_265731f201f1_tx_subcontext_field_left_dangling_on_failure | temporal | CAUGHT 25 | MISSED |
| ffmpeg 10_e7a65142b972_aacpsy_clears_the_local_not_the_field | temporal | CAUGHT 25 | MISSED |
| ffmpeg 11_ba28222a14ab_ratecontrol_expr_field_not_cleared | temporal | CAUGHT 25 | MISSED |
| ffmpeg 12_2e04d35c69e6_vlc_buf_freed_by_callee_then_read | temporal | CAUGHT 25 | MISSED |
| wireshark 00_19c51d27b9_netscaler_record_past_page | spatial | CAUGHT 28 | CAUGHT 28 |
| wireshark 01_373504f7c9_dfvm_error_message_wrong_index | spatial | CAUGHT 28 | CAUGHT 28 |
| wireshark 02_381681583b_pcapng_nrb_custom_string_over_copy | spatial | CAUGHT 28 | CAUGHT 28 |
| wireshark 03_c556b648aa_strptime_reads_past_null_timezone | spatial | CAUGHT 28 | CAUGHT 28 |
| wireshark 04_87803328179_blf_apptext_sized_without_terminator | spatial | CAUGHT 28 | CAUGHT 28 |
| wireshark 05_140aad08e081_nettrace_packet_buf_scanned_by_strstr | spatial | CAUGHT 28 | CAUGHT 28 |
| wireshark 06_3aad1ef236e6_t61_accent_branch_reads_the_following_byte | spatial | CAUGHT 28 | CAUGHT 28 |
| wireshark 07_e2ca71beaed2_uat_unbinstring_leaves_no_terminator | spatial | CAUGHT 28 | CAUGHT 28 |
| wireshark 08_0cae98570ebc_cl_find_custom_memcmp_reads_the_prefix_length | spatial | CAUGHT 28 | CAUGHT 28 |
| wireshark 09_bf123efe154d_uat_oid_empty_field_underflows_the_index | spatial | CAUGHT 28 | CAUGHT 28 |
| wireshark 10_e9b933473e8f_address_to_str_buf_clamps_to_size_not_last_index | spatial | CAUGHT 28 | CAUGHT 28 |
| wireshark 11_4b15bf76a7f7_oid_to_str_buf_tail_reservation_one_short | spatial | CAUGHT 28 | CAUGHT 28 |
| wireshark 00_f3c2e6087e7b_k12_callee_frees_its_own_argument | temporal | CAUGHT 25 | MISSED |
| wireshark 01_7dcf69480de8_peak_trc_callee_frees_state_struct | temporal | CAUGHT 25 | MISSED |
| wireshark 02_0fc7f3781351_wspstat_container_freed_before_its_contents | temporal | CAUGHT 25 | MISSED |
| wireshark 03_012a179785ab_filesystem_alias_named_copy_is_not_a_copy | temporal | CAUGHT 25 | MISSED |
| wireshark 04_07ffcf90426b_extcap_one_help_string_stored_in_many_owners | temporal | CAUGHT 25 | MISSED |
| wireshark 05_fb46cda19602_wtap_close_inner_loop_rewinds_outer_cursor | temporal | CAUGHT 25 | MISSED |
| wireshark 06_d3e3c00fbbe2_prefs_static_filter_label_not_cleared | temporal | CAUGHT 25 | MISSED |
| wireshark 07_8dc7d164dcdb_prefs_reset_not_idempotent | temporal | CAUGHT 25 | MISSED |
| wireshark 08_48a00fd55671_ftype_string_freed_twice_by_caller_and_callee | temporal | CAUGHT 25 | MISSED |
| wireshark 09_cfc15838bdec_capture_ifinfo_out_parameter_left_unassigned | temporal | CAUGHT 25 | MISSED |
