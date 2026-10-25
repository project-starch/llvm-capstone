bounds model sha256 0b0377dc0d20436a; allocator alignment 8; positive control (65535 B, read at +65535): MISSED

| program | corpus | case | virtual-malloc | size | offset | width | slack below/above (worst) | silicon, child at the request size | silicon, child at the size rounded to the alignment |
|---|---|---|---|---:|---:|---:|---|---|---|
| memcached | plain-heap-repros | 00_ddee3e2_authfile_scan_past_calloc | CAUGHT | 9 | 9 | 1 | 0/0 | CAUGHT | MISSED (size 16) (CHANGES) |
| memcached | plain-heap-repros | 01_d5d9ff0_cachedump_end_marker_off_by_one | CAUGHT | 64 | 64 | 1 | 0/0 | CAUGHT | CAUGHT (size 64) |
| memcached | plain-heap-repros | 02_391f2e4762bf_freesuffix_realloc_sized_in_bytes | CAUGHT | 16 | 16 | 1 | 0/0 | CAUGHT | CAUGHT (size 16) |
| memcached | plain-heap-repros | 03_16a809e2a062_cache_create_freelist_sized_by_object | CAUGHT | 64 | 64 | 1 | 0/0 | CAUGHT | CAUGHT (size 64) |
| memcached | plain-heap-repros | 04_40aff8b0f113_stats_end_marker_past_exact_buffer | CAUGHT | 64 | 64 | 1 | 0/0 | CAUGHT | CAUGHT (size 64) |
| memcached | plain-heap-repros | 05_49f3b0ca9b57_out_string_crlf_copied_with_its_nul | CAUGHT | 16 | 16 | 1 | 0/0 | CAUGHT | CAUGHT (size 16) |
| memcached | plain-heap-repros | 06_212c3820c7bb_key_hash_filter_tag_length_underflow | CAUGHT | 16 | 16 | 1 | 0/0 | CAUGHT | CAUGHT (size 16) |
| memcached | plain-heap-repros | 07_0f605245cf3f_bin_delete_logs_unterminated_key | CAUGHT | 16 | 16 | 1 | 0/0 | CAUGHT | CAUGHT (size 16) |
| memcached | plain-heap-repros | 08_fa51ad8452d5_slab_list_shuffle_reads_one_past | CAUGHT | 32 | 32 | 1 | 0/0 | CAUGHT | CAUGHT (size 32) |
| memcached | plain-temporal-repros | 00_0d4901071c74_restart_line_freed_at_entry_but_not_cleared | CAUGHT | | | | | revoking: CAUGHT (cause 25); level0: MISSED | same |
| memcached | plain-temporal-repros | 01_e7793811f8c8_logger_write_to_watcher_freed_by_the_poll | CAUGHT | | | | | revoking: CAUGHT (cause 25); level0: MISSED | same |
| memcached | plain-temporal-repros | 02_3bc58f6ea55a_logger_loop_condition_still_reads_the_freed_watcher | CAUGHT | | | | | revoking: CAUGHT (cause 25); level0: MISSED | same |
| ffmpeg | plain-heap-repros | 00_d133b4a231_showcwt_kernel_scan_past_array | CAUGHT | 16 | 16 | 4 | 0/0 | CAUGHT | CAUGHT (size 16) |
| ffmpeg | plain-heap-repros | 01_bcbf3a5630_vf_scale_format_list_compaction | CAUGHT | 24 | 24 | 1 | 0/0 | CAUGHT | CAUGHT (size 24) |
| ffmpeg | plain-heap-repros | 02_56309e476a_vf_vif_mirror_below_base | CAUGHT | 16 | -4 | 4 | 0/0 | CAUGHT | CAUGHT (size 16) |
| ffmpeg | plain-heap-repros | 03_495b402f27_diracdec_edge_emu_buffer_undersized | CAUGHT | 512 | 512 | 1 | 0/0 | CAUGHT | CAUGHT (size 512) |
| ffmpeg | plain-heap-repros | 04_b3c7ebc1ed_swaprect_temp_sized_for_plane0 | CAUGHT | 17 | 17 | 1 | 0/0 | CAUGHT | MISSED (size 24) (CHANGES) |
| ffmpeg | plain-heap-repros | 05_8553e6ef57_cbs_av1_t35_payload_unpadded | CAUGHT | 24 | 27 | 1 | 0/0 | CAUGHT | CAUGHT (size 24) |
| ffmpeg | plain-heap-repros | 06_041d4f010e_prores_raw_header_len_unchecked | CAUGHT | 48 | 48 | 1 | 0/0 | CAUGHT | CAUGHT (size 48) |
| ffmpeg | plain-heap-repros | 07_8880a174d0_librist_read_ignores_caller_size | CAUGHT | 32 | 32 | 1 | 0/0 | CAUGHT | CAUGHT (size 32) |
| ffmpeg | plain-heap-repros | 08_b2df2f4f22_mpegenc_system_header_fixed_128 | CAUGHT | 160 | 160 | 1 | 0/0 | CAUGHT | CAUGHT (size 160) |
| ffmpeg | plain-heap-repros | 09_16b2049d4d_cfhd_transform2_wider_than_plane | CAUGHT | 64 | 64 | 1 | 0/0 | CAUGHT | CAUGHT (size 64) |
| ffmpeg | plain-heap-repros | 10_989444060d5f_lut3d_size2_computed_before_directive | CAUGHT | 64 | 64 | 1 | 0/0 | CAUGHT | CAUGHT (size 64) |
| ffmpeg | plain-heap-repros | 11_76645e096fab_exif_string_clone_drops_terminator | CAUGHT | 16 | 16 | 1 | 0/0 | CAUGHT | CAUGHT (size 16) |
| ffmpeg | plain-heap-repros | 12_ad956ff076ea_drawtext_bbox_buffer_ignores_separators | CAUGHT | 16 | 16 | 1 | 0/0 | CAUGHT | CAUGHT (size 16) |
| ffmpeg | plain-heap-repros | 14_789d7b1b1dff_ffv1dec_fltmap_index_unmasked | CAUGHT | 16 | 16 | 1 | 0/0 | CAUGHT | CAUGHT (size 16) |
| ffmpeg | plain-heap-repros | 15_e9e6fb879835_tdsc_raw_tile_read_by_geometry_not_size | CAUGHT | 48 | 48 | 1 | 0/0 | CAUGHT | CAUGHT (size 48) |
| ffmpeg | plain-heap-repros | 16_ca1c1f29ce47_img2enc_split_planes_read_by_geometry | CAUGHT | 32 | 32 | 1 | 0/0 | CAUGHT | CAUGHT (size 32) |
| ffmpeg | plain-heap-repros | 17_18761f9fb55c_rtpdec_av1_output_cursor_advanced_for_skipped_obu | CAUGHT | 16 | 16 | 1 | 0/0 | CAUGHT | CAUGHT (size 16) |
| ffmpeg | plain-heap-repros | 18_e8031e5b9ad2_showcwt_du_scroll_reads_row_past_plane | CAUGHT | 32 | 32 | 1 | 0/0 | CAUGHT | CAUGHT (size 32) |
| ffmpeg | plain-heap-repros | 19_905a4324030e_showcwt_position_initialised_to_sono_size | CAUGHT | 32 | 32 | 1 | 0/0 | CAUGHT | CAUGHT (size 32) |
| ffmpeg | plain-heap-repros | 20_bbdce45fda1e_diracdec_mctmp_rows_below_worst_case | CAUGHT | 16 | 16 | 1 | 0/0 | CAUGHT | CAUGHT (size 16) |
| ffmpeg | plain-heap-repros | 21_3fbb9560821b_afir_crossfade_reads_beyond_input_frame | CAUGHT | 16 | 16 | 4 | 0/0 | CAUGHT | CAUGHT (size 16) |
| ffmpeg | plain-heap-repros | 22_242ff799c75f_tdsc_cursor_advanced_by_a_stride_the_loop_already_added | CAUGHT | 512 | 512 | 1 | 0/0 | CAUGHT | CAUGHT (size 512) |
| ffmpeg | plain-heap-repros | 23_f45da79b2c33_exr_dwa_block_written_full_eight_by_eight | CAUGHT | 16 | 16 | 1 | 0/0 | CAUGHT | CAUGHT (size 16) |
| ffmpeg | plain-heap-repros | 24_162f75b5e679_exr_pxr24_uint_branch_strides_by_the_data_window | CAUGHT | 16 | 16 | 1 | 0/0 | CAUGHT | CAUGHT (size 16) |
| ffmpeg | plain-temporal-repros | 00_716d2a47c565_ops_dispatch_interior_alias_after_free | CAUGHT | | | | | revoking: CAUGHT (cause 25); level0: MISSED | same |
| ffmpeg | plain-temporal-repros | 01_c98810ab47fa_hw_base_encode_list_walk_reads_freed_link | CAUGHT | | | | | revoking: CAUGHT (cause 25); level0: MISSED | same |
| ffmpeg | plain-temporal-repros | 02_43de8b328b62_lzf_write_cursor_stale_after_realloc | CAUGHT | | | | | revoking: CAUGHT (cause 25); level0: MISSED | same |
| ffmpeg | plain-temporal-repros | 03_dc87758775e2_aviobuf_checksum_ptr_stale_after_realloc | CAUGHT | | | | | revoking: CAUGHT (cause 25); level0: MISSED | same |
| ffmpeg | plain-temporal-repros | 04_4b2248594c7f_jpeg2000_packed_headers_stream_stale_base | CAUGHT | | | | | revoking: CAUGHT (cause 25); level0: MISSED | same |
| ffmpeg | plain-temporal-repros | 05_d6458f6a8bf1_aacdec_tag_che_map_keeps_freed_element | CAUGHT | | | | | revoking: CAUGHT (cause 25); level0: MISSED | same |
| ffmpeg | plain-temporal-repros | 06_e8714f6f93d1_h264_delayed_pic_holds_interior_pointers | CAUGHT | | | | | revoking: CAUGHT (cause 25); level0: MISSED | same |
| ffmpeg | plain-temporal-repros | 07_a43e9cdd442b_isom_extradata_freed_before_callee_frees_it | CAUGHT | | | | | revoking: CAUGHT (cause 25); level0: MISSED | same |
| ffmpeg | plain-temporal-repros | 08_8a4ea9644833_diracdec_realloc_on_the_wrong_field | CAUGHT | | | | | revoking: CAUGHT (cause 25); level0: MISSED | same |
| ffmpeg | plain-temporal-repros | 09_265731f201f1_tx_subcontext_field_left_dangling_on_failure | CAUGHT | | | | | revoking: CAUGHT (cause 25); level0: MISSED | same |
| ffmpeg | plain-temporal-repros | 10_e7a65142b972_aacpsy_clears_the_local_not_the_field | CAUGHT | | | | | revoking: CAUGHT (cause 25); level0: MISSED | same |
| ffmpeg | plain-temporal-repros | 11_ba28222a14ab_ratecontrol_expr_field_not_cleared | CAUGHT | | | | | revoking: CAUGHT (cause 25); level0: MISSED | same |
| ffmpeg | plain-temporal-repros | 12_2e04d35c69e6_vlc_buf_freed_by_callee_then_read | CAUGHT | | | | | revoking: CAUGHT (cause 25); level0: MISSED | same |
| ffmpeg | subobject-repros | 00_8864fd0aec_cbs_h265_pic_timing_member | MISSED | | | | | MISSED | MISSED |
| ffmpeg | subobject-repros | 01_68845e26f7_vulkan_hevc_refpicset_member | MISSED | | | | | MISSED | MISSED |
| ffmpeg | subobject-repros | 02_e058af88ab_vulkan_hevc_dpb_member | MISSED | | | | | MISSED | MISSED |
| ffmpeg | subobject-repros | 03_89de2f0de1_aac_arith_last_member | MISSED | | | | | MISSED | MISSED |
| ffmpeg | subobject-repros | 04_1a00ea51cb_rtsp_control_url_underflow | MISSED | | | | | MISSED | MISSED |
| ffmpeg | subobject-repros | 05_d29ff88422_vulkan_av1_tile_sizes | MISSED | | | | | MISSED | MISSED |
| ffmpeg | subobject-repros | 06_a809a784ec_vvc_entry_point_start_ctu | MISSED | | | | | MISSED | MISSED |
| ffmpeg | subobject-repros | 07_275e217b10_hlsenc_key_uri_strlcpy_size | MISSED | | | | | MISSED | MISSED |
| ffmpeg | subobject-repros | 08_fb862976df_cbs_h266_col_width_val | MISSED | | | | | MISSED | MISSED |
| wireshark | plain-heap-repros | 00_19c51d27b9_netscaler_record_past_page | CAUGHT | 8192 | 8192 | 1 | 8/8 | DEPENDS (1/2 placements catch) (CHANGES) | DEPENDS (1/2 placements catch) (size 8192) (CHANGES) |
| wireshark | plain-heap-repros | 01_373504f7c9_dfvm_error_message_wrong_index | CAUGHT | 32 | 48 | 1 | 0/0 | CAUGHT | CAUGHT (size 32) |
| wireshark | plain-heap-repros | 02_381681583b_pcapng_nrb_custom_string_over_copy | CAUGHT | 16 | 16 | 1 | 0/0 | CAUGHT | CAUGHT (size 16) |
| wireshark | plain-heap-repros | 03_c556b648aa_strptime_reads_past_null_timezone | CAUGHT | 1 | 1 | 1 | 0/0 | CAUGHT | MISSED (size 8) (CHANGES) |
| wireshark | plain-heap-repros | 04_87803328179_blf_apptext_sized_without_terminator | CAUGHT | 16 | 16 | 1 | 0/0 | CAUGHT | CAUGHT (size 16) |
| wireshark | plain-heap-repros | 05_140aad08e081_nettrace_packet_buf_scanned_by_strstr | CAUGHT | 16 | 16 | 1 | 0/0 | CAUGHT | CAUGHT (size 16) |
| wireshark | plain-heap-repros | 06_3aad1ef236e6_t61_accent_branch_reads_the_following_byte | CAUGHT | 16 | 16 | 1 | 0/0 | CAUGHT | CAUGHT (size 16) |
| wireshark | plain-heap-repros | 07_e2ca71beaed2_uat_unbinstring_leaves_no_terminator | CAUGHT | 16 | 16 | 1 | 0/0 | CAUGHT | CAUGHT (size 16) |
| wireshark | plain-heap-repros | 08_0cae98570ebc_cl_find_custom_memcmp_reads_the_prefix_length | CAUGHT | 16 | 16 | 1 | 0/0 | CAUGHT | CAUGHT (size 16) |
| wireshark | plain-heap-repros | 09_bf123efe154d_uat_oid_empty_field_underflows_the_index | CAUGHT | 16 | 16 | 1 | 0/0 | CAUGHT | CAUGHT (size 16) |
| wireshark | plain-heap-repros | 10_e9b933473e8f_address_to_str_buf_clamps_to_size_not_last_index | CAUGHT | 16 | 16 | 1 | 0/0 | CAUGHT | CAUGHT (size 16) |
| wireshark | plain-heap-repros | 11_4b15bf76a7f7_oid_to_str_buf_tail_reservation_one_short | CAUGHT | 16 | 16 | 1 | 0/0 | CAUGHT | CAUGHT (size 16) |
| wireshark | plain-temporal-repros | 00_f3c2e6087e7b_k12_callee_frees_its_own_argument | CAUGHT | | | | | revoking: CAUGHT (cause 25); level0: MISSED | same |
| wireshark | plain-temporal-repros | 01_7dcf69480de8_peak_trc_callee_frees_state_struct | CAUGHT | | | | | revoking: CAUGHT (cause 25); level0: MISSED | same |
| wireshark | plain-temporal-repros | 02_0fc7f3781351_wspstat_container_freed_before_its_contents | CAUGHT | | | | | revoking: CAUGHT (cause 25); level0: MISSED | same |
| wireshark | plain-temporal-repros | 03_012a179785ab_filesystem_alias_named_copy_is_not_a_copy | CAUGHT | | | | | revoking: CAUGHT (cause 25); level0: MISSED | same |
| wireshark | plain-temporal-repros | 04_07ffcf90426b_extcap_one_help_string_stored_in_many_owners | CAUGHT | | | | | revoking: CAUGHT (cause 25); level0: MISSED | same |
| wireshark | plain-temporal-repros | 05_fb46cda19602_wtap_close_inner_loop_rewinds_outer_cursor | CAUGHT | | | | | revoking: CAUGHT (cause 25); level0: MISSED | same |
| wireshark | plain-temporal-repros | 06_d3e3c00fbbe2_prefs_static_filter_label_not_cleared | CAUGHT | | | | | revoking: CAUGHT (cause 25); level0: MISSED | same |
| wireshark | plain-temporal-repros | 07_8dc7d164dcdb_prefs_reset_not_idempotent | CAUGHT | | | | | revoking: CAUGHT (cause 25); level0: MISSED | same |
| wireshark | plain-temporal-repros | 08_48a00fd55671_ftype_string_freed_twice_by_caller_and_callee | CAUGHT | | | | | revoking: CAUGHT (cause 25); level0: MISSED | same |
| wireshark | plain-temporal-repros | 09_cfc15838bdec_capture_ifinfo_out_parameter_left_unassigned | CAUGHT | | | | | revoking: CAUGHT (cause 25); level0: MISSED | same |
