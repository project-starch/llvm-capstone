# ffmpeg/plain-heap-repros, arm `poisoncap-spatial` -- 2026-10-09

Platform: PoisonCap mode 0: the rebuilt published PoisonCap platform (CHERI-PURECAP-QEMU-POISON kernel, libc with the mrs poison-retire fix), runtime revocation OFF (booted on the second host).
Predictions committed before the run: the corpus's committed predictions and 2852edcdd747 (size-class slack). Runtime revocation reported by the ABI control: 0.

Result LINES only; the boot capture is contaminated by construction.

## Controls, from the same boot

    cheribsd-abi: CHERI_ABI pointer_bytes=16 runtime_revocation=0
    cheribsd-bounds: CHERI_BOUNDARY_READY

## Cases: {'CAUGHT': 22, 'NOT CAUGHT': 3}

    00_d133b4a231_showcwt_kernel_scan_past_array
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x1020ca INSIDE ffh_read_probe [0x1020ba, 0x1020d6)
        SUPERVISE fault signal=34 code=1 addr=0x1020ca pc=0x1020ca
    01_bcbf3a5630_vf_scale_format_list_compaction
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED the shift loop's guard was j < nb_formats while its body read formats[j + 1], so it read one element past av_malloc_array -- the value
    02_56309e476a_vf_vif_mirror_below_base
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x102284 INSIDE ffh_read_probe [0x102274, 0x102290)
        SUPERVISE fault signal=34 code=1 addr=0x102284 pc=0x102284
    03_495b402f27_diracdec_edge_emu_buffer_undersized
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x102208 INSIDE ffh_write_probe_u8 [0x1021f0, 0x102214)
        SUPERVISE fault signal=34 code=1 addr=0x102208 pc=0x102208
    04_b3c7ebc1ed_swaprect_temp_sized_for_plane0
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED the scratch row was sized w * pixsteps[0], the luma plane, while the chroma row of a semi-planar 4:2:0 format at odd width needs one b
    05_8553e6ef57_cbs_av1_t35_payload_unpadded
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED the T.35 payload was allocated at exactly payload_size, so a reader entitled to FFmpeg's 64-byte input padding read past av_buffer_all
    06_041d4f010e_prores_raw_header_len_unchecked
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x1021c0 INSIDE ffh_read_probe_u8 [0x1021b0, 0x1021cc)
        SUPERVISE fault signal=34 code=1 addr=0x1021c0 pc=0x1021c0
    07_8880a174d0_librist_read_ignores_caller_size
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x1021ee INSIDE ffh_write_probe_u8 [0x1021d6, 0x1021fa)
        SUPERVISE fault signal=34 code=1 addr=0x1021ee pc=0x1021ee
    08_b2df2f4f22_mpegenc_system_header_fixed_128
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x10219a INSIDE ffh_write_probe_u8 [0x102182, 0x1021a6)
        SUPERVISE fault signal=34 code=1 addr=0x10219a pc=0x10219a
    09_16b2049d4d_cfhd_transform2_wider_than_plane
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x102176 INSIDE ffh_write_probe_u8 [0x10215e, 0x102182)
        SUPERVISE fault signal=34 code=1 addr=0x102176 pc=0x102176
    10_989444060d5f_lut3d_size2_computed_before_directive
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x102136 INSIDE ffh_write_probe_u8 [0x10211e, 0x102142)
        SUPERVISE fault signal=34 code=1 addr=0x102136 pc=0x102136
    11_76645e096fab_exif_string_clone_drops_terminator
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x102220 INSIDE ffh_read_probe_u8 [0x102210, 0x10222c)
        SUPERVISE fault signal=34 code=1 addr=0x102220 pc=0x102220
    12_ad956ff076ea_drawtext_bbox_buffer_ignores_separators
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x10216a INSIDE ffh_write_probe_u8 [0x102152, 0x102176)
        SUPERVISE fault signal=34 code=1 addr=0x10216a pc=0x10216a
    13_8af6c71d96f4_vif_single_reflection_indexes_below_base
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x10219c INSIDE ffh_read_probe [0x10218c, 0x1021a8)
        SUPERVISE fault signal=34 code=1 addr=0x10219c pc=0x10219c
    14_789d7b1b1dff_ffv1dec_fltmap_index_unmasked
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x10212c INSIDE ffh_read_probe_u8 [0x10211c, 0x102138)
        SUPERVISE fault signal=34 code=1 addr=0x10212c pc=0x10212c
    15_e9e6fb879835_tdsc_raw_tile_read_by_geometry_not_size
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x102092 INSIDE ffh_read_probe_u8 [0x102082, 0x10209e)
        SUPERVISE fault signal=34 code=1 addr=0x102092 pc=0x102092
    16_ca1c1f29ce47_img2enc_split_planes_read_by_geometry
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x10228c INSIDE ffh_read_probe_u8 [0x10227c, 0x102298)
        SUPERVISE fault signal=34 code=1 addr=0x10228c pc=0x10228c
    17_18761f9fb55c_rtpdec_av1_output_cursor_advanced_for_skipped_obu
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x1020d4 INSIDE ffh_write_probe_u8 [0x1020bc, 0x1020e0)
        SUPERVISE fault signal=34 code=1 addr=0x1020d4 pc=0x1020d4
    18_e8031e5b9ad2_showcwt_du_scroll_reads_row_past_plane
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x102150 INSIDE ffh_read_probe_u8 [0x102140, 0x10215c)
        SUPERVISE fault signal=34 code=1 addr=0x102150 pc=0x102150
    19_905a4324030e_showcwt_position_initialised_to_sono_size
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x10209c INSIDE ffh_write_probe_u8 [0x102084, 0x1020a8)
        SUPERVISE fault signal=34 code=1 addr=0x10209c pc=0x10209c
    20_bbdce45fda1e_diracdec_mctmp_rows_below_worst_case
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x102178 INSIDE ffh_write_probe_u8 [0x102160, 0x102184)
        SUPERVISE fault signal=34 code=1 addr=0x102178 pc=0x102178
    21_3fbb9560821b_afir_crossfade_reads_beyond_input_frame
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x10211a INSIDE ffh_read_probe [0x10210a, 0x102126)
        SUPERVISE fault signal=34 code=1 addr=0x10211a pc=0x10211a
    22_242ff799c75f_tdsc_cursor_advanced_by_a_stride_the_loop_already_added
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x102140 INSIDE ffh_write_probe_u8 [0x102128, 0x10214c)
        SUPERVISE fault signal=34 code=1 addr=0x102140 pc=0x102140
    23_f45da79b2c33_exr_dwa_block_written_full_eight_by_eight
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x1021a8 INSIDE ffh_write_probe_u8 [0x102190, 0x1021b4)
        SUPERVISE fault signal=34 code=1 addr=0x1021a8 pc=0x1021a8
    24_162f75b5e679_exr_pxr24_uint_branch_strides_by_the_data_window
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x102150 INSIDE ffh_read_probe_u8 [0x102140, 0x10215c)
        SUPERVISE fault signal=34 code=1 addr=0x102150 pc=0x102150
