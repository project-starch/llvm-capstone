# ffmpeg/plain-heap-repros, arm `cheribsd-subobject` -- 2026-10-09

Platform: stock CheriBSD purecap, kernel CHERI-PURECAP-QEMU, revocation on, every case built with -Xclang -cheri-bounds=subobject-safe (booted on the second host).
Predictions committed before the run: 886abf4cc336. Runtime revocation reported by the ABI control: 1.

Result LINES only; the boot capture is contaminated by construction.

## Controls, from the same boot

    cheribsd-abi: CHERI_ABI pointer_bytes=16 runtime_revocation=1
    cheribsd-bounds: CHERI_BOUNDARY_READY
    subobj-control: exit 162 (SIGPROT before any output; run.py lists the row as FAIL because its expectation needs a stdout line, the runner's own check reads the exit status -- the control was changed the same day to print first)

## Cases: {'CAUGHT': 22, 'NOT CAUGHT': 3}

    00_d133b4a231_showcwt_kernel_scan_past_array
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x102156 INSIDE ffh_read_probe [0x102146, 0x102162)
        SUPERVISE fault signal=34 code=1 addr=0x102156 pc=0x102156
    01_bcbf3a5630_vf_scale_format_list_compaction
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED the shift loop's guard was j < nb_formats while its body read formats[j + 1], so it read one element past av_malloc_array -- the value
    02_56309e476a_vf_vif_mirror_below_base
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x102310 INSIDE ffh_read_probe [0x102300, 0x10231c)
        SUPERVISE fault signal=34 code=1 addr=0x102310 pc=0x102310
    03_495b402f27_diracdec_edge_emu_buffer_undersized
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x102294 INSIDE ffh_write_probe_u8 [0x10227c, 0x1022a0)
        SUPERVISE fault signal=34 code=1 addr=0x102294 pc=0x102294
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
        buggy: exit 162 -> CAUGHT; fault pc 0x10224c INSIDE ffh_read_probe_u8 [0x10223c, 0x102258)
        SUPERVISE fault signal=34 code=1 addr=0x10224c pc=0x10224c
    07_8880a174d0_librist_read_ignores_caller_size
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x10227a INSIDE ffh_write_probe_u8 [0x102262, 0x102286)
        SUPERVISE fault signal=34 code=1 addr=0x10227a pc=0x10227a
    08_b2df2f4f22_mpegenc_system_header_fixed_128
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x102226 INSIDE ffh_write_probe_u8 [0x10220e, 0x102232)
        SUPERVISE fault signal=34 code=1 addr=0x102226 pc=0x102226
    09_16b2049d4d_cfhd_transform2_wider_than_plane
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x102202 INSIDE ffh_write_probe_u8 [0x1021ea, 0x10220e)
        SUPERVISE fault signal=34 code=1 addr=0x102202 pc=0x102202
    10_989444060d5f_lut3d_size2_computed_before_directive
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x1021c2 INSIDE ffh_write_probe_u8 [0x1021aa, 0x1021ce)
        SUPERVISE fault signal=34 code=1 addr=0x1021c2 pc=0x1021c2
    11_76645e096fab_exif_string_clone_drops_terminator
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x1022ac INSIDE ffh_read_probe_u8 [0x10229c, 0x1022b8)
        SUPERVISE fault signal=34 code=1 addr=0x1022ac pc=0x1022ac
    12_ad956ff076ea_drawtext_bbox_buffer_ignores_separators
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x1021f6 INSIDE ffh_write_probe_u8 [0x1021de, 0x102202)
        SUPERVISE fault signal=34 code=1 addr=0x1021f6 pc=0x1021f6
    13_8af6c71d96f4_vif_single_reflection_indexes_below_base
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x102228 INSIDE ffh_read_probe [0x102218, 0x102234)
        SUPERVISE fault signal=34 code=1 addr=0x102228 pc=0x102228
    14_789d7b1b1dff_ffv1dec_fltmap_index_unmasked
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x1021b8 INSIDE ffh_read_probe_u8 [0x1021a8, 0x1021c4)
        SUPERVISE fault signal=34 code=1 addr=0x1021b8 pc=0x1021b8
    15_e9e6fb879835_tdsc_raw_tile_read_by_geometry_not_size
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x10211e INSIDE ffh_read_probe_u8 [0x10210e, 0x10212a)
        SUPERVISE fault signal=34 code=1 addr=0x10211e pc=0x10211e
    16_ca1c1f29ce47_img2enc_split_planes_read_by_geometry
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x102320 INSIDE ffh_read_probe_u8 [0x102310, 0x10232c)
        SUPERVISE fault signal=34 code=1 addr=0x102320 pc=0x102320
    17_18761f9fb55c_rtpdec_av1_output_cursor_advanced_for_skipped_obu
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x102160 INSIDE ffh_write_probe_u8 [0x102148, 0x10216c)
        SUPERVISE fault signal=34 code=1 addr=0x102160 pc=0x102160
    18_e8031e5b9ad2_showcwt_du_scroll_reads_row_past_plane
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x1021dc INSIDE ffh_read_probe_u8 [0x1021cc, 0x1021e8)
        SUPERVISE fault signal=34 code=1 addr=0x1021dc pc=0x1021dc
    19_905a4324030e_showcwt_position_initialised_to_sono_size
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x102128 INSIDE ffh_write_probe_u8 [0x102110, 0x102134)
        SUPERVISE fault signal=34 code=1 addr=0x102128 pc=0x102128
    20_bbdce45fda1e_diracdec_mctmp_rows_below_worst_case
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x102204 INSIDE ffh_write_probe_u8 [0x1021ec, 0x102210)
        SUPERVISE fault signal=34 code=1 addr=0x102204 pc=0x102204
    21_3fbb9560821b_afir_crossfade_reads_beyond_input_frame
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x1021a6 INSIDE ffh_read_probe [0x102196, 0x1021b2)
        SUPERVISE fault signal=34 code=1 addr=0x1021a6 pc=0x1021a6
    22_242ff799c75f_tdsc_cursor_advanced_by_a_stride_the_loop_already_added
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x1021cc INSIDE ffh_write_probe_u8 [0x1021b4, 0x1021d8)
        SUPERVISE fault signal=34 code=1 addr=0x1021cc pc=0x1021cc
    23_f45da79b2c33_exr_dwa_block_written_full_eight_by_eight
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x102234 INSIDE ffh_write_probe_u8 [0x10221c, 0x102240)
        SUPERVISE fault signal=34 code=1 addr=0x102234 pc=0x102234
    24_162f75b5e679_exr_pxr24_uint_branch_strides_by_the_data_window
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x1021dc INSIDE ffh_read_probe_u8 [0x1021cc, 0x1021e8)
        SUPERVISE fault signal=34 code=1 addr=0x1021dc pc=0x1021dc
