# ffmpeg/plain-heap-repros on the Capstone arm `capstone-subobject` -- 2026-10-09

Predictions committed before the run: 49498f856e87.
Platform: Capstone application VM on capstone-qemu 32e7c9754f9a, kernel e58613598c89, firmware 6f2b082cb677, capstone-exec 0752aa7c49c9, module a88ed2159b43; SDK heap=level0 heap_log=22 runtime 3c72f36acd5b; compiler 7d01722aab88.

Reproduce (one VM up, see ports/common/application/README.md):

    python3 capstone/bug-corpora/tools/run-capstone-domain.py \
      --corpus capstone/bug-corpora/ffmpeg/plain-heap-repros --arm capstone-subobject --sdk <SDK, heap=level0> \
      --state <capstone-vm state> --out <fresh dir> \
      --cc-arg=-Xclang \
      --cc-arg=-fcapstone-subobject-bounds

## Controls, from the same boot

    clean  RETURNED  required RETURNED  CONTROL clean RETURNED 1
    oob    CAUGHT    required CAUGHT    cause=7 pc=0xc670e5b8 address=0xc6751958 in ctl_write_probe+0x58 (the labelled probe)
    uaf    RETURNED  required RETURNED  CONTROL uaf RETURNED observed=0xaa reissued=1
    subobj CAUGHT    required CAUGHT    cause=7 pc=0xc670e5b8 address=0xc6751948 in ctl_write_probe+0x58 (the labelled probe)

## Cases: {'CAUGHT': 25}

    case  outcome  as-predicted  fixed  detail  image(sha256/16)
    00_d133b4a231_showcwt_kernel_scan_past_array
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=5 pc=0xc670ea5c address=0xc6752100 in ffh_read_probe+0x38 (the labelled probe)  eca48bb855c530ee
    01_bcbf3a5630_vf_scale_format_list_compaction
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=5 pc=0xc670ed28 address=0xc6752338 in ffh_read_probe_u8+0x38 (the labelled probe)  3422adb641eb7234
    02_56309e476a_vf_vif_mirror_below_base
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=5 pc=0xc670eea4 address=0xc6752b0c in ffh_read_probe+0x38 (the labelled probe)  7a10c17f9fdd93dc
    03_495b402f27_diracdec_edge_emu_buffer_undersized
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=7 pc=0xc670ec7c address=0xc6752450 in ffh_write_probe_u8+0x58 (the labelled probe)  6bd768271a8f373c
    04_b3c7ebc1ed_swaprect_temp_sized_for_plane0
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=7 pc=0xc670ecb0 address=0xc6752261 in ffh_write_probe_u8+0x58 (the labelled probe)  6c1d02108b8b1149
    05_8553e6ef57_cbs_av1_t35_payload_unpadded
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=5 pc=0xc670eaf4 address=0xc675211b in ffh_read_probe_u8+0x38 (the labelled probe)  0ed8b2d83936d740
    06_041d4f010e_prores_raw_header_len_unchecked
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=5 pc=0xc670ebc8 address=0xc6752220 in ffh_read_probe_u8+0x38 (the labelled probe)  c74abc8d57475431
    07_8880a174d0_librist_read_ignores_caller_size
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=7 pc=0xc670ecc4 address=0xc6752260 in ffh_write_probe_u8+0x58 (the labelled probe)  05185bf77408b4a1
    08_b2df2f4f22_mpegenc_system_header_fixed_128
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=7 pc=0xc670eb78 address=0xc67521c0 in ffh_write_probe_u8+0x58 (the labelled probe)  1e801599dd54e698
    09_16b2049d4d_cfhd_transform2_wider_than_plane
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=7 pc=0xc670eb08 address=0xc6752100 in ffh_write_probe_u8+0x58 (the labelled probe)  85e1a9258374849a
    10_989444060d5f_lut3d_size2_computed_before_directive
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=7 pc=0xc670ea8c address=0xc6752060 in ffh_write_probe_u8+0x58 (the labelled probe)  dc23e62d73306599
    11_76645e096fab_exif_string_clone_drops_terminator
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=5 pc=0xc670ecfc address=0xc6752350 in ffh_read_probe_u8+0x38 (the labelled probe)  fcedf4528960bb67
    12_ad956ff076ea_drawtext_bbox_buffer_ignores_separators
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=7 pc=0xc670eba8 address=0xc6752140 in ffh_write_probe_u8+0x58 (the labelled probe)  72f3319768ffa801
    13_8af6c71d96f4_vif_single_reflection_indexes_below_base
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=5 pc=0xc670ec44 address=0xc67523c0 in ffh_read_probe+0x38 (the labelled probe)  79a26faa52231318
    14_789d7b1b1dff_ffv1dec_fltmap_index_unmasked
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=5 pc=0xc670eaa8 address=0xc67520a0 in ffh_read_probe_u8+0x38 (the labelled probe)  b2232c7baafadc00
    15_e9e6fb879835_tdsc_raw_tile_read_by_geometry_not_size
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=5 pc=0xc670e96c address=0xc6751f80 in ffh_read_probe_u8+0x38 (the labelled probe)  0d60b07499dbb124
    16_ca1c1f29ce47_img2enc_split_planes_read_by_geometry
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=5 pc=0xc670ec14 address=0xc6752240 in ffh_read_probe_u8+0x38 (the labelled probe)  739832962558968a
    17_18761f9fb55c_rtpdec_av1_output_cursor_advanced_for_skipped_obu
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=7 pc=0xc670ea24 address=0xc6751fb0 in ffh_write_probe_u8+0x58 (the labelled probe)  70cdd87254e332c6
    18_e8031e5b9ad2_showcwt_du_scroll_reads_row_past_plane
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=5 pc=0xc670ed04 address=0xc6752310 in ffh_read_probe_u8+0x38 (the labelled probe)  2f2188c02522e7f2
    19_905a4324030e_showcwt_position_initialised_to_sono_size
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=7 pc=0xc670e9fc address=0xc6751f70 in ffh_write_probe_u8+0x58 (the labelled probe)  40d801fd8262ad24
    20_bbdce45fda1e_diracdec_mctmp_rows_below_worst_case
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=7 pc=0xc670eba0 address=0xc6752140 in ffh_write_probe_u8+0x58 (the labelled probe)  08188d80d9483d4b
    21_3fbb9560821b_afir_crossfade_reads_beyond_input_frame
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=5 pc=0xc670eb58 address=0xc6752300 in ffh_read_probe+0x38 (the labelled probe)  4bf5009ecf467d81
    22_242ff799c75f_tdsc_cursor_advanced_by_a_stride_the_loop_already_added
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=7 pc=0xc670eb4c address=0xc67522e0 in ffh_write_probe_u8+0x58 (the labelled probe)  5d48aec7542c36e3
    23_f45da79b2c33_exr_dwa_block_written_full_eight_by_eight
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=7 pc=0xc670ec30 address=0xc67521d0 in ffh_write_probe_u8+0x58 (the labelled probe)  bb786221be7123f7
    24_162f75b5e679_exr_pxr24_uint_branch_strides_by_the_data_window
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=5 pc=0xc670eb84 address=0xc6752180 in ffh_read_probe_u8+0x38 (the labelled probe)  76ad64c4a14fae13
