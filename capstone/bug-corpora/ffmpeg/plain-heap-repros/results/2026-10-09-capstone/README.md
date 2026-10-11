# ffmpeg/plain-heap-repros on the Capstone application domain arms -- 2026-10-09

Predictions committed before the run: 60f1c5be25a5 (tools/run-capstone-domain.py, predicted()).

Reproduce (one VM up, see ports/common/application/README.md):

    python3 capstone/bug-corpora/tools/run-capstone-domain.py \
      --corpus capstone/bug-corpora/ffmpeg/plain-heap-repros --arm spatial|sublet \
      --sdk <application SDK built with CAPSTONE_APPLICATION_HEAP=level0|sublet> \
      --state <capstone-vm state> --out <fresh dir>

Result LINES only, from each arm's record.json.

## spatial

    platform  Capstone application VM on capstone-qemu 32e7c9754f9a, kernel e58613598c89, firmware 6f2b082cb677, capstone-exec 0752aa7c49c9, module a88ed2159b43; SDK heap=level0 runtime 3c72f36acd5b; compiler 7d01722aab88
    env       {'CAPSTONE_GP_NONLIN': '1', 'CAPSTONE_REV_NODES': '65536'}
    tool      0383d3c2caa89fb6   predicted buggy outcome: CAUGHT
    control clean RETURNED  required RETURNED  CONTROL clean RETURNED 1
    control oob   CAUGHT    required CAUGHT    cause=7 pc=0xc020e5b8 address=0xc0251768 in ctl_write_probe+0x58 (the labelled probe)
    control uaf   RETURNED  required RETURNED  CONTROL uaf RETURNED observed=0xaa reissued=1
    00_d133b4a231_showcwt_kernel_scan_past_array                       fixed FIXED exit=0   buggy CAUGHT             as predicted cause=5 pc=0xc020ea5c address=0xc0252100 in ffh_read_probe+0x38 (the labelled probe)  image eca48bb855c530ee
    01_bcbf3a5630_vf_scale_format_list_compaction                      fixed FIXED exit=0   buggy CAUGHT             as predicted cause=5 pc=0xc020ed28 address=0xc0252338 in ffh_read_probe_u8+0x38 (the labelled probe)  image 3422adb641eb7234
    02_56309e476a_vf_vif_mirror_below_base                             fixed FIXED exit=0   buggy CAUGHT             as predicted cause=5 pc=0xc020eea4 address=0xc0252b0c in ffh_read_probe+0x38 (the labelled probe)  image 7a10c17f9fdd93dc
    03_495b402f27_diracdec_edge_emu_buffer_undersized                  fixed FIXED exit=0   buggy CAUGHT             as predicted cause=7 pc=0xc020ec7c address=0xc0252450 in ffh_write_probe_u8+0x58 (the labelled probe)  image 6bd768271a8f373c
    04_b3c7ebc1ed_swaprect_temp_sized_for_plane0                       fixed FIXED exit=0   buggy CAUGHT             as predicted cause=7 pc=0xc020ecb0 address=0xc0252261 in ffh_write_probe_u8+0x58 (the labelled probe)  image 6c1d02108b8b1149
    05_8553e6ef57_cbs_av1_t35_payload_unpadded                         fixed FIXED exit=0   buggy CAUGHT             as predicted cause=5 pc=0xc020eaf4 address=0xc025211b in ffh_read_probe_u8+0x38 (the labelled probe)  image 0ed8b2d83936d740
    06_041d4f010e_prores_raw_header_len_unchecked                      fixed FIXED exit=0   buggy CAUGHT             as predicted cause=5 pc=0xc020ebc8 address=0xc0252220 in ffh_read_probe_u8+0x38 (the labelled probe)  image c74abc8d57475431
    07_8880a174d0_librist_read_ignores_caller_size                     fixed FIXED exit=0   buggy CAUGHT             as predicted cause=7 pc=0xc020ecc4 address=0xc0252260 in ffh_write_probe_u8+0x58 (the labelled probe)  image 05185bf77408b4a1
    08_b2df2f4f22_mpegenc_system_header_fixed_128                      fixed FIXED exit=0   buggy CAUGHT             as predicted cause=7 pc=0xc020eb78 address=0xc02521c0 in ffh_write_probe_u8+0x58 (the labelled probe)  image 1e801599dd54e698
    09_16b2049d4d_cfhd_transform2_wider_than_plane                     fixed FIXED exit=0   buggy CAUGHT             as predicted cause=7 pc=0xc020eb08 address=0xc0252100 in ffh_write_probe_u8+0x58 (the labelled probe)  image 85e1a9258374849a
    10_989444060d5f_lut3d_size2_computed_before_directive              fixed FIXED exit=0   buggy CAUGHT             as predicted cause=7 pc=0xc020ea8c address=0xc0252060 in ffh_write_probe_u8+0x58 (the labelled probe)  image dc23e62d73306599
    11_76645e096fab_exif_string_clone_drops_terminator                 fixed FIXED exit=0   buggy CAUGHT             as predicted cause=5 pc=0xc020ecfc address=0xc0252350 in ffh_read_probe_u8+0x38 (the labelled probe)  image fcedf4528960bb67
    12_ad956ff076ea_drawtext_bbox_buffer_ignores_separators            fixed FIXED exit=0   buggy CAUGHT             as predicted cause=7 pc=0xc020eba8 address=0xc0252140 in ffh_write_probe_u8+0x58 (the labelled probe)  image 72f3319768ffa801
    13_8af6c71d96f4_vif_single_reflection_indexes_below_base           fixed FIXED exit=0   buggy CAUGHT             as predicted cause=5 pc=0xc020ec44 address=0xc02523c0 in ffh_read_probe+0x38 (the labelled probe)  image 79a26faa52231318
    14_789d7b1b1dff_ffv1dec_fltmap_index_unmasked                      fixed FIXED exit=0   buggy CAUGHT             as predicted cause=5 pc=0xc020eaa8 address=0xc02520a0 in ffh_read_probe_u8+0x38 (the labelled probe)  image b2232c7baafadc00
    15_e9e6fb879835_tdsc_raw_tile_read_by_geometry_not_size            fixed FIXED exit=0   buggy CAUGHT             as predicted cause=5 pc=0xc020e96c address=0xc0251f80 in ffh_read_probe_u8+0x38 (the labelled probe)  image 0d60b07499dbb124
    16_ca1c1f29ce47_img2enc_split_planes_read_by_geometry              fixed FIXED exit=0   buggy CAUGHT             as predicted cause=5 pc=0xc020ec14 address=0xc0252240 in ffh_read_probe_u8+0x38 (the labelled probe)  image 739832962558968a
    17_18761f9fb55c_rtpdec_av1_output_cursor_advanced_for_skipped_obu  fixed FIXED exit=0   buggy CAUGHT             as predicted cause=7 pc=0xc020ea24 address=0xc0251fb0 in ffh_write_probe_u8+0x58 (the labelled probe)  image 70cdd87254e332c6
    18_e8031e5b9ad2_showcwt_du_scroll_reads_row_past_plane             fixed FIXED exit=0   buggy CAUGHT             as predicted cause=5 pc=0xc020ed04 address=0xc0252310 in ffh_read_probe_u8+0x38 (the labelled probe)  image 2f2188c02522e7f2
    19_905a4324030e_showcwt_position_initialised_to_sono_size          fixed FIXED exit=0   buggy CAUGHT             as predicted cause=7 pc=0xc020e9fc address=0xc0251f70 in ffh_write_probe_u8+0x58 (the labelled probe)  image 40d801fd8262ad24
    20_bbdce45fda1e_diracdec_mctmp_rows_below_worst_case               fixed FIXED exit=0   buggy CAUGHT             as predicted cause=7 pc=0xc020eba0 address=0xc0252140 in ffh_write_probe_u8+0x58 (the labelled probe)  image 08188d80d9483d4b
    21_3fbb9560821b_afir_crossfade_reads_beyond_input_frame            fixed FIXED exit=0   buggy CAUGHT             as predicted cause=5 pc=0xc020eb58 address=0xc0252300 in ffh_read_probe+0x38 (the labelled probe)  image 4bf5009ecf467d81
    22_242ff799c75f_tdsc_cursor_advanced_by_a_stride_the_loop_already_added fixed FIXED exit=0   buggy CAUGHT             as predicted cause=7 pc=0xc020eb4c address=0xc02522e0 in ffh_write_probe_u8+0x58 (the labelled probe)  image 5d48aec7542c36e3
    23_f45da79b2c33_exr_dwa_block_written_full_eight_by_eight          fixed FIXED exit=0   buggy CAUGHT             as predicted cause=7 pc=0xc020ec30 address=0xc02521d0 in ffh_write_probe_u8+0x58 (the labelled probe)  image bb786221be7123f7
    24_162f75b5e679_exr_pxr24_uint_branch_strides_by_the_data_window   fixed FIXED exit=0   buggy CAUGHT             as predicted cause=5 pc=0xc020eb84 address=0xc0252180 in ffh_read_probe_u8+0x38 (the labelled probe)  image 76ad64c4a14fae13

## sublet

    platform  Capstone application VM on capstone-qemu 32e7c9754f9a, kernel e58613598c89, firmware 6f2b082cb677, capstone-exec 0752aa7c49c9, module a88ed2159b43; SDK heap=sublet runtime b73508e5005b; compiler 7d01722aab88
    env       {'CAPSTONE_GP_NONLIN': '1', 'CAPSTONE_REV_NODES': '65536'}
    tool      0383d3c2caa89fb6   predicted buggy outcome: CAUGHT
    control clean RETURNED  required RETURNED  CONTROL clean RETURNED 1
    control oob   CAUGHT    required CAUGHT    cause=7 pc=0xc020f70c address=0xc8802018 in ctl_write_probe+0x58 (the labelled probe)
    control uaf   CAUGHT    required CAUGHT    cause=24 pc=0xc020f758 address=0x0 in ctl_read_probe+0x38 (the labelled probe)
    00_d133b4a231_showcwt_kernel_scan_past_array                       fixed FIXED exit=0   buggy CAUGHT             as predicted cause=5 pc=0xc020f960 address=0xc8802010 in ffh_read_probe+0x38 (the labelled probe)  image c69f9712b87303e8
    01_bcbf3a5630_vf_scale_format_list_compaction                      fixed FIXED exit=0   buggy CAUGHT             as predicted cause=5 pc=0xc020fc2c address=0xc8802018 in ffh_read_probe_u8+0x38 (the labelled probe)  image e479bd160dac5802
    02_56309e476a_vf_vif_mirror_below_base                             fixed FIXED exit=0   buggy CAUGHT             as predicted cause=5 pc=0xc020fda8 address=0xc8801ffc in ffh_read_probe+0x38 (the labelled probe)  image 9d3bfdb8d280a5ad
    03_495b402f27_diracdec_edge_emu_buffer_undersized                  fixed FIXED exit=0   buggy CAUGHT             as predicted cause=7 pc=0xc020fb80 address=0xc8802200 in ffh_write_probe_u8+0x58 (the labelled probe)  image ca86a55dfdbae136
    04_b3c7ebc1ed_swaprect_temp_sized_for_plane0                       fixed FIXED exit=0   buggy CAUGHT             as predicted cause=7 pc=0xc020fbb4 address=0xc8802011 in ffh_write_probe_u8+0x58 (the labelled probe)  image f490c0ce9f19baf6
    05_8553e6ef57_cbs_av1_t35_payload_unpadded                         fixed FIXED exit=0   buggy CAUGHT             as predicted cause=5 pc=0xc020f9f8 address=0xc880201b in ffh_read_probe_u8+0x38 (the labelled probe)  image fe1806e9470767ff
    06_041d4f010e_prores_raw_header_len_unchecked                      fixed FIXED exit=0   buggy CAUGHT             as predicted cause=5 pc=0xc020facc address=0xc8802030 in ffh_read_probe_u8+0x38 (the labelled probe)  image 95542e9300fa664a
    07_8880a174d0_librist_read_ignores_caller_size                     fixed FIXED exit=0   buggy CAUGHT             as predicted cause=7 pc=0xc020fbc8 address=0xc8802020 in ffh_write_probe_u8+0x58 (the labelled probe)  image 0125044e032ee7a4
    08_b2df2f4f22_mpegenc_system_header_fixed_128                      fixed FIXED exit=0   buggy CAUGHT             as predicted cause=7 pc=0xc020fa7c address=0xc88020a0 in ffh_write_probe_u8+0x58 (the labelled probe)  image 0e3cfc5415803e76
    09_16b2049d4d_cfhd_transform2_wider_than_plane                     fixed FIXED exit=0   buggy CAUGHT             as predicted cause=7 pc=0xc020fa0c address=0xc8802040 in ffh_write_probe_u8+0x58 (the labelled probe)  image b5af5ac2306f1cf5
    10_989444060d5f_lut3d_size2_computed_before_directive              fixed FIXED exit=0   buggy CAUGHT             as predicted cause=7 pc=0xc020f990 address=0xc8802040 in ffh_write_probe_u8+0x58 (the labelled probe)  image f8d465ae487921fd
    11_76645e096fab_exif_string_clone_drops_terminator                 fixed FIXED exit=0   buggy CAUGHT             as predicted cause=5 pc=0xc020fc00 address=0xc8802110 in ffh_read_probe_u8+0x38 (the labelled probe)  image 91df59fd38331937
    12_ad956ff076ea_drawtext_bbox_buffer_ignores_separators            fixed FIXED exit=0   buggy CAUGHT             as predicted cause=7 pc=0xc020faac address=0xc8802010 in ffh_write_probe_u8+0x58 (the labelled probe)  image f72b8d9f40f82373
    13_8af6c71d96f4_vif_single_reflection_indexes_below_base           fixed FIXED exit=0   buggy CAUGHT             as predicted cause=5 pc=0xc020fb48 address=0xc8801ff0 in ffh_read_probe+0x38 (the labelled probe)  image dc2ecfd06f68e31a
    14_789d7b1b1dff_ffv1dec_fltmap_index_unmasked                      fixed FIXED exit=0   buggy CAUGHT             as predicted cause=5 pc=0xc020f9ac address=0xc8802010 in ffh_read_probe_u8+0x38 (the labelled probe)  image 6e97383f48d2f8f7
    15_e9e6fb879835_tdsc_raw_tile_read_by_geometry_not_size            fixed FIXED exit=0   buggy CAUGHT             as predicted cause=5 pc=0xc020f870 address=0xc8802030 in ffh_read_probe_u8+0x38 (the labelled probe)  image 24cb58bf6448f441
    16_ca1c1f29ce47_img2enc_split_planes_read_by_geometry              fixed FIXED exit=0   buggy CAUGHT             as predicted cause=5 pc=0xc020fb18 address=0xc8802020 in ffh_read_probe_u8+0x38 (the labelled probe)  image 4c44ae26336f0d2f
    17_18761f9fb55c_rtpdec_av1_output_cursor_advanced_for_skipped_obu  fixed FIXED exit=0   buggy CAUGHT             as predicted cause=7 pc=0xc020f928 address=0xc8802010 in ffh_write_probe_u8+0x58 (the labelled probe)  image 191867dbc92a4b0e
    18_e8031e5b9ad2_showcwt_du_scroll_reads_row_past_plane             fixed FIXED exit=0   buggy CAUGHT             as predicted cause=5 pc=0xc020fc08 address=0xc8802020 in ffh_read_probe_u8+0x38 (the labelled probe)  image e6b84362223fd85e
    19_905a4324030e_showcwt_position_initialised_to_sono_size          fixed FIXED exit=0   buggy CAUGHT             as predicted cause=7 pc=0xc020f900 address=0xc8802020 in ffh_write_probe_u8+0x58 (the labelled probe)  image 73735fd86aa62523
    20_bbdce45fda1e_diracdec_mctmp_rows_below_worst_case               fixed FIXED exit=0   buggy CAUGHT             as predicted cause=7 pc=0xc020faa4 address=0xc8802010 in ffh_write_probe_u8+0x58 (the labelled probe)  image 1b492531a47ae79f
    21_3fbb9560821b_afir_crossfade_reads_beyond_input_frame            fixed FIXED exit=0   buggy CAUGHT             as predicted cause=5 pc=0xc020fa5c address=0xc8802010 in ffh_read_probe+0x38 (the labelled probe)  image 71e0022881babf10
    22_242ff799c75f_tdsc_cursor_advanced_by_a_stride_the_loop_already_added fixed FIXED exit=0   buggy CAUGHT             as predicted cause=7 pc=0xc020fa50 address=0xc8802200 in ffh_write_probe_u8+0x58 (the labelled probe)  image 4c09f2291c0e7cd3
    23_f45da79b2c33_exr_dwa_block_written_full_eight_by_eight          fixed FIXED exit=0   buggy CAUGHT             as predicted cause=7 pc=0xc020fb34 address=0xc8802010 in ffh_write_probe_u8+0x58 (the labelled probe)  image 1fcc2ea8ef07405f
    24_162f75b5e679_exr_pxr24_uint_branch_strides_by_the_data_window   fixed FIXED exit=0   buggy CAUGHT             as predicted cause=5 pc=0xc020fa88 address=0xc8802010 in ffh_read_probe_u8+0x38 (the labelled probe)  image d1c678206894ff58

