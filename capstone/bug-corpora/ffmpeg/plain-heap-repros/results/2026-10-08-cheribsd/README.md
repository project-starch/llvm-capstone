# ffmpeg/plain-heap-repros on stock CheriBSD purecap, revocation ON -- 2026-10-08

Reproduced by: CHERI_SDK=~/cheri/output/sdk CHERI_SYSROOT=~/cheri/rootfs-purecap \
               CHERI_IMAGE=~/cheri/output/cheribsd-riscv64-purecap.img \
                 bash runners/run-cheribsd.sh <fresh-outdir>

Kernel CHERI-PURECAP-QEMU. Result LINES only: the boot capture is contaminated by
construction, so what is kept is the controls, the per-case verdict, and the supervise
lines carrying the signal, the si_code and the probe address resolved from the ELF.

## Platform controls -- a suite whose controls did not fire is not a reading

    cheribsd-abi:
        CHERI_ABI pointer_bytes=16 runtime_revocation=1
    cheribsd-bounds:
        CHERI_ABI pointer_bytes=16 runtime_revocation=1
        CHERI_BOUNDARY_READY

## Per-case verdicts

    PASS cheribsd-abi
    PASS cheribsd-bounds
    PASS cap-bounds
    PASS ffh-00-fixed
    PASS ffh-00-buggy
    PASS ffh-01-fixed
    PASS ffh-01-buggy
    PASS ffh-02-fixed
    PASS ffh-02-buggy
    PASS ffh-03-fixed
    PASS ffh-03-buggy
    PASS ffh-04-fixed
    PASS ffh-04-buggy
    PASS ffh-05-fixed
    PASS ffh-05-buggy
    PASS ffh-06-fixed
    PASS ffh-06-buggy
    PASS ffh-07-fixed
    PASS ffh-07-buggy
    PASS ffh-08-fixed
    PASS ffh-08-buggy
    PASS ffh-09-fixed
    PASS ffh-09-buggy
    PASS ffh-10-fixed
    PASS ffh-10-buggy
    PASS ffh-11-fixed
    PASS ffh-11-buggy
    PASS ffh-12-fixed
    PASS ffh-12-buggy
    PASS ffh-13-fixed
    PASS ffh-13-buggy
    PASS ffh-14-fixed
    PASS ffh-14-buggy
    PASS ffh-15-fixed
    PASS ffh-15-buggy
    PASS ffh-16-fixed
    PASS ffh-16-buggy
    PASS ffh-17-fixed
    PASS ffh-17-buggy
    PASS ffh-18-fixed
    PASS ffh-18-buggy
    PASS ffh-19-fixed
    PASS ffh-19-buggy
    PASS ffh-20-fixed
    PASS ffh-20-buggy
    PASS ffh-21-fixed
    PASS ffh-21-buggy
    PASS ffh-22-fixed
    PASS ffh-22-buggy
    PASS ffh-23-fixed
    PASS ffh-23-buggy
    PASS ffh-24-fixed
    PASS ffh-24-buggy

## The supervise lines, per buggy arm

    00_d133b4a231_showcwt_kernel_scan_past_array
        SUPERVISE base 0x100000 /tmp/allocator-tests/ffh-00-buggy/target
        SUPERVISE expect ffh_read_probe 0x10213a
        SUPERVISE fault signal=34 code=1 addr=0x10214a pc=0x10214a
        SUPERVISE exit signalled=34
    01_bcbf3a5630_vf_scale_format_list_compaction
        SUPERVISE base 0x100000 /tmp/allocator-tests/ffh-01-buggy/target
        SUPERVISE expect ffh_read_probe_u8 0x102266
        case=1 arm=buggy
        cap=6 touched=6 extent=1 crossed=1 damage=0
        VERDICT DEFECT-REPRODUCED the shift loop's guard was j < nb_formats while its body read formats[j + 1], so it read one element past av_malloc_array -- the value is discarded, the crossing is not
        SUPERVISE exit status=0
    02_56309e476a_vf_vif_mirror_below_base
        SUPERVISE base 0x100000 /tmp/allocator-tests/ffh-02-buggy/target
        SUPERVISE expect ffh_read_probe 0x1022f4
        SUPERVISE fault signal=34 code=1 addr=0x102304 pc=0x102304
        SUPERVISE exit signalled=34
    03_495b402f27_diracdec_edge_emu_buffer_undersized
        SUPERVISE base 0x100000 /tmp/allocator-tests/ffh-03-buggy/target
        SUPERVISE expect ffh_write_probe_u8 0x102270
        SUPERVISE fault signal=34 code=1 addr=0x102288 pc=0x102288
        SUPERVISE exit signalled=34
    04_b3c7ebc1ed_swaprect_temp_sized_for_plane0
        SUPERVISE base 0x100000 /tmp/allocator-tests/ffh-04-buggy/target
        SUPERVISE expect ffh_write_probe_u8 0x10223e
        case=4 arm=buggy
        cap=17 touched=17 extent=1 crossed=1 damage=1
        VERDICT DEFECT-REPRODUCED the scratch row was sized w * pixsteps[0], the luma plane, while the chroma row of a semi-planar 4:2:0 format at odd width needs one byte more -- so the copy wrote past av_malloc_array
        SUPERVISE exit status=0
    05_8553e6ef57_cbs_av1_t35_payload_unpadded
        SUPERVISE base 0x100000 /tmp/allocator-tests/ffh-05-buggy/target
        SUPERVISE expect ffh_read_probe_u8 0x1021c6
        case=5 arm=buggy
        cap=24 touched=27 extent=64 crossed=1 damage=1
        VERDICT DEFECT-REPRODUCED the T.35 payload was allocated at exactly payload_size, so a reader entitled to FFmpeg's 64-byte input padding read past av_buffer_alloc
        SUPERVISE exit status=0
    06_041d4f010e_prores_raw_header_len_unchecked
        SUPERVISE base 0x100000 /tmp/allocator-tests/ffh-06-buggy/target
        SUPERVISE expect ffh_read_probe_u8 0x102230
        SUPERVISE fault signal=34 code=1 addr=0x102240 pc=0x102240
        SUPERVISE exit signalled=34
    07_8880a174d0_librist_read_ignores_caller_size
        SUPERVISE base 0x100000 /tmp/allocator-tests/ffh-07-buggy/target
        SUPERVISE expect ffh_write_probe_u8 0x102256
        SUPERVISE fault signal=34 code=1 addr=0x10226e pc=0x10226e
        SUPERVISE exit signalled=34
    08_b2df2f4f22_mpegenc_system_header_fixed_128
        SUPERVISE base 0x100000 /tmp/allocator-tests/ffh-08-buggy/target
        SUPERVISE expect ffh_write_probe_u8 0x102202
        SUPERVISE fault signal=34 code=1 addr=0x10221a pc=0x10221a
        SUPERVISE exit signalled=34
    09_16b2049d4d_cfhd_transform2_wider_than_plane
        SUPERVISE base 0x100000 /tmp/allocator-tests/ffh-09-buggy/target
        SUPERVISE expect ffh_write_probe_u8 0x1021de
        SUPERVISE fault signal=34 code=1 addr=0x1021f6 pc=0x1021f6
        SUPERVISE exit signalled=34
    10_989444060d5f_lut3d_size2_computed_before_directive
        SUPERVISE base 0x100000 /tmp/allocator-tests/ffh-10-buggy/target
        SUPERVISE expect ffh_write_probe_u8 0x10219e
        SUPERVISE fault signal=34 code=1 addr=0x1021b6 pc=0x1021b6
        SUPERVISE exit signalled=34
    11_76645e096fab_exif_string_clone_drops_terminator
        SUPERVISE base 0x100000 /tmp/allocator-tests/ffh-11-buggy/target
        SUPERVISE expect ffh_read_probe_u8 0x102290
        SUPERVISE fault signal=34 code=1 addr=0x1022a0 pc=0x1022a0
        SUPERVISE exit signalled=34
    12_ad956ff076ea_drawtext_bbox_buffer_ignores_separators
        SUPERVISE base 0x100000 /tmp/allocator-tests/ffh-12-buggy/target
        SUPERVISE expect ffh_write_probe_u8 0x1021d2
        SUPERVISE fault signal=34 code=1 addr=0x1021ea pc=0x1021ea
        SUPERVISE exit signalled=34
    13_8af6c71d96f4_vif_single_reflection_indexes_below_base
        SUPERVISE base 0x100000 /tmp/allocator-tests/ffh-13-buggy/target
        SUPERVISE expect ffh_read_probe 0x10220c
        SUPERVISE fault signal=34 code=1 addr=0x10221c pc=0x10221c
        SUPERVISE exit signalled=34
    14_789d7b1b1dff_ffv1dec_fltmap_index_unmasked
        SUPERVISE base 0x100000 /tmp/allocator-tests/ffh-14-buggy/target
        SUPERVISE expect ffh_read_probe_u8 0x10219c
        SUPERVISE fault signal=34 code=1 addr=0x1021ac pc=0x1021ac
        SUPERVISE exit signalled=34
    15_e9e6fb879835_tdsc_raw_tile_read_by_geometry_not_size
        SUPERVISE base 0x100000 /tmp/allocator-tests/ffh-15-buggy/target
        SUPERVISE expect ffh_read_probe_u8 0x102102
        SUPERVISE fault signal=34 code=1 addr=0x102112 pc=0x102112
        SUPERVISE exit signalled=34
    16_ca1c1f29ce47_img2enc_split_planes_read_by_geometry
        SUPERVISE base 0x100000 /tmp/allocator-tests/ffh-16-buggy/target
        SUPERVISE expect ffh_read_probe_u8 0x1022fc
        SUPERVISE fault signal=34 code=1 addr=0x10230c pc=0x10230c
        SUPERVISE exit signalled=34
    17_18761f9fb55c_rtpdec_av1_output_cursor_advanced_for_skipped_obu
        SUPERVISE base 0x100000 /tmp/allocator-tests/ffh-17-buggy/target
        SUPERVISE expect ffh_write_probe_u8 0x10213c
        SUPERVISE fault signal=34 code=1 addr=0x102154 pc=0x102154
        SUPERVISE exit signalled=34
    18_e8031e5b9ad2_showcwt_du_scroll_reads_row_past_plane
        SUPERVISE base 0x100000 /tmp/allocator-tests/ffh-18-buggy/target
        SUPERVISE expect ffh_read_probe_u8 0x1021c0
        SUPERVISE fault signal=34 code=1 addr=0x1021d0 pc=0x1021d0
        SUPERVISE exit signalled=34
    19_905a4324030e_showcwt_position_initialised_to_sono_size
        SUPERVISE base 0x100000 /tmp/allocator-tests/ffh-19-buggy/target
        SUPERVISE expect ffh_write_probe_u8 0x102104
        SUPERVISE fault signal=34 code=1 addr=0x10211c pc=0x10211c
        SUPERVISE exit signalled=34
    20_bbdce45fda1e_diracdec_mctmp_rows_below_worst_case
        SUPERVISE base 0x100000 /tmp/allocator-tests/ffh-20-buggy/target
        SUPERVISE expect ffh_write_probe_u8 0x1021e0
        SUPERVISE fault signal=34 code=1 addr=0x1021f8 pc=0x1021f8
        SUPERVISE exit signalled=34
    21_3fbb9560821b_afir_crossfade_reads_beyond_input_frame
        SUPERVISE base 0x100000 /tmp/allocator-tests/ffh-21-buggy/target
        SUPERVISE expect ffh_read_probe 0x10218a
        SUPERVISE fault signal=34 code=1 addr=0x10219a pc=0x10219a
        SUPERVISE exit signalled=34
    22_242ff799c75f_tdsc_cursor_advanced_by_a_stride_the_loop_already_added
        SUPERVISE base 0x100000 /tmp/allocator-tests/ffh-22-buggy/target
        SUPERVISE expect ffh_write_probe_u8 0x1021a8
        SUPERVISE fault signal=34 code=1 addr=0x1021c0 pc=0x1021c0
        SUPERVISE exit signalled=34
    23_f45da79b2c33_exr_dwa_block_written_full_eight_by_eight
        SUPERVISE base 0x100000 /tmp/allocator-tests/ffh-23-buggy/target
        SUPERVISE expect ffh_write_probe_u8 0x102210
        SUPERVISE fault signal=34 code=1 addr=0x102228 pc=0x102228
        SUPERVISE exit signalled=34
    24_162f75b5e679_exr_pxr24_uint_branch_strides_by_the_data_window
        SUPERVISE base 0x100000 /tmp/allocator-tests/ffh-24-buggy/target
        SUPERVISE expect ffh_read_probe_u8 0x1021c0
        SUPERVISE fault signal=34 code=1 addr=0x1021d0 pc=0x1021d0
        SUPERVISE exit signalled=34
