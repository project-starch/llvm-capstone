# ffmpeg/carved-repros, arm `cheribsd-carve-bounds` -- 2026-10-09

Platform: stock CheriBSD purecap, kernel CHERI-PURECAP-QEMU, revocation on, the cases built with -DFFC_CARVE_BOUNDS (booted on the second host).
Predictions committed before the run: 2852edcdd747. Runtime revocation reported by the ABI control: 1.

Result LINES only; the boot capture is contaminated by construction.

## Controls, from the same boot

    cheribsd-abi: CHERI_ABI pointer_bytes=16 runtime_revocation=1
    cheribsd-bounds: CHERI_BOUNDARY_READY
    carve-control-buggy: exit 162; SUPERVISE fault signal=34 code=1 addr=0x1022ba pc=0x1022ba
    carve-control-fixed: exit 0

## Cases: {'CAUGHT': 12}

    00_85407c7e63_mpegvideo_edge_emu_u_block_one_row_short
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x1024e2 INSIDE ffc_write_probe_u8 [0x1024ca, 0x1024ee)
        SUPERVISE fault signal=34 code=1 addr=0x1024e2 pc=0x1024e2
    01_699341d647_apedec_array_0000_writes_64_into_channel_1
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x102486 INSIDE ffc_write_probe_u32 [0x10246e, 0x102490)
        SUPERVISE fault signal=34 code=1 addr=0x102486 pc=0x102486
    02_cd7524fdd1_apedec_filter_3800_reads_past_short_channel
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x10243a INSIDE ffc_read_probe_u32 [0x10242a, 0x102444)
        SUPERVISE fault signal=34 code=1 addr=0x10243a pc=0x10243a
    03_55937bb4a7_alsdec_ra_block_runs_into_next_channel_prefix
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x102570 INSIDE ffc_read_probe_u32 [0x102560, 0x10257a)
        SUPERVISE fault signal=34 code=1 addr=0x102570 pc=0x102570
    04_cd09284924_alsdec_mcc_chan_data_one_slot_per_channel
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x10263e INSIDE ffc_write_probe_u32 [0x102626, 0x102648)
        SUPERVISE fault signal=34 code=1 addr=0x10263e pc=0x10263e
    05_9d3032b960_alsdec_opt_order_past_max_order
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x1025e2 INSIDE ffc_write_probe_u32 [0x1025ca, 0x1025ec)
        SUPERVISE fault signal=34 code=1 addr=0x1025e2 pc=0x1025e2
    06_2d0bea4719_vp9_uv_nnz_ctx_sized_for_420
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x102db4 INSIDE ffc_write_probe_u8 [0x102d9c, 0x102dc0)
        SUPERVISE fault signal=34 code=1 addr=0x102db4 pc=0x102db4
    07_2563a33856_vp9_intra_pred_carved_for_old_bpp
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x102a80 INSIDE ffc_write_probe_u8 [0x102a68, 0x102a8c)
        SUPERVISE fault signal=34 code=1 addr=0x102a80 pc=0x102a80
    08_b5ff61695f_swscale_v_line_at_pre_doubling_stride
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x102566 INSIDE ffc_write_probe_u32 [0x10254e, 0x102570)
        SUPERVISE fault signal=34 code=1 addr=0x102566 pc=0x102566
    09_043bcdcdb0_svq1enc_inter_block_runs_into_source
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x1025a6 INSIDE ffc_write_probe_u8 [0x10258e, 0x1025b2)
        SUPERVISE fault signal=34 code=1 addr=0x1025a6 pc=0x1025a6
    10_d2213b6493_rv34_b_block_carved_at_old_linesize
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x102612 INSIDE ffc_write_probe_u8 [0x1025fa, 0x10261e)
        SUPERVISE fault signal=34 code=1 addr=0x102612 pc=0x102612
    11_68226ed9ec_vorbis_type1_residue_end_spans_channels
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x102642 INSIDE ffc_read_probe_u32 [0x102632, 0x10264c)
        SUPERVISE fault signal=34 code=1 addr=0x102642 pc=0x102642

Granted bounds: every `carve` line the runs printed reads `bounds` equal to the requested `len` with
`base_eq=1`, so no region was rounded up by CHERI-128's representability and no catch or miss here
comes from a round-up.
