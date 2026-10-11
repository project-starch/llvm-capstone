# ffmpeg/carved-repros on the Capstone arm `capstone-carve-bounds` -- 2026-10-10

Predictions committed before the run: 26cdee4e4d8d (case 12); the 2026-10-09 readings for cases 0-11.
Platform: Capstone application VM on capstone-qemu 32e7c9754f9a, kernel e58613598c89, firmware 6f2b082cb677, capstone-exec 0752aa7c49c9, module a88ed2159b43, rootfs 9903242c37a7; SDK heap=level0 heap_log=22 runtime 3c72f36acd5b; compiler 7d01722aab88; tool f7b00cf3bc4c.

Reproduce (one VM up, see ports/common/application/README.md):

    python3 capstone/bug-corpora/tools/run-capstone-domain.py \
      --corpus capstone/bug-corpora/ffmpeg/carved-repros --arm capstone-carve-bounds --sdk <SDK, heap=level0> \
      --state <capstone-vm state> --out <fresh dir> \
      --cc-arg=-DFFC_CARVE_BOUNDS

## Controls, from the same boot

    clean    RETURNED  required RETURNED  CONTROL clean RETURNED 1
    oob      CAUGHT    required CAUGHT    cause=7 pc=0xc020e5b8 address=0xc0251948 in ctl_write_probe+0x58 (the labelled probe)
    uaf      RETURNED  required RETURNED  CONTROL uaf RETURNED observed=0xaa reissued=1
    subobj   RETURNED  required RETURNED  CONTROL subobj RETURNED b0=0x5a

## Cases: {'CAUGHT': 13}

    case  outcome  as-predicted  fixed  detail  image(sha256/16)
    00_85407c7e63_mpegvideo_edge_emu_u_block_one_row_short
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=7 pc=0xc020ed74 address=0xc0253160 in ffc_write_probe_u8+0x58 (the labelled probe)  02289abcb39dec50
    01_699341d647_apedec_array_0000_writes_64_into_channel_1
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=7 pc=0xc020edcc address=0xc0252bf0 in ffc_write_probe_u32+0x58 (the labelled probe)  90525d516e5492d0
    02_cd7524fdd1_apedec_filter_3800_reads_past_short_channel
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=5 pc=0xc020ecbc address=0xc0252bc0 in ffc_read_probe_u32+0x38 (the labelled probe)  deb8de07537a1503
    03_55937bb4a7_alsdec_ra_block_runs_into_next_channel_prefix
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=5 pc=0xc020eff4 address=0xc0252e80 in ffc_read_probe_u32+0x38 (the labelled probe)  32321c58c98b45e6
    04_cd09284924_alsdec_mcc_chan_data_one_slot_per_channel
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=7 pc=0xc020f0d8 address=0xc025305c in ffc_write_probe_u32+0x58 (the labelled probe)  3fea85c43053367d
    05_9d3032b960_alsdec_opt_order_past_max_order
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=7 pc=0xc020f0e0 address=0xc0252ea0 in ffc_write_probe_u32+0x58 (the labelled probe)  22c44363137138b0
    06_2d0bea4719_vp9_uv_nnz_ctx_sized_for_420
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=7 pc=0xc020f4fc address=0xc02536c0 in ffc_write_probe_u8+0x58 (the labelled probe)  65f6addee34bddd1
    07_2563a33856_vp9_intra_pred_carved_for_old_bpp
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=7 pc=0xc020f2b8 address=0xc0253290 in ffc_write_probe_u8+0x58 (the labelled probe)  6072c32f1fbba67b
    08_b5ff61695f_swscale_v_line_at_pre_doubling_stride
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=7 pc=0xc020f0c4 address=0xc0252f10 in ffc_write_probe_u32+0x58 (the labelled probe)  7da5a5821155908d
    09_043bcdcdb0_svq1enc_inter_block_runs_into_source
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=7 pc=0xc020ef80 address=0xc0252ed0 in ffc_write_probe_u8+0x58 (the labelled probe)  6a8dbb85f70ccd9b
    10_d2213b6493_rv34_b_block_carved_at_old_linesize
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=7 pc=0xc020efb0 address=0xc0253220 in ffc_write_probe_u8+0x58 (the labelled probe)  b33efa567bce0200
    11_68226ed9ec_vorbis_type1_residue_end_spans_channels
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=5 pc=0xc020f034 address=0xc0253500 in ffc_read_probe_u32+0x38 (the labelled probe)  5dbeeeb5933c1411
    12_ac59fc542f_thumbnail_hbd_tail_index_past_plane_slice
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=5 pc=0xc020ed8c address=0xc0253e8c in ffc_read_probe_u32+0x38 (the labelled probe)  92d4c86b7d6da2ba
