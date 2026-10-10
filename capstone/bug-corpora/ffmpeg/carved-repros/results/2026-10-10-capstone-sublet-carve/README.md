# ffmpeg/carved-repros on the Capstone arm `sublet-carve` -- 2026-10-10

Predictions committed before the run: 26cdee4e4d8d (case 12); the 2026-10-09 readings for cases 0-11.
Platform: Capstone application VM on capstone-qemu 32e7c9754f9a, kernel e58613598c89, firmware 6f2b082cb677, capstone-exec 0752aa7c49c9, module a88ed2159b43, rootfs 9903242c37a7; SDK heap=sublet heap_log=22 runtime b73508e5005b; compiler 7d01722aab88; tool f7b00cf3bc4c.

Reproduce (one VM up, see ports/common/application/README.md):

    python3 capstone/bug-corpora/tools/run-capstone-domain.py \
      --corpus capstone/bug-corpora/ffmpeg/carved-repros --arm sublet-carve --sdk <SDK, heap=sublet> \
      --state <capstone-vm state> --out <fresh dir> \
      --cc-arg=-DFFC_SUBLET_CARVE \
      --cc-arg=-I<repo>/capstone/runtime/include

## Controls, from the same boot

    clean    RETURNED  required RETURNED  CONTROL clean RETURNED 1
    oob      CAUGHT    required CAUGHT    cause=7 pc=0xc020f70c address=0xc8802018 in ctl_write_probe+0x58 (the labelled probe)
    uaf      CAUGHT    required CAUGHT    cause=24 pc=0xc020f758 address=0x0 in ctl_read_probe+0x38 (the labelled probe)
    subobj   RETURNED  required RETURNED  CONTROL subobj RETURNED b0=0x5a
    carve-99 fixed: FIXED VERDICT FIXED control: the carve's last byte | buggy: CAUGHT cause=7 pc=0xc020fffc address=0xc8802010 in ffc_write_probe_u8+0x58 (the labelled probe) | required: fixed FIXED, buggy CAUGHT at the probe
    carve-98 fixed: FIXED VERDICT FIXED control: the same write, no re-carve | buggy: CAUGHT cause=24 pc=0xc020ffbc address=0x0 in ffc_write_probe_u8+0x58 (the labelled probe) | required: fixed FIXED, buggy CAUGHT at the probe

## Cases: {'CAUGHT': 13}

    case  outcome  as-predicted  fixed  detail  image(sha256/16)
    00_85407c7e63_mpegvideo_edge_emu_u_block_one_row_short
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=7 pc=0xc0210370 address=0xc88205a0 in ffc_write_probe_u8+0x58 (the labelled probe)  d9c6c0c76a99c816
    01_699341d647_apedec_array_0000_writes_64_into_channel_1
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=7 pc=0xc02103cc address=0xc88020a0 in ffc_write_probe_u32+0x58 (the labelled probe)  ba7f019640da36fd
    02_cd7524fdd1_apedec_filter_3800_reads_past_short_channel
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=5 pc=0xc02102bc address=0xc8802120 in ffc_read_probe_u32+0x38 (the labelled probe)  151d9065e30125e8
    03_55937bb4a7_alsdec_ra_block_runs_into_next_channel_prefix
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=5 pc=0xc02105f4 address=0xc88020a0 in ffc_read_probe_u32+0x38 (the labelled probe)  8ee1ad5b745cadd0
    04_cd09284924_alsdec_mcc_chan_data_one_slot_per_channel
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=7 pc=0xc02106dc address=0xc880202c in ffc_write_probe_u32+0x58 (the labelled probe)  00eb8c5bead2a27f
    05_9d3032b960_alsdec_opt_order_past_max_order
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=7 pc=0xc02106e0 address=0xc8802050 in ffc_write_probe_u32+0x58 (the labelled probe)  20e65feeb244f1cc
    06_2d0bea4719_vp9_uv_nnz_ctx_sized_for_420
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=7 pc=0xc0210b00 address=0xc8802280 in ffc_write_probe_u8+0x58 (the labelled probe)  a572f605305e301c
    07_2563a33856_vp9_intra_pred_carved_for_old_bpp
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=7 pc=0xc02108bc address=0xc8802080 in ffc_write_probe_u8+0x58 (the labelled probe)  7c1864836cf80bdc
    08_b5ff61695f_swscale_v_line_at_pre_doubling_stride
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=7 pc=0xc02106c8 address=0xc88020d0 in ffc_write_probe_u32+0x58 (the labelled probe)  6e2a6ec4ca8772fb
    09_043bcdcdb0_svq1enc_inter_block_runs_into_source
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=7 pc=0xc0210584 address=0xc8802100 in ffc_write_probe_u8+0x58 (the labelled probe)  2fc3d697b9654def
    10_d2213b6493_rv34_b_block_carved_at_old_linesize
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=7 pc=0xc02105b4 address=0xc8802400 in ffc_write_probe_u8+0x58 (the labelled probe)  3ac680c8c4fca545
    11_68226ed9ec_vorbis_type1_residue_end_spans_channels
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=5 pc=0xc0210650 address=0xc8802200 in ffc_read_probe_u32+0x38 (the labelled probe)  c52ab4cf67446e3e
    12_ac59fc542f_thumbnail_hbd_tail_index_past_plane_slice
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=5 pc=0xc0210380 address=0xc8804ffc in ffc_read_probe_u32+0x38 (the labelled probe)  013420a374d09953
