# ffmpeg/carved-repros under AddressSanitizer -- 2026-10-09

Runner: runners/run-asan.sh -> tools/run-native-asan.py. The prediction (SILENT) was written in
run-asan.sh's header before the run and committed with the result (2852edcdd747), not before it.

Build: cc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0; -fsanitize=address -fno-omit-frame-pointer -g -O0
ASAN_OPTIONS: detect_leaks=0:abort_on_error=0:halt_on_error=1:color=never

## Controls

    asan-control past 34816: heap-buffer-overflow (required heap-buffer-overflow)
    asan-control past 132: heap-buffer-overflow (required heap-buffer-overflow)
    asan-control uaf 34816: heap-use-after-free (required heap-use-after-free)

## Cases

    00_85407c7e63_mpegvideo_edge_emu_u_block_one_row_short         fixed rc=0  buggy SILENT    the defect reproduced (VERDICT DEFECT-REPRODUCED) and ASan printed nothing
    01_699341d647_apedec_array_0000_writes_64_into_channel_1       fixed rc=0  buggy SILENT    the defect reproduced (VERDICT DEFECT-REPRODUCED) and ASan printed nothing
    02_cd7524fdd1_apedec_filter_3800_reads_past_short_channel      fixed rc=0  buggy SILENT    the defect reproduced (VERDICT DEFECT-REPRODUCED) and ASan printed nothing
    03_55937bb4a7_alsdec_ra_block_runs_into_next_channel_prefix    fixed rc=0  buggy SILENT    the defect reproduced (VERDICT DEFECT-REPRODUCED) and ASan printed nothing
    04_cd09284924_alsdec_mcc_chan_data_one_slot_per_channel        fixed rc=0  buggy SILENT    the defect reproduced (VERDICT DEFECT-REPRODUCED) and ASan printed nothing
    05_9d3032b960_alsdec_opt_order_past_max_order                  fixed rc=0  buggy SILENT    the defect reproduced (VERDICT DEFECT-REPRODUCED) and ASan printed nothing
    06_2d0bea4719_vp9_uv_nnz_ctx_sized_for_420                     fixed rc=0  buggy SILENT    the defect reproduced (VERDICT DEFECT-REPRODUCED) and ASan printed nothing
    07_2563a33856_vp9_intra_pred_carved_for_old_bpp                fixed rc=0  buggy SILENT    the defect reproduced (VERDICT DEFECT-REPRODUCED) and ASan printed nothing
    08_b5ff61695f_swscale_v_line_at_pre_doubling_stride            fixed rc=0  buggy SILENT    the defect reproduced (VERDICT DEFECT-REPRODUCED) and ASan printed nothing
    09_043bcdcdb0_svq1enc_inter_block_runs_into_source             fixed rc=0  buggy SILENT    the defect reproduced (VERDICT DEFECT-REPRODUCED) and ASan printed nothing
    10_d2213b6493_rv34_b_block_carved_at_old_linesize              fixed rc=0  buggy SILENT    the defect reproduced (VERDICT DEFECT-REPRODUCED) and ASan printed nothing
    11_68226ed9ec_vorbis_type1_residue_end_spans_channels          fixed rc=0  buggy SILENT    the defect reproduced (VERDICT DEFECT-REPRODUCED) and ASan printed nothing
