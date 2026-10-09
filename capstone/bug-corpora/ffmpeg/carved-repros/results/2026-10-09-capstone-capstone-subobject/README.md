# ffmpeg/carved-repros on the Capstone arm `capstone-subobject` -- 2026-10-09

Predictions committed before the run: 2852edcdd747.
Platform: Capstone application VM on capstone-qemu 32e7c9754f9a, kernel e58613598c89, firmware 6f2b082cb677, capstone-exec 0752aa7c49c9, module a88ed2159b43; SDK heap=level0 heap_log=22 runtime 3c72f36acd5b; compiler 7d01722aab88.

Reproduce (one VM up, see ports/common/application/README.md):

    python3 capstone/bug-corpora/tools/run-capstone-domain.py \
      --corpus capstone/bug-corpora/ffmpeg/carved-repros --arm capstone-subobject --sdk <SDK, heap=level0> \
      --state <capstone-vm state> --out <fresh dir> \
      --cc-arg=-Xclang \
      --cc-arg=-fcapstone-subobject-bounds

## Controls, from the same boot

    clean  RETURNED  required RETURNED  CONTROL clean RETURNED 1
    oob    CAUGHT    required CAUGHT    cause=7 pc=0xc670e5b8 address=0xc6751958 in ctl_write_probe+0x58 (the labelled probe)
    uaf    RETURNED  required RETURNED  CONTROL uaf RETURNED observed=0xaa reissued=1
    subobj CAUGHT    required CAUGHT    cause=7 pc=0xc670e5b8 address=0xc6751948 in ctl_write_probe+0x58 (the labelled probe)

## Cases: {'DEFECT-REPRODUCED': 12}

    case  outcome  as-predicted  fixed  detail  image(sha256/16)
    00_85407c7e63_mpegvideo_edge_emu_u_block_one_row_short
        DEFECT-REPRODUCED  predicted DEFECT-REPRODUCED  held  fixed: FIXED exit=0
        VERDICT DEFECT-REPRODUCED U's edge emulation wrote 9 + field_based rows into a 9-row block, so its tenth row was V's first, which V's emulation then overwrote  218dcf0e7af839c4
    01_699341d647_apedec_array_0000_writes_64_into_channel_1
        DEFECT-REPRODUCED  predicted DEFECT-REPRODUCED  held  fixed: FIXED exit=0
        VERDICT DEFECT-REPRODUCED decode_array_0000 wrote 64 outputs into a 40-sample channel, so 24 of them landed in decoded[1]  f24c47c9814c2df9
    02_cd7524fdd1_apedec_filter_3800_reads_past_short_channel
        DEFECT-REPRODUCED  predicted DEFECT-REPRODUCED  held  fixed: FIXED exit=0
        VERDICT DEFECT-REPRODUCED the delay line read 128 samples from a 72-sample channel, 56 of them from decoded[1]  043f687bf9620828
    03_55937bb4a7_alsdec_ra_block_runs_into_next_channel_prefix
        DEFECT-REPRODUCED  predicted DEFECT-REPRODUCED  held  fixed: FIXED exit=0
        VERDICT DEFECT-REPRODUCED the RA loop reconstructed 24 samples of a 16-sample block, so it decremented the first of channel 1's carry-over samples  82dda23057777b28
    04_cd09284924_alsdec_mcc_chan_data_one_slot_per_channel
        DEFECT-REPRODUCED  predicted DEFECT-REPRODUCED  held  fixed: FIXED exit=0
        VERDICT DEFECT-REPRODUCED channel 0's dependency list ran one slot past its one-slot carve, and channel 1's entry then replaced its terminator  1218288e3ac8f2b8
    05_9d3032b960_alsdec_opt_order_past_max_order
        DEFECT-REPRODUCED  predicted DEFECT-REPRODUCED  held  fixed: FIXED exit=0
        VERDICT DEFECT-REPRODUCED opt_order 31 wrote 11 coefficients past channel 0's 20-word slice, into channel 1's  21dcf6ad2d463ede
    06_2d0bea4719_vp9_uv_nnz_ctx_sized_for_420
        DEFECT-REPRODUCED  predicted DEFECT-REPRODUCED  held  fixed: FIXED exit=0
        VERDICT DEFECT-REPRODUCED the 4:4:4 clear wrote 16 bytes per superblock column into an 8-byte above_uv_nnz_ctx[0], running into above_uv_nnz_ctx[1]  23ff3451395779f4
    07_2563a33856_vp9_intra_pred_carved_for_old_bpp
        DEFECT-REPRODUCED  predicted DEFECT-REPRODUCED  held  fixed: FIXED exit=0
        VERDICT DEFECT-REPRODUCED a thread at 10-bit backed up 256 luma bytes into an intra_pred_data[0] carved for 8-bit, 128 bytes, running into intra_pred_data[1]  9617849e63fca3a5
    08_b5ff61695f_swscale_v_line_at_pre_doubling_stride
        DEFECT-REPRODUCED  predicted DEFECT-REPRODUCED  held  fixed: FIXED exit=0
        VERDICT DEFECT-REPRODUCED U's 256 bytes of int32 samples ran past V's offset of 208 bytes, and V then overwrote U's tail  bd7d6bb27067281d
    09_043bcdcdb0_svq1enc_inter_block_runs_into_source
        DEFECT-REPRODUCED  predicted DEFECT-REPRODUCED  held  fixed: FIXED exit=0
        VERDICT DEFECT-REPRODUCED the 16x16 inter prediction at temp + 16 with a 16-byte stride ended past temp's 16 rows, overwriting src's first row  e0bfbc0b2d1a4f60
    10_d2213b6493_rv34_b_block_carved_at_old_linesize
        DEFECT-REPRODUCED  predicted DEFECT-REPRODUCED  held  fixed: FIXED exit=0
        VERDICT DEFECT-REPRODUCED after the linesize doubled, direction 0's prediction ran from tmp_b_block_y[0] into tmp_b_block_y[1], which direction 1 then overwrote  f97b1849a5458a47
    11_68226ed9ec_vorbis_type1_residue_end_spans_channels
        DEFECT-REPRODUCED  predicted DEFECT-REPRODUCED  held  fixed: FIXED exit=0
        VERDICT DEFECT-REPRODUCED a type-1 residue ending at 2 * vlen added channel 0's codevectors into channel 1's slice of channel_residues  7fd7189dcd314f1f
