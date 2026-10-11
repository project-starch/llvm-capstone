# ffmpeg/carved-repros, arm `cheribsd-subobject` -- 2026-10-09

Platform: stock CheriBSD purecap, kernel CHERI-PURECAP-QEMU, revocation on, every case built with -Xclang -cheri-bounds=subobject-safe (booted on the second host).
Predictions committed before the run: 2852edcdd747. Runtime revocation reported by the ABI control: 1.

Result LINES only; the boot capture is contaminated by construction.

## Controls, from the same boot

    cheribsd-abi: CHERI_ABI pointer_bytes=16 runtime_revocation=1
    cheribsd-bounds: CHERI_BOUNDARY_READY
    subobj-control: exit 162
    carve-control-buggy: exit 0
    carve-control-fixed: exit 0

## Cases: {'NOT CAUGHT': 12}

    00_85407c7e63_mpegvideo_edge_emu_u_block_one_row_short
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED U's edge emulation wrote 9 + field_based rows into a 9-row block, so its tenth row was V's first, which V's emulation then overwrote
    01_699341d647_apedec_array_0000_writes_64_into_channel_1
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED decode_array_0000 wrote 64 outputs into a 40-sample channel, so 24 of them landed in decoded[1]
    02_cd7524fdd1_apedec_filter_3800_reads_past_short_channel
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED the delay line read 128 samples from a 72-sample channel, 56 of them from decoded[1]
    03_55937bb4a7_alsdec_ra_block_runs_into_next_channel_prefix
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED the RA loop reconstructed 24 samples of a 16-sample block, so it decremented the first of channel 1's carry-over samples
    04_cd09284924_alsdec_mcc_chan_data_one_slot_per_channel
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED channel 0's dependency list ran one slot past its one-slot carve, and channel 1's entry then replaced its terminator
    05_9d3032b960_alsdec_opt_order_past_max_order
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED opt_order 31 wrote 11 coefficients past channel 0's 20-word slice, into channel 1's
    06_2d0bea4719_vp9_uv_nnz_ctx_sized_for_420
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED the 4:4:4 clear wrote 16 bytes per superblock column into an 8-byte above_uv_nnz_ctx[0], running into above_uv_nnz_ctx[1]
    07_2563a33856_vp9_intra_pred_carved_for_old_bpp
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED a thread at 10-bit backed up 256 luma bytes into an intra_pred_data[0] carved for 8-bit, 128 bytes, running into intra_pred_data[1]
    08_b5ff61695f_swscale_v_line_at_pre_doubling_stride
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED U's 256 bytes of int32 samples ran past V's offset of 208 bytes, and V then overwrote U's tail
    09_043bcdcdb0_svq1enc_inter_block_runs_into_source
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED the 16x16 inter prediction at temp + 16 with a 16-byte stride ended past temp's 16 rows, overwriting src's first row
    10_d2213b6493_rv34_b_block_carved_at_old_linesize
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED after the linesize doubled, direction 0's prediction ran from tmp_b_block_y[0] into tmp_b_block_y[1], which direction 1 then overwrote
    11_68226ed9ec_vorbis_type1_residue_end_spans_channels
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED a type-1 residue ending at 2 * vlen added channel 0's codevectors into channel 1's slice of channel_residues
