# ffmpeg/carved-repros, arm `cheribsd-revocation`, all 13 cases -- 2026-10-10

Pre-registration: 26cdee4e4d8d (case 12), bbc7db0bd3d5 (P2). Runtime revocation: on (guest default preserved).
Platform sha256: qemu 16135483052dfdd6, firmware f0e1fe57b0f85075, kernel 8ab453f46dc76cf2, libc fdce2289224bb519, image 0cb16209c16c5edf

Result LINES only; the boot capture is contaminated by construction.

## Programs: 30 of 30 passed their oracle

    PASS cheribsd-abi  exit=0 (expected 0)  program bc4faad5503df79c
        CHERI_ABI pointer_bytes=16 runtime_revocation=1
    PASS cheribsd-bounds  exit=162 (expected 162)  program bc4faad5503df79c
        CHERI_ABI pointer_bytes=16 runtime_revocation=1
        CHERI_BOUNDARY_READY
    PASS carve-control-fixed  exit=0 (expected 0)  program 6dbf8e6d71704bad
        carve control off=0 len=16 bounds=allocation
        block=64 region=16 touched=15 extent=0 noted=1 crossed=0 contained=1 damage=0
        VERDICT FIXED control: the carve's last byte
    PASS carve-control-buggy  exit=0 (expected 0)  program 3ea1c35ee8556454 target 6dbf8e6d71704bad
        SUPERVISE expect ffc_write_probe_u8 0x10229a
        carve control off=0 len=16 bounds=allocation
        block=64 region=16 touched=16 extent=1 noted=1 crossed=1 contained=1 damage=0
        VERDICT DEFECT-REPRODUCED control: one byte past a 16-byte carve, inside its 64-byte block
        SUPERVISE exit status=0
    PASS ffc-00-fixed  exit=0 (expected 0)  program b5cee48abc8d2dac
        carve ubuf off=1152 len=320 bounds=allocation
        carve vbuf off=1472 len=320 bounds=allocation
        block=35840 region=320 touched=288 extent=0 noted=1 crossed=0 contained=1 damage=0
        VERDICT FIXED the fix carves V ten U-rows on, so all ten U rows stay in U
    PASS ffc-00-buggy  exit=0 (expected 0)  program 3ea1c35ee8556454 target b5cee48abc8d2dac
        SUPERVISE expect ffc_write_probe_u8 0x1024ca
        carve ubuf off=1152 len=288 bounds=allocation
        carve vbuf off=1440 len=320 bounds=allocation
        block=34816 region=288 touched=288 extent=9 noted=1 crossed=1 contained=1 damage=1
        VERDICT DEFECT-REPRODUCED U's edge emulation wrote 9 + field_based rows into a 9-row block, so its tenth row was V's first, which V's emulation then overwrote
        SUPERVISE exit status=0
    PASS ffc-01-fixed  exit=0 (expected 0)  program a3385c0fae293071
        carve decoded[0] off=0 len=160 bounds=allocation
        carve decoded[1] off=160 len=160 bounds=allocation
        block=320 region=160 touched=156 extent=0 noted=1 crossed=0 contained=1 damage=0
        VERDICT FIXED the fix stops the loop at blockstodecode, the channel's own length
    PASS ffc-01-buggy  exit=0 (expected 0)  program 7867743c6d8dc0b6 target a3385c0fae293071
        SUPERVISE expect ffc_write_probe_u32 0x102466
        carve decoded[0] off=0 len=160 bounds=allocation
        carve decoded[1] off=160 len=160 bounds=allocation
        block=320 region=160 touched=160 extent=96 noted=1 crossed=1 contained=1 damage=0
        VERDICT DEFECT-REPRODUCED decode_array_0000 wrote 64 outputs into a 40-sample channel, so 24 of them landed in decoded[1]
        SUPERVISE exit status=0
    PASS ffc-02-fixed  exit=0 (expected 0)  program 1a3ca3c33eb5365d
        carve decoded[0] off=0 len=288 bounds=allocation
        carve decoded[1] off=288 len=288 bounds=allocation
        block=0 region=0 touched=0 extent=0 noted=0 crossed=0 contained=0 damage=0
        VERDICT FIXED the fix returns before the delay line when order >= length
    PASS ffc-02-buggy  exit=0 (expected 0)  program af46d12341ca6116 target 1a3ca3c33eb5365d
        SUPERVISE expect ffc_read_probe_u32 0x102422
        carve decoded[0] off=0 len=288 bounds=allocation
        carve decoded[1] off=288 len=288 bounds=allocation
        block=576 region=288 touched=288 extent=224 noted=1 crossed=1 contained=1 damage=0
        VERDICT DEFECT-REPRODUCED the delay line read 128 samples from a 72-sample channel, 56 of them from decoded[1]
        SUPERVISE exit status=0
    PASS ffc-03-fixed  exit=0 (expected 0)  program 8aece675440225af
        carve channel[0] off=0 len=160 bounds=allocation
        carve channel[1] off=160 len=160 bounds=allocation
        block=320 region=160 touched=156 extent=0 noted=1 crossed=0 contained=1 damage=0
        VERDICT FIXED the fix stops the loop at block_length
    PASS ffc-03-buggy  exit=0 (expected 0)  program af46d12341ca6116 target 8aece675440225af
        SUPERVISE expect ffc_read_probe_u32 0x102558
        carve channel[0] off=0 len=160 bounds=allocation
        carve channel[1] off=160 len=160 bounds=allocation
        block=320 region=160 touched=160 extent=32 noted=1 crossed=1 contained=1 damage=1
        VERDICT DEFECT-REPRODUCED the RA loop reconstructed 24 samples of a 16-sample block, so it decremented the first of channel 1's carry-over samples
        SUPERVISE exit status=0
    PASS ffc-04-fixed  exit=0 (expected 0)  program 14230ada1ca95a1b
        carve chan_data[0] off=0 len=132 bounds=allocation
        carve chan_data[1] off=132 len=132 bounds=allocation
        carve chan_data[2] off=264 len=132 bounds=allocation
        block=396 region=132 touched=44 extent=0 noted=1 crossed=0 contained=1 damage=0
        VERDICT FIXED the fix gives each channel num_buffers slots, so the terminator stays in channel 0's own carve
    PASS ffc-04-buggy  exit=0 (expected 0)  program 7867743c6d8dc0b6 target 14230ada1ca95a1b
        SUPERVISE expect ffc_write_probe_u32 0x102626
        carve chan_data[0] off=0 len=44 bounds=allocation
        carve chan_data[1] off=44 len=44 bounds=allocation
        carve chan_data[2] off=88 len=44 bounds=allocation
        block=132 region=44 touched=44 extent=4 noted=1 crossed=1 contained=1 damage=1
        VERDICT DEFECT-REPRODUCED channel 0's dependency list ran one slot past its one-slot carve, and channel 1's entry then replaced its terminator
        SUPERVISE exit status=0
    PASS ffc-05-fixed  exit=0 (expected 0)  program e9995b6a39d57d97
        carve quant_cof[0] off=0 len=80 bounds=allocation
        carve quant_cof[1] off=80 len=80 bounds=allocation
        block=0 region=0 touched=0 extent=0 noted=0 crossed=0 contained=0 damage=0
        VERDICT FIXED the fix refuses a block whose opt_order exceeds max_order
    PASS ffc-05-buggy  exit=0 (expected 0)  program 7867743c6d8dc0b6 target e9995b6a39d57d97
        SUPERVISE expect ffc_write_probe_u32 0x1025c2
        carve quant_cof[0] off=0 len=80 bounds=allocation
        carve quant_cof[1] off=80 len=80 bounds=allocation
        block=160 region=80 touched=80 extent=44 noted=1 crossed=1 contained=1 damage=1
        VERDICT DEFECT-REPRODUCED opt_order 31 wrote 11 coefficients past channel 0's 20-word slice, into channel 1's
        SUPERVISE exit status=0
    PASS ffc-06-fixed  exit=0 (expected 0)  program 147163700c9621bd
        carve intra_pred_data[0] off=0 len=128 bounds=allocation
        carve intra_pred_data[1] off=128 len=128 bounds=allocation
        carve intra_pred_data[2] off=256 len=128 bounds=allocation
        carve above_y_nnz_ctx off=384 len=32 bounds=allocation
        carve above_mode_ctx off=416 len=32 bounds=allocation
        carve above_mv_ctx off=448 len=256 bounds=allocation
        carve above_uv_nnz_ctx[0] off=704 len=32 bounds=allocation
        carve above_uv_nnz_ctx[1] off=736 len=32 bounds=allocation
        carve above_partition_ctx off=768 len=16 bounds=allocation
        carve above_skip_ctx off=784 len=16 bounds=allocation
        carve above_txfm_ctx off=800 len=16 bounds=allocation
        carve above_segpred_ctx off=816 len=16 bounds=allocation
        carve above_intra_ctx off=832 len=16 bounds=allocation
        carve above_comp_ctx off=848 len=16 bounds=allocation
        carve above_ref_ctx off=864 len=16 bounds=allocation
        carve above_filter_ctx off=880 len=16 bounds=allocation
        carve lflvl off=896 len=384 bounds=allocation
        block=1280 region=32 touched=31 extent=0 noted=1 crossed=0 contained=1 damage=0
        VERDICT FIXED the fix carves each uv context at 16 bytes per column
    PASS ffc-06-buggy  exit=0 (expected 0)  program 3ea1c35ee8556454 target 147163700c9621bd
        SUPERVISE expect ffc_write_probe_u8 0x102d9c
        carve intra_pred_data[0] off=0 len=128 bounds=allocation
        carve intra_pred_data[1] off=128 len=64 bounds=allocation
        carve intra_pred_data[2] off=192 len=64 bounds=allocation
        carve above_y_nnz_ctx off=256 len=32 bounds=allocation
        carve above_mode_ctx off=288 len=32 bounds=allocation
        carve above_mv_ctx off=320 len=256 bounds=allocation
        carve above_partition_ctx off=576 len=16 bounds=allocation
        carve above_skip_ctx off=592 len=16 bounds=allocation
        carve above_txfm_ctx off=608 len=16 bounds=allocation
        carve above_uv_nnz_ctx[0] off=624 len=16 bounds=allocation
        carve above_uv_nnz_ctx[1] off=640 len=16 bounds=allocation
        carve above_segpred_ctx off=656 len=16 bounds=allocation
        carve above_intra_ctx off=672 len=16 bounds=allocation
        carve above_comp_ctx off=688 len=16 bounds=allocation
        carve above_ref_ctx off=704 len=16 bounds=allocation
        carve above_filter_ctx off=720 len=16 bounds=allocation
        carve lflvl off=736 len=384 bounds=allocation
        block=1120 region=16 touched=16 extent=16 noted=1 crossed=1 contained=1 damage=0
        VERDICT DEFECT-REPRODUCED the 4:4:4 clear wrote 16 bytes per superblock column into an 8-byte above_uv_nnz_ctx[0], running into above_uv_nnz_ctx[1]
        SUPERVISE exit status=0
    PASS ffc-07-fixed  exit=0 (expected 0)  program 304f63ab2ae9a28f
        carve intra_pred_data[0] off=0 len=256 bounds=allocation
        carve intra_pred_data[1] off=256 len=256 bounds=allocation
        carve intra_pred_data[2] off=512 len=256 bounds=allocation
        carve above_y_nnz_ctx off=768 len=32 bounds=allocation
        carve above_mode_ctx off=800 len=32 bounds=allocation
        carve above_mv_ctx off=832 len=256 bounds=allocation
        carve above_uv_nnz_ctx[0] off=1088 len=32 bounds=allocation
        carve above_uv_nnz_ctx[1] off=1120 len=32 bounds=allocation
        carve above_partition_ctx off=1152 len=16 bounds=allocation
        carve above_skip_ctx off=1168 len=16 bounds=allocation
        carve above_txfm_ctx off=1184 len=16 bounds=allocation
        carve above_segpred_ctx off=1200 len=16 bounds=allocation
        carve above_intra_ctx off=1216 len=16 bounds=allocation
        carve above_comp_ctx off=1232 len=16 bounds=allocation
        carve above_ref_ctx off=1248 len=16 bounds=allocation
        carve above_filter_ctx off=1264 len=16 bounds=allocation
        carve lflvl off=1280 len=384 bounds=allocation
        block=1664 region=256 touched=255 extent=0 noted=1 crossed=0 contained=1 damage=0
        VERDICT FIXED the fix re-carves on a bit-depth change, so [0] is 256 bytes
    PASS ffc-07-buggy  exit=0 (expected 0)  program 3ea1c35ee8556454 target 304f63ab2ae9a28f
        SUPERVISE expect ffc_write_probe_u8 0x102a60
        carve intra_pred_data[0] off=0 len=128 bounds=allocation
        carve intra_pred_data[1] off=128 len=128 bounds=allocation
        carve intra_pred_data[2] off=256 len=128 bounds=allocation
        carve above_y_nnz_ctx off=384 len=32 bounds=allocation
        carve above_mode_ctx off=416 len=32 bounds=allocation
        carve above_mv_ctx off=448 len=256 bounds=allocation
        carve above_uv_nnz_ctx[0] off=704 len=32 bounds=allocation
        carve above_uv_nnz_ctx[1] off=736 len=32 bounds=allocation
        carve above_partition_ctx off=768 len=16 bounds=allocation
        carve above_skip_ctx off=784 len=16 bounds=allocation
        carve above_txfm_ctx off=800 len=16 bounds=allocation
        carve above_segpred_ctx off=816 len=16 bounds=allocation
        carve above_intra_ctx off=832 len=16 bounds=allocation
        carve above_comp_ctx off=848 len=16 bounds=allocation
        carve above_ref_ctx off=864 len=16 bounds=allocation
        carve above_filter_ctx off=880 len=16 bounds=allocation
        carve lflvl off=896 len=384 bounds=allocation
        block=1280 region=128 touched=128 extent=128 noted=1 crossed=1 contained=1 damage=1
        VERDICT DEFECT-REPRODUCED a thread at 10-bit backed up 256 luma bytes into an intra_pred_data[0] carved for 8-bit, 128 bytes, running into intra_pred_data[1]
        SUPERVISE exit status=0
    PASS ffc-08-fixed  exit=0 (expected 0)  program d5ba5d152212f262
        carve chrUPixBuf off=0 len=416 bounds=allocation
        carve chrVPixBuf off=416 len=416 bounds=allocation
        block=833 region=416 touched=252 extent=0 noted=1 crossed=0 contained=1 damage=0
        VERDICT FIXED the fix places V half a doubled stride on, 416 bytes, past U's 256
    PASS ffc-08-buggy  exit=0 (expected 0)  program 7867743c6d8dc0b6 target d5ba5d152212f262
        SUPERVISE expect ffc_write_probe_u32 0x102546
        carve chrUPixBuf off=0 len=208 bounds=allocation
        carve chrVPixBuf off=208 len=416 bounds=allocation
        block=833 region=208 touched=208 extent=48 noted=1 crossed=1 contained=1 damage=1
        VERDICT DEFECT-REPRODUCED U's 256 bytes of int32 samples ran past V's offset of 208 bytes, and V then overwrote U's tail
        SUPERVISE exit status=0
    PASS ffc-09-fixed  exit=0 (expected 0)  program afcbb16eb9891064
        carve temp off=0 len=512 bounds=allocation
        carve src off=512 len=256 bounds=allocation
        block=1536 region=512 touched=511 extent=0 noted=1 crossed=0 contained=1 damage=0
        VERDICT FIXED the fix predicts at temp + 16 * stride inside a 32-row temp, and src starts after it
    PASS ffc-09-buggy  exit=0 (expected 0)  program 3ea1c35ee8556454 target afcbb16eb9891064
        SUPERVISE expect ffc_write_probe_u8 0x10258e
        carve temp off=0 len=256 bounds=allocation
        carve src off=256 len=256 bounds=allocation
        block=1024 region=256 touched=256 extent=16 noted=1 crossed=1 contained=1 damage=1
        VERDICT DEFECT-REPRODUCED the 16x16 inter prediction at temp + 16 with a 16-byte stride ended past temp's 16 rows, overwriting src's first row
        SUPERVISE exit status=0
    PASS ffc-10-fixed  exit=0 (expected 0)  program ef6c0ef55b1322f9
        carve tmp_b_block_y[0] off=0 len=2048 bounds=allocation
        carve tmp_b_block_y[1] off=2048 len=2048 bounds=allocation
        carve tmp_b_block_uv off=4096 len=2048 bounds=allocation
        block=6144 region=2048 touched=1935 extent=0 noted=1 crossed=0 contained=1 damage=0
        VERDICT FIXED the fix re-carves the block at the new linesize
    PASS ffc-10-buggy  exit=0 (expected 0)  program 3ea1c35ee8556454 target ef6c0ef55b1322f9
        SUPERVISE expect ffc_write_probe_u8 0x1025f2
        carve tmp_b_block_y[0] off=0 len=1024 bounds=allocation
        carve tmp_b_block_y[1] off=1024 len=1024 bounds=allocation
        carve tmp_b_block_uv off=2048 len=1024 bounds=allocation
        block=3072 region=1024 touched=1024 extent=912 noted=1 crossed=1 contained=1 damage=1
        VERDICT DEFECT-REPRODUCED after the linesize doubled, direction 0's prediction ran from tmp_b_block_y[0] into tmp_b_block_y[1], which direction 1 then overwrote
        SUPERVISE exit status=0
    PASS ffc-11-fixed  exit=0 (expected 0)  program c7b46022f93dfe31
        carve channel_residues[0] off=0 len=512 bounds=allocation
        carve channel_residues[1] off=512 len=512 bounds=allocation
        block=0 region=0 touched=0 extent=0 noted=0 crossed=0 contained=0 damage=0
        VERDICT FIXED the fix refuses a type-0/1 residue that ends past one channel's blocksize/2
    PASS ffc-11-buggy  exit=0 (expected 0)  program af46d12341ca6116 target c7b46022f93dfe31
        SUPERVISE expect ffc_read_probe_u32 0x10262a
        carve channel_residues[0] off=0 len=512 bounds=allocation
        carve channel_residues[1] off=512 len=512 bounds=allocation
        block=1024 region=512 touched=512 extent=512 noted=1 crossed=1 contained=1 damage=1
        VERDICT DEFECT-REPRODUCED a type-1 residue ending at 2 * vlen added channel 0's codevectors into channel 1's slice of channel_residues
        SUPERVISE exit status=0
    PASS ffc-12-fixed  exit=0 (expected 0)  program 726943f1eabe094a
        carve hist[t0][plane0] off=0 len=1024 bounds=allocation
        carve hist[t0][plane1] off=1024 len=1024 bounds=allocation
        carve hist[t0][plane2] off=2048 len=1024 bounds=allocation
        carve hist[t1][plane0] off=3072 len=1024 bounds=allocation
        carve hist[t1][plane1] off=4096 len=1024 bounds=allocation
        carve hist[t1][plane2] off=5120 len=1024 bounds=allocation
        carve hist[t2][plane0] off=6144 len=1024 bounds=allocation
        carve hist[t2][plane1] off=7168 len=1024 bounds=allocation
        carve hist[t2][plane2] off=8192 len=1024 bounds=allocation
        carve hist[t3][plane0] off=9216 len=1024 bounds=allocation
        carve hist[t3][plane1] off=10240 len=1024 bounds=allocation
        carve hist[t3][plane2] off=11264 len=1024 bounds=allocation
        block=12288 region=1024 touched=1020 extent=0 noted=1 crossed=0 contained=1 damage=0
        VERDICT FIXED the fix shifts by bitdepth-8 and masks to a byte, keeping the index in its sub-slice
    PASS ffc-12-buggy  exit=0 (expected 0)  program af46d12341ca6116 target 726943f1eabe094a
        SUPERVISE expect ffc_read_probe_u32 0x102526
        carve hist[t0][plane0] off=0 len=1024 bounds=allocation
        carve hist[t0][plane1] off=1024 len=1024 bounds=allocation
        carve hist[t0][plane2] off=2048 len=1024 bounds=allocation
        carve hist[t1][plane0] off=3072 len=1024 bounds=allocation
        carve hist[t1][plane1] off=4096 len=1024 bounds=allocation
        carve hist[t1][plane2] off=5120 len=1024 bounds=allocation
        carve hist[t2][plane0] off=6144 len=1024 bounds=allocation
        carve hist[t2][plane1] off=7168 len=1024 bounds=allocation
        carve hist[t2][plane2] off=8192 len=1024 bounds=allocation
        carve hist[t3][plane0] off=9216 len=1024 bounds=allocation
        carve hist[t3][plane1] off=10240 len=1024 bounds=allocation
        carve hist[t3][plane2] off=11264 len=1024 bounds=allocation
        block=12288 region=1024 touched=4092 extent=3072 noted=1 crossed=1 contained=1 damage=1
        VERDICT DEFECT-REPRODUCED a 10-bit sample indexed the 256-entry plane histogram unshifted, so the bump landed 767 entries past its carved sub-slice, in thread 1's slice
        SUPERVISE exit status=0
