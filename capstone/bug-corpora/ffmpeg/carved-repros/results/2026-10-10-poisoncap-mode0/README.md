# ffmpeg/carved-repros, arm `poisoncap-spatial`, all 13 cases -- 2026-10-10

Pre-registration: 26cdee4e4d8d (case 12), bbc7db0bd3d5 (P2). Runtime revocation: off (guest default preserved).
Platform sha256: qemu 5a9a8ef9cade9d92, firmware f0e1fe57b0f85075, kernel 7b4b2b5873f08c69, libc 8071c62d364f388f, image c8df9e17594b2614

Result LINES only; the boot capture is contaminated by construction.

## Programs: 30 of 30 passed their oracle

    PASS cheribsd-abi  exit=0 (expected 0)  program 3a40ff30f9e8b7fb
        CHERI_ABI pointer_bytes=16 runtime_revocation=0
    PASS cheribsd-bounds  exit=162 (expected 162)  program 3a40ff30f9e8b7fb
        CHERI_ABI pointer_bytes=16 runtime_revocation=0
        CHERI_BOUNDARY_READY
    PASS carve-control-fixed  exit=0 (expected 0)  program bbebceeb2f55d842
        carve control off=0 len=16 bounds=allocation
        block=64 region=16 touched=15 extent=0 noted=1 crossed=0 contained=1 damage=0
        VERDICT FIXED control: the carve's last byte
    PASS carve-control-buggy  exit=0 (expected 0)  program bab22413527305ed target bbebceeb2f55d842
        SUPERVISE expect ffc_write_probe_u8 0x10221a
        carve control off=0 len=16 bounds=allocation
        block=64 region=16 touched=16 extent=1 noted=1 crossed=1 contained=1 damage=0
        VERDICT DEFECT-REPRODUCED control: one byte past a 16-byte carve, inside its 64-byte block
        SUPERVISE exit status=0
    PASS ffc-00-fixed  exit=0 (expected 0)  program 5d1ddeda5e80d4a2
        carve ubuf off=1152 len=320 bounds=allocation
        carve vbuf off=1472 len=320 bounds=allocation
        block=35840 region=320 touched=288 extent=0 noted=1 crossed=0 contained=1 damage=0
        VERDICT FIXED the fix carves V ten U-rows on, so all ten U rows stay in U
    PASS ffc-00-buggy  exit=0 (expected 0)  program bab22413527305ed target 5d1ddeda5e80d4a2
        SUPERVISE expect ffc_write_probe_u8 0x10244a
        carve ubuf off=1152 len=288 bounds=allocation
        carve vbuf off=1440 len=320 bounds=allocation
        block=34816 region=288 touched=288 extent=9 noted=1 crossed=1 contained=1 damage=1
        VERDICT DEFECT-REPRODUCED U's edge emulation wrote 9 + field_based rows into a 9-row block, so its tenth row was V's first, which V's emulation then overwrote
        SUPERVISE exit status=0
    PASS ffc-01-fixed  exit=0 (expected 0)  program 44f6258bdb45c7d8
        carve decoded[0] off=0 len=160 bounds=allocation
        carve decoded[1] off=160 len=160 bounds=allocation
        block=320 region=160 touched=156 extent=0 noted=1 crossed=0 contained=1 damage=0
        VERDICT FIXED the fix stops the loop at blockstodecode, the channel's own length
    PASS ffc-01-buggy  exit=0 (expected 0)  program 9619c037da873b5b target 44f6258bdb45c7d8
        SUPERVISE expect ffc_write_probe_u32 0x1023e6
        carve decoded[0] off=0 len=160 bounds=allocation
        carve decoded[1] off=160 len=160 bounds=allocation
        block=320 region=160 touched=160 extent=96 noted=1 crossed=1 contained=1 damage=0
        VERDICT DEFECT-REPRODUCED decode_array_0000 wrote 64 outputs into a 40-sample channel, so 24 of them landed in decoded[1]
        SUPERVISE exit status=0
    PASS ffc-02-fixed  exit=0 (expected 0)  program 75bda2b3b1dd202b
        carve decoded[0] off=0 len=288 bounds=allocation
        carve decoded[1] off=288 len=288 bounds=allocation
        block=0 region=0 touched=0 extent=0 noted=0 crossed=0 contained=0 damage=0
        VERDICT FIXED the fix returns before the delay line when order >= length
    PASS ffc-02-buggy  exit=0 (expected 0)  program d31e0d1c373cec76 target 75bda2b3b1dd202b
        SUPERVISE expect ffc_read_probe_u32 0x1023a2
        carve decoded[0] off=0 len=288 bounds=allocation
        carve decoded[1] off=288 len=288 bounds=allocation
        block=576 region=288 touched=288 extent=224 noted=1 crossed=1 contained=1 damage=0
        VERDICT DEFECT-REPRODUCED the delay line read 128 samples from a 72-sample channel, 56 of them from decoded[1]
        SUPERVISE exit status=0
    PASS ffc-03-fixed  exit=0 (expected 0)  program 914980c3144a23ca
        carve channel[0] off=0 len=160 bounds=allocation
        carve channel[1] off=160 len=160 bounds=allocation
        block=320 region=160 touched=156 extent=0 noted=1 crossed=0 contained=1 damage=0
        VERDICT FIXED the fix stops the loop at block_length
    PASS ffc-03-buggy  exit=0 (expected 0)  program d31e0d1c373cec76 target 914980c3144a23ca
        SUPERVISE expect ffc_read_probe_u32 0x1024d8
        carve channel[0] off=0 len=160 bounds=allocation
        carve channel[1] off=160 len=160 bounds=allocation
        block=320 region=160 touched=160 extent=32 noted=1 crossed=1 contained=1 damage=1
        VERDICT DEFECT-REPRODUCED the RA loop reconstructed 24 samples of a 16-sample block, so it decremented the first of channel 1's carry-over samples
        SUPERVISE exit status=0
    PASS ffc-04-fixed  exit=0 (expected 0)  program ed9d11d22a6f097a
        carve chan_data[0] off=0 len=132 bounds=allocation
        carve chan_data[1] off=132 len=132 bounds=allocation
        carve chan_data[2] off=264 len=132 bounds=allocation
        block=396 region=132 touched=44 extent=0 noted=1 crossed=0 contained=1 damage=0
        VERDICT FIXED the fix gives each channel num_buffers slots, so the terminator stays in channel 0's own carve
    PASS ffc-04-buggy  exit=0 (expected 0)  program 9619c037da873b5b target ed9d11d22a6f097a
        SUPERVISE expect ffc_write_probe_u32 0x1025a6
        carve chan_data[0] off=0 len=44 bounds=allocation
        carve chan_data[1] off=44 len=44 bounds=allocation
        carve chan_data[2] off=88 len=44 bounds=allocation
        block=132 region=44 touched=44 extent=4 noted=1 crossed=1 contained=1 damage=1
        VERDICT DEFECT-REPRODUCED channel 0's dependency list ran one slot past its one-slot carve, and channel 1's entry then replaced its terminator
        SUPERVISE exit status=0
    PASS ffc-05-fixed  exit=0 (expected 0)  program 75023fe755391be5
        carve quant_cof[0] off=0 len=80 bounds=allocation
        carve quant_cof[1] off=80 len=80 bounds=allocation
        block=0 region=0 touched=0 extent=0 noted=0 crossed=0 contained=0 damage=0
        VERDICT FIXED the fix refuses a block whose opt_order exceeds max_order
    PASS ffc-05-buggy  exit=0 (expected 0)  program 9619c037da873b5b target 75023fe755391be5
        SUPERVISE expect ffc_write_probe_u32 0x102542
        carve quant_cof[0] off=0 len=80 bounds=allocation
        carve quant_cof[1] off=80 len=80 bounds=allocation
        block=160 region=80 touched=80 extent=44 noted=1 crossed=1 contained=1 damage=1
        VERDICT DEFECT-REPRODUCED opt_order 31 wrote 11 coefficients past channel 0's 20-word slice, into channel 1's
        SUPERVISE exit status=0
    PASS ffc-06-fixed  exit=0 (expected 0)  program 084b487d4c1dbe03
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
    PASS ffc-06-buggy  exit=0 (expected 0)  program bab22413527305ed target 084b487d4c1dbe03
        SUPERVISE expect ffc_write_probe_u8 0x102d1c
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
    PASS ffc-07-fixed  exit=0 (expected 0)  program 9384ede8ba8fcab4
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
    PASS ffc-07-buggy  exit=0 (expected 0)  program bab22413527305ed target 9384ede8ba8fcab4
        SUPERVISE expect ffc_write_probe_u8 0x1029e0
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
    PASS ffc-08-fixed  exit=0 (expected 0)  program 2bb39c7b92de54f9
        carve chrUPixBuf off=0 len=416 bounds=allocation
        carve chrVPixBuf off=416 len=416 bounds=allocation
        block=833 region=416 touched=252 extent=0 noted=1 crossed=0 contained=1 damage=0
        VERDICT FIXED the fix places V half a doubled stride on, 416 bytes, past U's 256
    PASS ffc-08-buggy  exit=0 (expected 0)  program 9619c037da873b5b target 2bb39c7b92de54f9
        SUPERVISE expect ffc_write_probe_u32 0x1024c6
        carve chrUPixBuf off=0 len=208 bounds=allocation
        carve chrVPixBuf off=208 len=416 bounds=allocation
        block=833 region=208 touched=208 extent=48 noted=1 crossed=1 contained=1 damage=1
        VERDICT DEFECT-REPRODUCED U's 256 bytes of int32 samples ran past V's offset of 208 bytes, and V then overwrote U's tail
        SUPERVISE exit status=0
    PASS ffc-09-fixed  exit=0 (expected 0)  program 5a595c5765581aa3
        carve temp off=0 len=512 bounds=allocation
        carve src off=512 len=256 bounds=allocation
        block=1536 region=512 touched=511 extent=0 noted=1 crossed=0 contained=1 damage=0
        VERDICT FIXED the fix predicts at temp + 16 * stride inside a 32-row temp, and src starts after it
    PASS ffc-09-buggy  exit=0 (expected 0)  program bab22413527305ed target 5a595c5765581aa3
        SUPERVISE expect ffc_write_probe_u8 0x10250e
        carve temp off=0 len=256 bounds=allocation
        carve src off=256 len=256 bounds=allocation
        block=1024 region=256 touched=256 extent=16 noted=1 crossed=1 contained=1 damage=1
        VERDICT DEFECT-REPRODUCED the 16x16 inter prediction at temp + 16 with a 16-byte stride ended past temp's 16 rows, overwriting src's first row
        SUPERVISE exit status=0
    PASS ffc-10-fixed  exit=0 (expected 0)  program 22e5885dc09f6c77
        carve tmp_b_block_y[0] off=0 len=2048 bounds=allocation
        carve tmp_b_block_y[1] off=2048 len=2048 bounds=allocation
        carve tmp_b_block_uv off=4096 len=2048 bounds=allocation
        block=6144 region=2048 touched=1935 extent=0 noted=1 crossed=0 contained=1 damage=0
        VERDICT FIXED the fix re-carves the block at the new linesize
    PASS ffc-10-buggy  exit=0 (expected 0)  program bab22413527305ed target 22e5885dc09f6c77
        SUPERVISE expect ffc_write_probe_u8 0x102572
        carve tmp_b_block_y[0] off=0 len=1024 bounds=allocation
        carve tmp_b_block_y[1] off=1024 len=1024 bounds=allocation
        carve tmp_b_block_uv off=2048 len=1024 bounds=allocation
        block=3072 region=1024 touched=1024 extent=912 noted=1 crossed=1 contained=1 damage=1
        VERDICT DEFECT-REPRODUCED after the linesize doubled, direction 0's prediction ran from tmp_b_block_y[0] into tmp_b_block_y[1], which direction 1 then overwrote
        SUPERVISE exit status=0
    PASS ffc-11-fixed  exit=0 (expected 0)  program af61cb6062b69d62
        carve channel_residues[0] off=0 len=512 bounds=allocation
        carve channel_residues[1] off=512 len=512 bounds=allocation
        block=0 region=0 touched=0 extent=0 noted=0 crossed=0 contained=0 damage=0
        VERDICT FIXED the fix refuses a type-0/1 residue that ends past one channel's blocksize/2
    PASS ffc-11-buggy  exit=0 (expected 0)  program d31e0d1c373cec76 target af61cb6062b69d62
        SUPERVISE expect ffc_read_probe_u32 0x1025aa
        carve channel_residues[0] off=0 len=512 bounds=allocation
        carve channel_residues[1] off=512 len=512 bounds=allocation
        block=1024 region=512 touched=512 extent=512 noted=1 crossed=1 contained=1 damage=1
        VERDICT DEFECT-REPRODUCED a type-1 residue ending at 2 * vlen added channel 0's codevectors into channel 1's slice of channel_residues
        SUPERVISE exit status=0
    PASS ffc-12-fixed  exit=0 (expected 0)  program 9015a87472e1ce4e
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
    PASS ffc-12-buggy  exit=0 (expected 0)  program d31e0d1c373cec76 target 9015a87472e1ce4e
        SUPERVISE expect ffc_read_probe_u32 0x1024a6
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
