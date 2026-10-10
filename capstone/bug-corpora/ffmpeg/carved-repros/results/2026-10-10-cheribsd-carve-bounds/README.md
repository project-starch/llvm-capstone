# ffmpeg/carved-repros, arm `cheribsd-carve-bounds`, all 13 cases -- 2026-10-10

Pre-registration: 26cdee4e4d8d (case 12), bbc7db0bd3d5 (P2). Runtime revocation: on (guest default preserved).
Platform sha256: qemu 16135483052dfdd6, firmware f0e1fe57b0f85075, kernel 8ab453f46dc76cf2, libc fdce2289224bb519, image 0cb16209c16c5edf

Result LINES only; the boot capture is contaminated by construction.

## Programs: 30 of 30 passed their oracle

    PASS cheribsd-abi  exit=0 (expected 0)  program bc4faad5503df79c
        CHERI_ABI pointer_bytes=16 runtime_revocation=1
    PASS cheribsd-bounds  exit=162 (expected 162)  program bc4faad5503df79c
        CHERI_ABI pointer_bytes=16 runtime_revocation=1
        CHERI_BOUNDARY_READY
    PASS carve-control-fixed  exit=0 (expected 0)  program 1178ab42bf19ab20
        carve control off=0 len=16 bounds=16 base_eq=1
        block=64 region=16 touched=15 extent=0 noted=1 crossed=0 contained=1 damage=0
        VERDICT FIXED control: the carve's last byte
    PASS carve-control-buggy  exit=162 (expected 162)  program 3ea1c35ee8556454 target 1178ab42bf19ab20
        SUPERVISE expect ffc_write_probe_u8 0x1022a2
        SUPERVISE fault signal=34 code=1 addr=0x1022ba pc=0x1022ba
        SUPERVISE exit signalled=34
    PASS ffc-00-fixed  exit=0 (expected 0)  program 040bac1385fbbedb
        carve ubuf off=1152 len=320 bounds=320 base_eq=1
        carve vbuf off=1472 len=320 bounds=320 base_eq=1
        block=35840 region=320 touched=288 extent=0 noted=1 crossed=0 contained=1 damage=0
        VERDICT FIXED the fix carves V ten U-rows on, so all ten U rows stay in U
    PASS ffc-00-buggy  exit=162 (expected 162)  program 3ea1c35ee8556454 target 040bac1385fbbedb
        SUPERVISE expect ffc_write_probe_u8 0x1024ca
        SUPERVISE fault signal=34 code=1 addr=0x1024e2 pc=0x1024e2
        SUPERVISE exit signalled=34
    PASS ffc-01-fixed  exit=0 (expected 0)  program a955f693c1c8d007
        carve decoded[0] off=0 len=160 bounds=160 base_eq=1
        carve decoded[1] off=160 len=160 bounds=160 base_eq=1
        block=320 region=160 touched=156 extent=0 noted=1 crossed=0 contained=1 damage=0
        VERDICT FIXED the fix stops the loop at blockstodecode, the channel's own length
    PASS ffc-01-buggy  exit=162 (expected 162)  program 7867743c6d8dc0b6 target a955f693c1c8d007
        SUPERVISE expect ffc_write_probe_u32 0x10246e
        SUPERVISE fault signal=34 code=1 addr=0x102486 pc=0x102486
        SUPERVISE exit signalled=34
    PASS ffc-02-fixed  exit=0 (expected 0)  program 7c987f888762f1f0
        carve decoded[0] off=0 len=288 bounds=288 base_eq=1
        carve decoded[1] off=288 len=288 bounds=288 base_eq=1
        block=0 region=0 touched=0 extent=0 noted=0 crossed=0 contained=0 damage=0
        VERDICT FIXED the fix returns before the delay line when order >= length
    PASS ffc-02-buggy  exit=162 (expected 162)  program af46d12341ca6116 target 7c987f888762f1f0
        SUPERVISE expect ffc_read_probe_u32 0x10242a
        SUPERVISE fault signal=34 code=1 addr=0x10243a pc=0x10243a
        SUPERVISE exit signalled=34
    PASS ffc-03-fixed  exit=0 (expected 0)  program 8b3404219dc21229
        carve channel[0] off=0 len=160 bounds=160 base_eq=1
        carve channel[1] off=160 len=160 bounds=160 base_eq=1
        block=320 region=160 touched=156 extent=0 noted=1 crossed=0 contained=1 damage=0
        VERDICT FIXED the fix stops the loop at block_length
    PASS ffc-03-buggy  exit=162 (expected 162)  program af46d12341ca6116 target 8b3404219dc21229
        SUPERVISE expect ffc_read_probe_u32 0x102560
        SUPERVISE fault signal=34 code=1 addr=0x102570 pc=0x102570
        SUPERVISE exit signalled=34
    PASS ffc-04-fixed  exit=0 (expected 0)  program b556131cf167fa86
        carve chan_data[0] off=0 len=132 bounds=132 base_eq=1
        carve chan_data[1] off=132 len=132 bounds=132 base_eq=1
        carve chan_data[2] off=264 len=132 bounds=132 base_eq=1
        block=396 region=132 touched=44 extent=0 noted=1 crossed=0 contained=1 damage=0
        VERDICT FIXED the fix gives each channel num_buffers slots, so the terminator stays in channel 0's own carve
    PASS ffc-04-buggy  exit=162 (expected 162)  program 7867743c6d8dc0b6 target b556131cf167fa86
        SUPERVISE expect ffc_write_probe_u32 0x102626
        SUPERVISE fault signal=34 code=1 addr=0x10263e pc=0x10263e
        SUPERVISE exit signalled=34
    PASS ffc-05-fixed  exit=0 (expected 0)  program c243bcb5669c703b
        carve quant_cof[0] off=0 len=80 bounds=80 base_eq=1
        carve quant_cof[1] off=80 len=80 bounds=80 base_eq=1
        block=0 region=0 touched=0 extent=0 noted=0 crossed=0 contained=0 damage=0
        VERDICT FIXED the fix refuses a block whose opt_order exceeds max_order
    PASS ffc-05-buggy  exit=162 (expected 162)  program 7867743c6d8dc0b6 target c243bcb5669c703b
        SUPERVISE expect ffc_write_probe_u32 0x1025ca
        SUPERVISE fault signal=34 code=1 addr=0x1025e2 pc=0x1025e2
        SUPERVISE exit signalled=34
    PASS ffc-06-fixed  exit=0 (expected 0)  program d88ea1ffb2fc5c79
        carve intra_pred_data[0] off=0 len=128 bounds=128 base_eq=1
        carve intra_pred_data[1] off=128 len=128 bounds=128 base_eq=1
        carve intra_pred_data[2] off=256 len=128 bounds=128 base_eq=1
        carve above_y_nnz_ctx off=384 len=32 bounds=32 base_eq=1
        carve above_mode_ctx off=416 len=32 bounds=32 base_eq=1
        carve above_mv_ctx off=448 len=256 bounds=256 base_eq=1
        carve above_uv_nnz_ctx[0] off=704 len=32 bounds=32 base_eq=1
        carve above_uv_nnz_ctx[1] off=736 len=32 bounds=32 base_eq=1
        carve above_partition_ctx off=768 len=16 bounds=16 base_eq=1
        carve above_skip_ctx off=784 len=16 bounds=16 base_eq=1
        carve above_txfm_ctx off=800 len=16 bounds=16 base_eq=1
        carve above_segpred_ctx off=816 len=16 bounds=16 base_eq=1
        carve above_intra_ctx off=832 len=16 bounds=16 base_eq=1
        carve above_comp_ctx off=848 len=16 bounds=16 base_eq=1
        carve above_ref_ctx off=864 len=16 bounds=16 base_eq=1
        carve above_filter_ctx off=880 len=16 bounds=16 base_eq=1
        carve lflvl off=896 len=384 bounds=384 base_eq=1
        block=1280 region=32 touched=31 extent=0 noted=1 crossed=0 contained=1 damage=0
        VERDICT FIXED the fix carves each uv context at 16 bytes per column
    PASS ffc-06-buggy  exit=162 (expected 162)  program 3ea1c35ee8556454 target d88ea1ffb2fc5c79
        SUPERVISE expect ffc_write_probe_u8 0x102d9c
        SUPERVISE fault signal=34 code=1 addr=0x102db4 pc=0x102db4
        SUPERVISE exit signalled=34
    PASS ffc-07-fixed  exit=0 (expected 0)  program 29f187b0e49cbed6
        carve intra_pred_data[0] off=0 len=256 bounds=256 base_eq=1
        carve intra_pred_data[1] off=256 len=256 bounds=256 base_eq=1
        carve intra_pred_data[2] off=512 len=256 bounds=256 base_eq=1
        carve above_y_nnz_ctx off=768 len=32 bounds=32 base_eq=1
        carve above_mode_ctx off=800 len=32 bounds=32 base_eq=1
        carve above_mv_ctx off=832 len=256 bounds=256 base_eq=1
        carve above_uv_nnz_ctx[0] off=1088 len=32 bounds=32 base_eq=1
        carve above_uv_nnz_ctx[1] off=1120 len=32 bounds=32 base_eq=1
        carve above_partition_ctx off=1152 len=16 bounds=16 base_eq=1
        carve above_skip_ctx off=1168 len=16 bounds=16 base_eq=1
        carve above_txfm_ctx off=1184 len=16 bounds=16 base_eq=1
        carve above_segpred_ctx off=1200 len=16 bounds=16 base_eq=1
        carve above_intra_ctx off=1216 len=16 bounds=16 base_eq=1
        carve above_comp_ctx off=1232 len=16 bounds=16 base_eq=1
        carve above_ref_ctx off=1248 len=16 bounds=16 base_eq=1
        carve above_filter_ctx off=1264 len=16 bounds=16 base_eq=1
        carve lflvl off=1280 len=384 bounds=384 base_eq=1
        block=1664 region=256 touched=255 extent=0 noted=1 crossed=0 contained=1 damage=0
        VERDICT FIXED the fix re-carves on a bit-depth change, so [0] is 256 bytes
    PASS ffc-07-buggy  exit=162 (expected 162)  program 3ea1c35ee8556454 target 29f187b0e49cbed6
        SUPERVISE expect ffc_write_probe_u8 0x102a68
        SUPERVISE fault signal=34 code=1 addr=0x102a80 pc=0x102a80
        SUPERVISE exit signalled=34
    PASS ffc-08-fixed  exit=0 (expected 0)  program b99ebfffbf6a95b4
        carve chrUPixBuf off=0 len=416 bounds=416 base_eq=1
        carve chrVPixBuf off=416 len=416 bounds=416 base_eq=1
        block=833 region=416 touched=252 extent=0 noted=1 crossed=0 contained=1 damage=0
        VERDICT FIXED the fix places V half a doubled stride on, 416 bytes, past U's 256
    PASS ffc-08-buggy  exit=162 (expected 162)  program 7867743c6d8dc0b6 target b99ebfffbf6a95b4
        SUPERVISE expect ffc_write_probe_u32 0x10254e
        SUPERVISE fault signal=34 code=1 addr=0x102566 pc=0x102566
        SUPERVISE exit signalled=34
    PASS ffc-09-fixed  exit=0 (expected 0)  program a712bb4e3ae128be
        carve temp off=0 len=512 bounds=512 base_eq=1
        carve src off=512 len=256 bounds=256 base_eq=1
        block=1536 region=512 touched=511 extent=0 noted=1 crossed=0 contained=1 damage=0
        VERDICT FIXED the fix predicts at temp + 16 * stride inside a 32-row temp, and src starts after it
    PASS ffc-09-buggy  exit=162 (expected 162)  program 3ea1c35ee8556454 target a712bb4e3ae128be
        SUPERVISE expect ffc_write_probe_u8 0x10258e
        SUPERVISE fault signal=34 code=1 addr=0x1025a6 pc=0x1025a6
        SUPERVISE exit signalled=34
    PASS ffc-10-fixed  exit=0 (expected 0)  program d382c318050250a8
        carve tmp_b_block_y[0] off=0 len=2048 bounds=2048 base_eq=1
        carve tmp_b_block_y[1] off=2048 len=2048 bounds=2048 base_eq=1
        carve tmp_b_block_uv off=4096 len=2048 bounds=2048 base_eq=1
        block=6144 region=2048 touched=1935 extent=0 noted=1 crossed=0 contained=1 damage=0
        VERDICT FIXED the fix re-carves the block at the new linesize
    PASS ffc-10-buggy  exit=162 (expected 162)  program 3ea1c35ee8556454 target d382c318050250a8
        SUPERVISE expect ffc_write_probe_u8 0x1025fa
        SUPERVISE fault signal=34 code=1 addr=0x102612 pc=0x102612
        SUPERVISE exit signalled=34
    PASS ffc-11-fixed  exit=0 (expected 0)  program 14ed3d8bf15a8a04
        carve channel_residues[0] off=0 len=512 bounds=512 base_eq=1
        carve channel_residues[1] off=512 len=512 bounds=512 base_eq=1
        block=0 region=0 touched=0 extent=0 noted=0 crossed=0 contained=0 damage=0
        VERDICT FIXED the fix refuses a type-0/1 residue that ends past one channel's blocksize/2
    PASS ffc-11-buggy  exit=162 (expected 162)  program af46d12341ca6116 target 14ed3d8bf15a8a04
        SUPERVISE expect ffc_read_probe_u32 0x102632
        SUPERVISE fault signal=34 code=1 addr=0x102642 pc=0x102642
        SUPERVISE exit signalled=34
    PASS ffc-12-fixed  exit=0 (expected 0)  program 4ce753e1754bed43
        carve hist[t0][plane0] off=0 len=1024 bounds=1024 base_eq=1
        carve hist[t0][plane1] off=1024 len=1024 bounds=1024 base_eq=1
        carve hist[t0][plane2] off=2048 len=1024 bounds=1024 base_eq=1
        carve hist[t1][plane0] off=3072 len=1024 bounds=1024 base_eq=1
        carve hist[t1][plane1] off=4096 len=1024 bounds=1024 base_eq=1
        carve hist[t1][plane2] off=5120 len=1024 bounds=1024 base_eq=1
        carve hist[t2][plane0] off=6144 len=1024 bounds=1024 base_eq=1
        carve hist[t2][plane1] off=7168 len=1024 bounds=1024 base_eq=1
        carve hist[t2][plane2] off=8192 len=1024 bounds=1024 base_eq=1
        carve hist[t3][plane0] off=9216 len=1024 bounds=1024 base_eq=1
        carve hist[t3][plane1] off=10240 len=1024 bounds=1024 base_eq=1
        carve hist[t3][plane2] off=11264 len=1024 bounds=1024 base_eq=1
        block=12288 region=1024 touched=1020 extent=0 noted=1 crossed=0 contained=1 damage=0
        VERDICT FIXED the fix shifts by bitdepth-8 and masks to a byte, keeping the index in its sub-slice
    PASS ffc-12-buggy  exit=162 (expected 162)  program af46d12341ca6116 target 4ce753e1754bed43
        SUPERVISE expect ffc_read_probe_u32 0x10252e
        SUPERVISE fault signal=34 code=1 addr=0x10253e pc=0x10253e
        SUPERVISE exit signalled=34
