# ffmpeg/plain-temporal-repros on stock CheriBSD purecap, revocation ON -- 2026-10-09

Reproduce: CHERI_SDK=~/cheri/output/sdk CHERI_SYSROOT=~/cheri/rootfs-purecap \
           CHERI_IMAGE=~/cheri/output/cheribsd-riscv64-purecap.img \
             bash runners/run-cheribsd.sh <fresh-outdir>

Result LINES only; the boot capture is contaminated by construction.

## Controls -- a suite whose controls did not fire is not a reading

    cheribsd-abi:
        CHERI_ABI pointer_bytes=16 runtime_revocation=1
    cheribsd-bounds:
        CHERI_ABI pointer_bytes=16 runtime_revocation=1
        CHERI_BOUNDARY_READY
    revocation-control:
        SUPERVISE base 0x100000 /tmp/allocator-tests/revocation-control/target
        SUPERVISE expect mc_defect_read 0x101e42
        REVOCATION_CONTROL revocation=1 tag_after_sweep=0 reissued=1
        SUPERVISE fault signal=34 code=2 addr=0x101e42 pc=0x101e42
        SUPERVISE exit signalled=34

## Per-case verdicts

    PASS cheribsd-abi
    PASS cheribsd-bounds
    PASS revocation-control
    PASS fft-00-fixed
    PASS fft-00-buggy
    PASS fft-01-fixed
    PASS fft-01-buggy
    PASS fft-02-fixed
    PASS fft-02-buggy
    PASS fft-03-fixed
    PASS fft-03-buggy
    PASS fft-04-fixed
    PASS fft-04-buggy
    PASS fft-05-fixed
    PASS fft-05-buggy
    PASS fft-06-fixed
    PASS fft-06-buggy
    PASS fft-07-fixed
    PASS fft-07-buggy
    PASS fft-08-fixed
    PASS fft-08-buggy
    PASS fft-09-fixed
    PASS fft-09-buggy
    PASS fft-10-fixed
    PASS fft-10-buggy
    PASS fft-11-fixed
    PASS fft-11-buggy
    PASS fft-12-fixed
    PASS fft-12-buggy

## Each buggy arm's record

    00_716d2a47c565_ops_dispatch_interior_alias_after_free
        SUPERVISE base 0x100000 /tmp/allocator-tests/fft-00-buggy/target
        SUPERVISE expect fft_read_probe 0x102310
        case=0 arm=buggy
        bytes=48 freed=1 marker=0xaa observed=0x11 aliased=0 damage=0
        VERDICT NOT-REISSUED the released chunk was withheld; the stale pointer still names the dead object, so nothing aliased (observed=0x11)
        SUPERVISE exit status=1
    01_c98810ab47fa_hw_base_encode_list_walk_reads_freed_link
        SUPERVISE base 0x100000 /tmp/allocator-tests/fft-01-buggy/target
        SUPERVISE expect fft_read_probe 0x1022fe
        case=1 arm=buggy
        bytes=48 freed=1 marker=0xaa observed=0x11 aliased=0 damage=0
        VERDICT NOT-REISSUED the released chunk was withheld; the stale pointer still names the dead object, so nothing aliased (observed=0x11)
        SUPERVISE exit status=1
    02_43de8b328b62_lzf_write_cursor_stale_after_realloc
        SUPERVISE base 0x100000 /tmp/allocator-tests/fft-02-buggy/target
        SUPERVISE expect fft_write_probe 0x1023fa
        case=2 arm=buggy
        bytes=48 freed=1 marker=0xaa observed=0xaa aliased=0 damage=0
        VERDICT NOT-REISSUED the released chunk was withheld; the stale pointer still names the dead object, so nothing aliased (observed=0xaa)
        SUPERVISE exit status=1
    03_dc87758775e2_aviobuf_checksum_ptr_stale_after_realloc
        SUPERVISE base 0x100000 /tmp/allocator-tests/fft-03-buggy/target
        SUPERVISE expect fft_read_probe 0x10231a
        case=3 arm=buggy
        bytes=48 freed=1 marker=0xaa observed=0x11 aliased=0 damage=0
        VERDICT NOT-REISSUED the released chunk was withheld; the stale pointer still names the dead object, so nothing aliased (observed=0x11)
        SUPERVISE exit status=1
    04_4b2248594c7f_jpeg2000_packed_headers_stream_stale_base
        SUPERVISE base 0x100000 /tmp/allocator-tests/fft-04-buggy/target
        SUPERVISE expect fft_read_probe 0x102450
        case=4 arm=buggy
        bytes=48 freed=1 marker=0xaa observed=0x11 aliased=0 damage=0
        VERDICT NOT-REISSUED the released chunk was withheld; the stale pointer still names the dead object, so nothing aliased (observed=0x11)
        SUPERVISE exit status=1
    05_d6458f6a8bf1_aacdec_tag_che_map_keeps_freed_element
        SUPERVISE base 0x100000 /tmp/allocator-tests/fft-05-buggy/target
        SUPERVISE expect fft_read_probe 0x10232a
        case=5 arm=buggy
        bytes=48 freed=1 marker=0xaa observed=0x11 aliased=0 damage=0
        VERDICT NOT-REISSUED the released chunk was withheld; the stale pointer still names the dead object, so nothing aliased (observed=0x11)
        SUPERVISE exit status=1
    06_e8714f6f93d1_h264_delayed_pic_holds_interior_pointers
        SUPERVISE base 0x100000 /tmp/allocator-tests/fft-06-buggy/target
        SUPERVISE expect fft_write_probe 0x102336
        case=6 arm=buggy
        bytes=48 freed=1 marker=0xaa observed=0xaa aliased=0 damage=0
        VERDICT NOT-REISSUED the released chunk was withheld; the stale pointer still names the dead object, so nothing aliased (observed=0xaa)
        SUPERVISE exit status=1
    07_a43e9cdd442b_isom_extradata_freed_before_callee_frees_it
        SUPERVISE base 0x100000 /tmp/allocator-tests/fft-07-buggy/target
        SUPERVISE expect fft_read_probe 0x1022d2
        case=7 arm=buggy
        bytes=48 freed=1 marker=0xaa observed=0x11 aliased=0 damage=0
        VERDICT NOT-REISSUED the released chunk was withheld; the stale pointer still names the dead object, so nothing aliased (observed=0x11)
        SUPERVISE exit status=1
    08_8a4ea9644833_diracdec_realloc_on_the_wrong_field
        SUPERVISE base 0x100000 /tmp/allocator-tests/fft-08-buggy/target
        SUPERVISE expect fft_read_probe 0x10247a
        case=8 arm=buggy
        bytes=48 freed=1 marker=0xaa observed=0x11 aliased=0 damage=0
        VERDICT NOT-REISSUED the released chunk was withheld; the stale pointer still names the dead object, so nothing aliased (observed=0x11)
        SUPERVISE exit status=1
    09_265731f201f1_tx_subcontext_field_left_dangling_on_failure
        SUPERVISE base 0x100000 /tmp/allocator-tests/fft-09-buggy/target
        SUPERVISE expect fft_read_probe 0x102316
        case=9 arm=buggy
        bytes=48 freed=1 marker=0xaa observed=0x11 aliased=0 damage=0
        VERDICT NOT-REISSUED the released chunk was withheld; the stale pointer still names the dead object, so nothing aliased (observed=0x11)
        SUPERVISE exit status=1
    10_e7a65142b972_aacpsy_clears_the_local_not_the_field
        SUPERVISE base 0x100000 /tmp/allocator-tests/fft-10-buggy/target
        SUPERVISE expect fft_read_probe 0x10231e
        case=10 arm=buggy
        bytes=48 freed=1 marker=0xaa observed=0x11 aliased=0 damage=0
        VERDICT NOT-REISSUED the released chunk was withheld; the stale pointer still names the dead object, so nothing aliased (observed=0x11)
        SUPERVISE exit status=1
    11_ba28222a14ab_ratecontrol_expr_field_not_cleared
        SUPERVISE base 0x100000 /tmp/allocator-tests/fft-11-buggy/target
        SUPERVISE expect fft_read_probe 0x1022de
        case=11 arm=buggy
        bytes=48 freed=1 marker=0xaa observed=0x11 aliased=0 damage=0
        VERDICT NOT-REISSUED the released chunk was withheld; the stale pointer still names the dead object, so nothing aliased (observed=0x11)
        SUPERVISE exit status=1
    12_2e04d35c69e6_vlc_buf_freed_by_callee_then_read
        SUPERVISE base 0x100000 /tmp/allocator-tests/fft-12-buggy/target
        SUPERVISE expect fft_read_probe 0x102342
        case=12 arm=buggy
        bytes=48 freed=1 marker=0xaa observed=0x11 aliased=0 damage=0
        VERDICT NOT-REISSUED the released chunk was withheld; the stale pointer still names the dead object, so nothing aliased (observed=0x11)
        SUPERVISE exit status=1
