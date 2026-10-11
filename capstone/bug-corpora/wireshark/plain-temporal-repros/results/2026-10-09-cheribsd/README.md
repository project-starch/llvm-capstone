# wireshark/plain-temporal-repros on stock CheriBSD purecap, revocation ON -- 2026-10-09

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
    PASS wst-00-fixed
    PASS wst-00-buggy
    PASS wst-01-fixed
    PASS wst-01-buggy
    PASS wst-02-fixed
    PASS wst-02-buggy
    PASS wst-03-fixed
    PASS wst-03-buggy
    PASS wst-04-fixed
    PASS wst-04-buggy
    PASS wst-05-fixed
    PASS wst-05-buggy
    PASS wst-06-fixed
    PASS wst-06-buggy
    PASS wst-07-fixed
    PASS wst-07-buggy
    PASS wst-08-fixed
    PASS wst-08-buggy
    PASS wst-09-fixed
    PASS wst-09-buggy

## Each buggy arm's record

    00_f3c2e6087e7b_k12_callee_frees_its_own_argument
        SUPERVISE base 0x100000 /tmp/allocator-tests/wst-00-buggy/target
        SUPERVISE expect wst_read_probe 0x1022f6
        case=0 arm=buggy
        bytes=48 freed=1 marker=0xaa observed=0x11 aliased=0 damage=0
        VERDICT NOT-REISSUED the released chunk was withheld; the stale pointer still names the dead object, so nothing aliased (observed=0x11)
        SUPERVISE exit status=1
    01_7dcf69480de8_peak_trc_callee_frees_state_struct
        SUPERVISE base 0x100000 /tmp/allocator-tests/wst-01-buggy/target
        SUPERVISE expect wst_read_probe 0x1022f6
        case=1 arm=buggy
        bytes=48 freed=1 marker=0xaa observed=0x11 aliased=0 damage=0
        VERDICT NOT-REISSUED the released chunk was withheld; the stale pointer still names the dead object, so nothing aliased (observed=0x11)
        SUPERVISE exit status=1
    02_0fc7f3781351_wspstat_container_freed_before_its_contents
        SUPERVISE base 0x100000 /tmp/allocator-tests/wst-02-buggy/target
        SUPERVISE expect wst_read_probe 0x102326
        case=2 arm=buggy
        bytes=48 freed=1 marker=0xaa observed=0x11 aliased=0 damage=0
        VERDICT NOT-REISSUED the released chunk was withheld; the stale pointer still names the dead object, so nothing aliased (observed=0x11)
        SUPERVISE exit status=1
    03_012a179785ab_filesystem_alias_named_copy_is_not_a_copy
        SUPERVISE base 0x100000 /tmp/allocator-tests/wst-03-buggy/target
        SUPERVISE expect wst_read_probe 0x102394
        case=3 arm=buggy
        bytes=48 freed=1 marker=0xaa observed=0x11 aliased=0 damage=0
        VERDICT NOT-REISSUED the released chunk was withheld; the stale pointer still names the dead object, so nothing aliased (observed=0x11)
        SUPERVISE exit status=1
    04_07ffcf90426b_extcap_one_help_string_stored_in_many_owners
        SUPERVISE base 0x100000 /tmp/allocator-tests/wst-04-buggy/target
        SUPERVISE expect wst_read_probe 0x10246e
        case=4 arm=buggy
        bytes=48 freed=1 marker=0xaa observed=0x11 aliased=0 damage=0
        VERDICT NOT-REISSUED the released chunk was withheld; the stale pointer still names the dead object, so nothing aliased (observed=0x11)
        SUPERVISE exit status=1
    05_fb46cda19602_wtap_close_inner_loop_rewinds_outer_cursor
        SUPERVISE base 0x100000 /tmp/allocator-tests/wst-05-buggy/target
        SUPERVISE expect wst_read_probe 0x102346
        case=5 arm=buggy
        bytes=48 freed=1 marker=0xaa observed=0x11 aliased=0 damage=0
        VERDICT NOT-REISSUED the released chunk was withheld; the stale pointer still names the dead object, so nothing aliased (observed=0x11)
        SUPERVISE exit status=1
    06_d3e3c00fbbe2_prefs_static_filter_label_not_cleared
        SUPERVISE base 0x100000 /tmp/allocator-tests/wst-06-buggy/target
        SUPERVISE expect wst_read_probe 0x1022d6
        case=6 arm=buggy
        bytes=48 freed=1 marker=0xaa observed=0x11 aliased=0 damage=0
        VERDICT NOT-REISSUED the released chunk was withheld; the stale pointer still names the dead object, so nothing aliased (observed=0x11)
        SUPERVISE exit status=1
    07_8dc7d164dcdb_prefs_reset_not_idempotent
        SUPERVISE base 0x100000 /tmp/allocator-tests/wst-07-buggy/target
        SUPERVISE expect wst_read_probe 0x1022e6
        case=7 arm=buggy
        bytes=48 freed=1 marker=0xaa observed=0x11 aliased=0 damage=0
        VERDICT NOT-REISSUED the released chunk was withheld; the stale pointer still names the dead object, so nothing aliased (observed=0x11)
        SUPERVISE exit status=1
    08_48a00fd55671_ftype_string_freed_twice_by_caller_and_callee
        SUPERVISE base 0x100000 /tmp/allocator-tests/wst-08-buggy/target
        SUPERVISE expect wst_read_probe 0x1022d8
        case=8 arm=buggy
        bytes=48 freed=1 marker=0xaa observed=0x11 aliased=0 damage=0
        VERDICT NOT-REISSUED the released chunk was withheld; the stale pointer still names the dead object, so nothing aliased (observed=0x11)
        SUPERVISE exit status=1
    09_cfc15838bdec_capture_ifinfo_out_parameter_left_unassigned
        SUPERVISE base 0x100000 /tmp/allocator-tests/wst-09-buggy/target
        SUPERVISE expect wst_read_probe 0x1022fe
        case=9 arm=buggy
        bytes=48 freed=1 marker=0xaa observed=0x11 aliased=0 damage=0
        VERDICT NOT-REISSUED the released chunk was withheld; the stale pointer still names the dead object, so nothing aliased (observed=0x11)
        SUPERVISE exit status=1
