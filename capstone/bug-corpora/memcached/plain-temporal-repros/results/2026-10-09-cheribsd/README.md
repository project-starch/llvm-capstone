# memcached/plain-temporal-repros on stock CheriBSD purecap, revocation ON -- 2026-10-09

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
    PASS mct-00-fixed
    PASS mct-00-buggy
    PASS mct-01-fixed
    PASS mct-01-buggy
    PASS mct-02-fixed
    PASS mct-02-buggy

## Each buggy arm's record

    00_0d4901071c74_restart_line_freed_at_entry_but_not_cleared
        SUPERVISE base 0x100000 /tmp/allocator-tests/mct-00-buggy/target
        SUPERVISE expect mct_read_probe 0x10231e
        case=0 arm=buggy
        bytes=48 freed=1 marker=0xaa observed=0x11 aliased=0 damage=0
        VERDICT NOT-REISSUED the released chunk was withheld; the stale pointer still names the dead object, so nothing aliased (observed=0x11)
        SUPERVISE exit status=1
    01_e7793811f8c8_logger_write_to_watcher_freed_by_the_poll
        SUPERVISE base 0x100000 /tmp/allocator-tests/mct-01-buggy/target
        SUPERVISE expect mct_write_probe 0x1022f8
        case=1 arm=buggy
        bytes=48 freed=1 marker=0xaa observed=0xaa aliased=0 damage=0
        VERDICT NOT-REISSUED the released chunk was withheld; the stale pointer still names the dead object, so nothing aliased (observed=0xaa)
        SUPERVISE exit status=1
    02_3bc58f6ea55a_logger_loop_condition_still_reads_the_freed_watcher
        SUPERVISE base 0x100000 /tmp/allocator-tests/mct-02-buggy/target
        SUPERVISE expect mct_read_probe 0x102326
        case=2 arm=buggy
        bytes=48 freed=1 marker=0xaa observed=0x11 aliased=0 damage=0
        VERDICT NOT-REISSUED the released chunk was withheld; the stale pointer still names the dead object, so nothing aliased (observed=0x11)
        SUPERVISE exit status=1
