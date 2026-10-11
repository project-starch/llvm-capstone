# memcached/plain-heap-repros on stock CheriBSD purecap, revocation ON -- 2026-10-08

Reproduced by: CHERI_SDK=~/cheri/output/sdk CHERI_SYSROOT=~/cheri/rootfs-purecap \
               CHERI_IMAGE=~/cheri/output/cheribsd-riscv64-purecap.img \
                 bash runners/run-cheribsd.sh <fresh-outdir>

Kernel CHERI-PURECAP-QEMU. Result LINES only: the boot capture is contaminated by
construction, so what is kept is the controls, the per-case verdict, and the supervise
lines carrying the signal, the si_code and the probe address resolved from the ELF.

## Platform controls -- a suite whose controls did not fire is not a reading

    cheribsd-abi:
        CHERI_ABI pointer_bytes=16 runtime_revocation=1
    cheribsd-bounds:
        CHERI_ABI pointer_bytes=16 runtime_revocation=1
        CHERI_BOUNDARY_READY

## Per-case verdicts

    PASS cheribsd-abi
    PASS cheribsd-bounds
    PASS mch-00-fixed
    PASS mch-00-buggy
    PASS mch-01-fixed
    PASS mch-01-buggy
    PASS mch-02-fixed
    PASS mch-02-buggy
    PASS mch-03-fixed
    PASS mch-03-buggy
    PASS mch-04-fixed
    PASS mch-04-buggy
    PASS mch-05-fixed
    PASS mch-05-buggy
    PASS mch-06-fixed
    PASS mch-06-buggy
    PASS mch-07-fixed
    PASS mch-07-buggy
    PASS mch-08-fixed
    PASS mch-08-buggy

## The supervise lines, per buggy arm

    00_ddee3e2_authfile_scan_past_calloc
        SUPERVISE base 0x100000 /tmp/allocator-tests/mch-00-buggy/target
        SUPERVISE expect mch_write_probe 0x1023be
        case=0 arm=buggy
        cap=9 touched=9 crossed=1 damage=1
        VERDICT DEFECT-REPRODUCED fgets wrote its terminating NUL at offset sb.st_size of a calloc(1, sb.st_size), one byte past the allocation
        SUPERVISE exit status=0
    01_d5d9ff0_cachedump_end_marker_off_by_one
        SUPERVISE base 0x100000 /tmp/allocator-tests/mch-01-buggy/target
        SUPERVISE expect mch_write_probe 0x102468
        SUPERVISE fault signal=34 code=1 addr=0x102480 pc=0x102480
        SUPERVISE exit signalled=34
    02_391f2e4762bf_freesuffix_realloc_sized_in_bytes
        SUPERVISE base 0x100000 /tmp/allocator-tests/mch-02-buggy/target
        SUPERVISE expect mch_write_probe 0x102476
        SUPERVISE fault signal=34 code=1 addr=0x10248e pc=0x10248e
        SUPERVISE exit signalled=34
    03_16a809e2a062_cache_create_freelist_sized_by_object
        SUPERVISE base 0x100000 /tmp/allocator-tests/mch-03-buggy/target
        SUPERVISE expect mch_write_probe 0x102390
        SUPERVISE fault signal=34 code=1 addr=0x1023a8 pc=0x1023a8
        SUPERVISE exit signalled=34
    04_40aff8b0f113_stats_end_marker_past_exact_buffer
        SUPERVISE base 0x100000 /tmp/allocator-tests/mch-04-buggy/target
        SUPERVISE expect mch_write_probe 0x1023d8
        SUPERVISE fault signal=34 code=1 addr=0x1023f0 pc=0x1023f0
        SUPERVISE exit signalled=34
    05_49f3b0ca9b57_out_string_crlf_copied_with_its_nul
        SUPERVISE base 0x100000 /tmp/allocator-tests/mch-05-buggy/target
        SUPERVISE expect mch_write_probe 0x10240c
        SUPERVISE fault signal=34 code=1 addr=0x102424 pc=0x102424
        SUPERVISE exit signalled=34
    06_212c3820c7bb_key_hash_filter_tag_length_underflow
        SUPERVISE base 0x100000 /tmp/allocator-tests/mch-06-buggy/target
        SUPERVISE expect mch_write_probe 0x102578
        SUPERVISE fault signal=34 code=1 addr=0x1025ac pc=0x1025ac
        SUPERVISE exit signalled=34
    07_0f605245cf3f_bin_delete_logs_unterminated_key
        SUPERVISE base 0x100000 /tmp/allocator-tests/mch-07-buggy/target
        SUPERVISE expect mch_write_probe 0x102400
        SUPERVISE fault signal=34 code=1 addr=0x102434 pc=0x102434
        SUPERVISE exit signalled=34
    08_fa51ad8452d5_slab_list_shuffle_reads_one_past
        SUPERVISE base 0x100000 /tmp/allocator-tests/mch-08-buggy/target
        SUPERVISE expect mch_write_probe 0x102424
        SUPERVISE fault signal=34 code=1 addr=0x102458 pc=0x102458
        SUPERVISE exit signalled=34
