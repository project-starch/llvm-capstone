# wireshark/plain-heap-repros on stock CheriBSD purecap, revocation ON -- 2026-10-08

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
    PASS wsh-00-fixed
    PASS wsh-00-buggy
    PASS wsh-01-fixed
    PASS wsh-01-buggy
    PASS wsh-02-fixed
    PASS wsh-02-buggy
    PASS wsh-03-fixed
    PASS wsh-03-buggy
    PASS wsh-04-fixed
    PASS wsh-04-buggy
    PASS wsh-05-fixed
    PASS wsh-05-buggy
    PASS wsh-06-fixed
    PASS wsh-06-buggy
    PASS wsh-07-fixed
    PASS wsh-07-buggy
    PASS wsh-08-fixed
    PASS wsh-08-buggy
    PASS wsh-09-fixed
    PASS wsh-09-buggy
    PASS wsh-10-fixed
    PASS wsh-10-buggy
    PASS wsh-11-fixed
    PASS wsh-11-buggy

## The supervise lines, per buggy arm

    00_19c51d27b9_netscaler_record_past_page
        SUPERVISE base 0x100000 /tmp/allocator-tests/wsh-00-buggy/target
        SUPERVISE expect wsh_read_probe 0x1023dc
        SUPERVISE fault signal=34 code=1 addr=0x1023ec pc=0x1023ec
        SUPERVISE exit signalled=34
    01_373504f7c9_dfvm_error_message_wrong_index
        SUPERVISE base 0x100000 /tmp/allocator-tests/wsh-01-buggy/target
        SUPERVISE expect wsh_read_probe 0x10242e
        SUPERVISE fault signal=34 code=1 addr=0x10243e pc=0x10243e
        SUPERVISE exit signalled=34
    02_381681583b_pcapng_nrb_custom_string_over_copy
        SUPERVISE base 0x100000 /tmp/allocator-tests/wsh-02-buggy/target
        SUPERVISE expect wsh_read_probe 0x102470
        SUPERVISE fault signal=34 code=1 addr=0x102480 pc=0x102480
        SUPERVISE exit signalled=34
    03_c556b648aa_strptime_reads_past_null_timezone
        SUPERVISE base 0x100000 /tmp/allocator-tests/wsh-03-buggy/target
        SUPERVISE expect wsh_read_probe 0x10237a
        case=3 arm=buggy
        cap=1 touched=1 extent=1 crossed=1 damage=1
        VERDICT DEFECT-REPRODUCED the timezone scan had no case for the terminator, so it consumed the NUL and read the byte after it, one past the allocation
        SUPERVISE exit status=0
    04_87803328179_blf_apptext_sized_without_terminator
        SUPERVISE base 0x100000 /tmp/allocator-tests/wsh-04-buggy/target
        SUPERVISE expect wsh_read_probe 0x102416
        SUPERVISE fault signal=34 code=1 addr=0x102426 pc=0x102426
        SUPERVISE exit signalled=34
    05_140aad08e081_nettrace_packet_buf_scanned_by_strstr
        SUPERVISE base 0x100000 /tmp/allocator-tests/wsh-05-buggy/target
        SUPERVISE expect wsh_read_probe 0x1023fe
        SUPERVISE fault signal=34 code=1 addr=0x10240e pc=0x10240e
        SUPERVISE exit signalled=34
    06_3aad1ef236e6_t61_accent_branch_reads_the_following_byte
        SUPERVISE base 0x100000 /tmp/allocator-tests/wsh-06-buggy/target
        SUPERVISE expect wsh_read_probe 0x102458
        SUPERVISE fault signal=34 code=1 addr=0x102468 pc=0x102468
        SUPERVISE exit signalled=34
    07_e2ca71beaed2_uat_unbinstring_leaves_no_terminator
        SUPERVISE base 0x100000 /tmp/allocator-tests/wsh-07-buggy/target
        SUPERVISE expect wsh_read_probe 0x10240a
        SUPERVISE fault signal=34 code=1 addr=0x10241a pc=0x10241a
        SUPERVISE exit signalled=34
    08_0cae98570ebc_cl_find_custom_memcmp_reads_the_prefix_length
        SUPERVISE base 0x100000 /tmp/allocator-tests/wsh-08-buggy/target
        SUPERVISE expect wsh_read_probe 0x102442
        SUPERVISE fault signal=34 code=1 addr=0x102452 pc=0x102452
        SUPERVISE exit signalled=34
    09_bf123efe154d_uat_oid_empty_field_underflows_the_index
        SUPERVISE base 0x100000 /tmp/allocator-tests/wsh-09-buggy/target
        SUPERVISE expect wsh_read_probe 0x102324
        SUPERVISE fault signal=34 code=1 addr=0x102334 pc=0x102334
        SUPERVISE exit signalled=34
    10_e9b933473e8f_address_to_str_buf_clamps_to_size_not_last_index
        SUPERVISE base 0x100000 /tmp/allocator-tests/wsh-10-buggy/target
        SUPERVISE expect wsh_read_probe 0x10234a
        SUPERVISE fault signal=34 code=1 addr=0x10237e pc=0x10237e
        SUPERVISE exit signalled=34
    11_4b15bf76a7f7_oid_to_str_buf_tail_reservation_one_short
        SUPERVISE base 0x100000 /tmp/allocator-tests/wsh-11-buggy/target
        SUPERVISE expect wsh_read_probe 0x102372
        SUPERVISE fault signal=34 code=1 addr=0x1023a6 pc=0x1023a6
        SUPERVISE exit signalled=34
