# wireshark/plain-heap-repros, arm `cheribsd-subobject` -- 2026-10-09

Platform: stock CheriBSD purecap, kernel CHERI-PURECAP-QEMU, revocation on, every case built with -Xclang -cheri-bounds=subobject-safe (booted on the second host).
Predictions committed before the run: 886abf4cc336. Runtime revocation reported by the ABI control: 1.

Result LINES only; the boot capture is contaminated by construction.

## Controls, from the same boot

    cheribsd-abi: CHERI_ABI pointer_bytes=16 runtime_revocation=1
    cheribsd-bounds: CHERI_BOUNDARY_READY
    subobj-control: exit 162

## Cases: {'CAUGHT': 11, 'NOT CAUGHT': 1}

    00_19c51d27b9_netscaler_record_past_page
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x10242c INSIDE wsh_read_probe [0x10241c, 0x102438)
        SUPERVISE fault signal=34 code=1 addr=0x10242c pc=0x10242c
    01_373504f7c9_dfvm_error_message_wrong_index
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x10247e INSIDE wsh_read_probe [0x10246e, 0x10248a)
        SUPERVISE fault signal=34 code=1 addr=0x10247e pc=0x10247e
    02_381681583b_pcapng_nrb_custom_string_over_copy
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x1024c0 INSIDE wsh_read_probe [0x1024b0, 0x1024cc)
        SUPERVISE fault signal=34 code=1 addr=0x1024c0 pc=0x1024c0
    03_c556b648aa_strptime_reads_past_null_timezone
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED the timezone scan had no case for the terminator, so it consumed the NUL and read the byte after it, one past the allocation
    04_87803328179_blf_apptext_sized_without_terminator
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x102466 INSIDE wsh_read_probe [0x102456, 0x102472)
        SUPERVISE fault signal=34 code=1 addr=0x102466 pc=0x102466
    05_140aad08e081_nettrace_packet_buf_scanned_by_strstr
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x10244e INSIDE wsh_read_probe [0x10243e, 0x10245a)
        SUPERVISE fault signal=34 code=1 addr=0x10244e pc=0x10244e
    06_3aad1ef236e6_t61_accent_branch_reads_the_following_byte
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x1024a8 INSIDE wsh_read_probe [0x102498, 0x1024b4)
        SUPERVISE fault signal=34 code=1 addr=0x1024a8 pc=0x1024a8
    07_e2ca71beaed2_uat_unbinstring_leaves_no_terminator
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x10245a INSIDE wsh_read_probe [0x10244a, 0x102466)
        SUPERVISE fault signal=34 code=1 addr=0x10245a pc=0x10245a
    08_0cae98570ebc_cl_find_custom_memcmp_reads_the_prefix_length
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x102492 INSIDE wsh_read_probe [0x102482, 0x10249e)
        SUPERVISE fault signal=34 code=1 addr=0x102492 pc=0x102492
    09_bf123efe154d_uat_oid_empty_field_underflows_the_index
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x102374 INSIDE wsh_read_probe [0x102364, 0x102380)
        SUPERVISE fault signal=34 code=1 addr=0x102374 pc=0x102374
    10_e9b933473e8f_address_to_str_buf_clamps_to_size_not_last_index
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x1023be OUTSIDE wsh_read_probe [0x10238a, 0x1023a6)
        SUPERVISE fault signal=34 code=1 addr=0x1023be pc=0x1023be
    11_4b15bf76a7f7_oid_to_str_buf_tail_reservation_one_short
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x1023e6 OUTSIDE wsh_read_probe [0x1023b2, 0x1023ce)
        SUPERVISE fault signal=34 code=1 addr=0x1023e6 pc=0x1023e6
