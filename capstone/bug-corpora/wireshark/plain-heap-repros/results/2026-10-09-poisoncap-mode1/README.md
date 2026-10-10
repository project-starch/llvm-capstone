# wireshark/plain-heap-repros, arm `poisoncap-protected` -- 2026-10-09

Platform: PoisonCap mode 1: the same platform with runtime revocation ON -- poison on free, then revocation (booted on the second host).
Predictions committed before the run: the corpus's committed predictions and 2852edcdd747 (size-class slack). Runtime revocation reported by the ABI control: 1.

Result LINES only; the boot capture is contaminated by construction.

## Controls, from the same boot

    cheribsd-abi: CHERI_ABI pointer_bytes=16 runtime_revocation=1
    cheribsd-bounds: CHERI_BOUNDARY_READY

## Cases: {'CAUGHT': 11, 'NOT CAUGHT': 1}

    00_19c51d27b9_netscaler_record_past_page
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x10236c INSIDE wsh_read_probe [0x10235c, 0x102378)
        SUPERVISE fault signal=34 code=1 addr=0x10236c pc=0x10236c
    01_373504f7c9_dfvm_error_message_wrong_index
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x1023be INSIDE wsh_read_probe [0x1023ae, 0x1023ca)
        SUPERVISE fault signal=34 code=1 addr=0x1023be pc=0x1023be
    02_381681583b_pcapng_nrb_custom_string_over_copy
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x102400 INSIDE wsh_read_probe [0x1023f0, 0x10240c)
        SUPERVISE fault signal=34 code=1 addr=0x102400 pc=0x102400
    03_c556b648aa_strptime_reads_past_null_timezone
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED the timezone scan had no case for the terminator, so it consumed the NUL and read the byte after it, one past the allocation
    04_87803328179_blf_apptext_sized_without_terminator
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x1023a6 INSIDE wsh_read_probe [0x102396, 0x1023b2)
        SUPERVISE fault signal=34 code=1 addr=0x1023a6 pc=0x1023a6
    05_140aad08e081_nettrace_packet_buf_scanned_by_strstr
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x10238e INSIDE wsh_read_probe [0x10237e, 0x10239a)
        SUPERVISE fault signal=34 code=1 addr=0x10238e pc=0x10238e
    06_3aad1ef236e6_t61_accent_branch_reads_the_following_byte
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x1023e8 INSIDE wsh_read_probe [0x1023d8, 0x1023f4)
        SUPERVISE fault signal=34 code=1 addr=0x1023e8 pc=0x1023e8
    07_e2ca71beaed2_uat_unbinstring_leaves_no_terminator
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x10239a INSIDE wsh_read_probe [0x10238a, 0x1023a6)
        SUPERVISE fault signal=34 code=1 addr=0x10239a pc=0x10239a
    08_0cae98570ebc_cl_find_custom_memcmp_reads_the_prefix_length
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x1023d2 INSIDE wsh_read_probe [0x1023c2, 0x1023de)
        SUPERVISE fault signal=34 code=1 addr=0x1023d2 pc=0x1023d2
    09_bf123efe154d_uat_oid_empty_field_underflows_the_index
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x1022b4 INSIDE wsh_read_probe [0x1022a4, 0x1022c0)
        SUPERVISE fault signal=34 code=1 addr=0x1022b4 pc=0x1022b4
    10_e9b933473e8f_address_to_str_buf_clamps_to_size_not_last_index
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x1022fe OUTSIDE wsh_read_probe [0x1022ca, 0x1022e6)
        SUPERVISE fault signal=34 code=1 addr=0x1022fe pc=0x1022fe
    11_4b15bf76a7f7_oid_to_str_buf_tail_reservation_one_short
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x102326 OUTSIDE wsh_read_probe [0x1022f2, 0x10230e)
        SUPERVISE fault signal=34 code=1 addr=0x102326 pc=0x102326
