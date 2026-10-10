# wireshark/plain-temporal-repros, arm `poisoncap-protected` -- 2026-10-09

Platform: PoisonCap mode 1: the same platform with runtime revocation ON -- poison on free, then revocation (booted on the second host).
Predictions committed before the run: 2852edcdd747. Runtime revocation reported by the ABI control: 1.

Result LINES only; the boot capture is contaminated by construction.

## Controls, from the same boot

    cheribsd-abi: CHERI_ABI pointer_bytes=16 runtime_revocation=1
    cheribsd-bounds: CHERI_BOUNDARY_READY
    revocation-control: exit 162; SUPERVISE fault signal=34 code=2 addr=0x101dd6 pc=0x101dd6

## Cases: {'CAUGHT': 9, 'NOT-REISSUED': 1}

    00_f3c2e6087e7b_k12_callee_frees_its_own_argument
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x102286 INSIDE wst_read_probe [0x102276, 0x102292)
        SUPERVISE fault signal=34 code=3 addr=0x102286 pc=0x102286
    01_7dcf69480de8_peak_trc_callee_frees_state_struct
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x102286 INSIDE wst_read_probe [0x102276, 0x102292)
        SUPERVISE fault signal=34 code=3 addr=0x102286 pc=0x102286
    02_0fc7f3781351_wspstat_container_freed_before_its_contents
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x1022b6 INSIDE wst_read_probe [0x1022a6, 0x1022c2)
        SUPERVISE fault signal=34 code=3 addr=0x1022b6 pc=0x1022b6
    03_012a179785ab_filesystem_alias_named_copy_is_not_a_copy
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x102324 INSIDE wst_read_probe [0x102314, 0x102330)
        SUPERVISE fault signal=34 code=3 addr=0x102324 pc=0x102324
    04_07ffcf90426b_extcap_one_help_string_stored_in_many_owners
        fixed: exit 0 VERDICT FIXED
        buggy: exit 1 -> NOT-REISSUED
        VERDICT NOT-REISSUED the released chunk was withheld; the stale pointer still names the dead object, so nothing aliased (observed=0x30)
    05_fb46cda19602_wtap_close_inner_loop_rewinds_outer_cursor
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x1022d6 INSIDE wst_read_probe [0x1022c6, 0x1022e2)
        SUPERVISE fault signal=34 code=3 addr=0x1022d6 pc=0x1022d6
    06_d3e3c00fbbe2_prefs_static_filter_label_not_cleared
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x102266 INSIDE wst_read_probe [0x102256, 0x102272)
        SUPERVISE fault signal=34 code=3 addr=0x102266 pc=0x102266
    07_8dc7d164dcdb_prefs_reset_not_idempotent
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x102276 INSIDE wst_read_probe [0x102266, 0x102282)
        SUPERVISE fault signal=34 code=3 addr=0x102276 pc=0x102276
    08_48a00fd55671_ftype_string_freed_twice_by_caller_and_callee
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x102268 INSIDE wst_read_probe [0x102258, 0x102274)
        SUPERVISE fault signal=34 code=3 addr=0x102268 pc=0x102268
    09_cfc15838bdec_capture_ifinfo_out_parameter_left_unassigned
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x10228e INSIDE wst_read_probe [0x10227e, 0x10229a)
        SUPERVISE fault signal=34 code=3 addr=0x10228e pc=0x10228e

## Against the prediction

Case 04 did NOT fault, refuting its pre-registered SIGPROT: the stale read completed and returned 0x30.
FFmpeg plain-temporal 12 reads the same in the same mode; both keep a second same-size block live
across the free. N = 2; the mechanism is not established. The other nine faulted with si_code 3.
