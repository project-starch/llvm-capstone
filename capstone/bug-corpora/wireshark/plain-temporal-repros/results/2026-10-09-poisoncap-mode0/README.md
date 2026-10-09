# wireshark/plain-temporal-repros, arm `poisoncap-spatial` -- 2026-10-09

Platform: PoisonCap mode 0: the rebuilt published PoisonCap platform (CHERI-PURECAP-QEMU-POISON kernel, libc with the mrs poison-retire fix), runtime revocation OFF (booted on the second host).
Predictions committed before the run: 2852edcdd747. Runtime revocation reported by the ABI control: 0.

Result LINES only; the boot capture is contaminated by construction.

## Controls, from the same boot

    cheribsd-abi: CHERI_ABI pointer_bytes=16 runtime_revocation=0
    cheribsd-bounds: CHERI_BOUNDARY_READY
    revocation-control: exit 0

## Cases: {'NOT CAUGHT': 10}

    00_f3c2e6087e7b_k12_callee_frees_its_own_argument
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED destroy_k12_file_data frees the struct it is handed, so the caller's following g_free releases storage that now belongs to another obj
    01_7dcf69480de8_peak_trc_callee_frees_state_struct
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED clean_trc_state frees the state struct it is handed, so the caller's following g_free releases storage that now belongs to another obj
    02_0fc7f3781351_wspstat_container_freed_before_its_contents
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED the container was freed before the hash table it owns was torn down, so the teardown loads sp->hash out of storage that now belongs to
    03_012a179785ab_filesystem_alias_named_copy_is_not_a_copy
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED the variable named _copy was a plain assignment, so freeing it destroyed the buffer pf_dir_path still points at, and ws_mkdir reads it
    04_07ffcf90426b_extcap_one_help_string_stored_in_many_owners
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED one g_strdup'd help string was stored into every interface the loop produced, so the second interface's release reaches storage that n
    05_fb46cda19602_wtap_close_inner_loop_rewinds_outer_cursor
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED the inner loop reused the outer loop's index, rewinding it over description strings already freed and never nulled
    06_d3e3c00fbbe2_prefs_static_filter_label_not_cleared
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED the function-static filter label was freed but left set, so the next legacy entry passes storage that now belongs to another object
    07_8dc7d164dcdb_prefs_reset_not_idempotent
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED prefs_reset freed the version field without clearing it, so a second reset reaches storage that now belongs to another object
    08_48a00fd55671_ftype_string_freed_twice_by_caller_and_callee
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED the value was released up front and again by the delegate, because string_fvalue_free does not clear the field it frees
    09_cfc15838bdec_capture_ifinfo_out_parameter_left_unassigned
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED the early return left *err_str unassigned, so the caller's global still named the message it had just freed and the next refresh frees
