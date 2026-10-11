# wireshark/plain-temporal-repros on the Capstone application domain arms -- 2026-10-09

Predictions committed before the run: 60f1c5be25a5 (tools/run-capstone-domain.py, predicted()).

Reproduce (one VM up, see ports/common/application/README.md):

    python3 capstone/bug-corpora/tools/run-capstone-domain.py \
      --corpus capstone/bug-corpora/wireshark/plain-temporal-repros --arm spatial|sublet \
      --sdk <application SDK built with CAPSTONE_APPLICATION_HEAP=level0|sublet> \
      --state <capstone-vm state> --out <fresh dir>

Result LINES only, from each arm's record.json.

## spatial

    platform  Capstone application VM on capstone-qemu 32e7c9754f9a, kernel e58613598c89, firmware 6f2b082cb677, capstone-exec 0752aa7c49c9, module a88ed2159b43; SDK heap=level0 runtime 3c72f36acd5b; compiler 7d01722aab88
    env       {'CAPSTONE_GP_NONLIN': '1', 'CAPSTONE_REV_NODES': '65536'}
    tool      0383d3c2caa89fb6   predicted buggy outcome: DEFECT-REPRODUCED
    control clean RETURNED  required RETURNED  CONTROL clean RETURNED 1
    control oob   CAUGHT    required CAUGHT    cause=7 pc=0xc020e5b8 address=0xc0251768 in ctl_write_probe+0x58 (the labelled probe)
    control uaf   RETURNED  required RETURNED  CONTROL uaf RETURNED observed=0xaa reissued=1
    00_f3c2e6087e7b_k12_callee_frees_its_own_argument                  fixed FIXED exit=0   buggy DEFECT-REPRODUCED  as predicted VERDICT DEFECT-REPRODUCED destroy_k12_file_data frees the struct it is handed, so the caller's following g_free releases storage that now belongs to another object  image 2e02ef1ea389b5c2
    01_7dcf69480de8_peak_trc_callee_frees_state_struct                 fixed FIXED exit=0   buggy DEFECT-REPRODUCED  as predicted VERDICT DEFECT-REPRODUCED clean_trc_state frees the state struct it is handed, so the caller's following g_free releases storage that now belongs to another object  image 8833e865f724fd54
    02_0fc7f3781351_wspstat_container_freed_before_its_contents        fixed FIXED exit=0   buggy DEFECT-REPRODUCED  as predicted VERDICT DEFECT-REPRODUCED the container was freed before the hash table it owns was torn down, so the teardown loads sp->hash out of storage that now belongs to another object  image 463e809dcf7ec8ec
    03_012a179785ab_filesystem_alias_named_copy_is_not_a_copy          fixed FIXED exit=0   buggy DEFECT-REPRODUCED  as predicted VERDICT DEFECT-REPRODUCED the variable named _copy was a plain assignment, so freeing it destroyed the buffer pf_dir_path still points at, and ws_mkdir reads it  image bfcc6b34f2761bb3
    04_07ffcf90426b_extcap_one_help_string_stored_in_many_owners       fixed FIXED exit=0   buggy DEFECT-REPRODUCED  as predicted VERDICT DEFECT-REPRODUCED one g_strdup'd help string was stored into every interface the loop produced, so the second interface's release reaches storage that now belongs to another object  image 6b6a18244e6338fd
    05_fb46cda19602_wtap_close_inner_loop_rewinds_outer_cursor         fixed FIXED exit=0   buggy DEFECT-REPRODUCED  as predicted VERDICT DEFECT-REPRODUCED the inner loop reused the outer loop's index, rewinding it over description strings already freed and never nulled  image eaf4cea68539cd8d
    06_d3e3c00fbbe2_prefs_static_filter_label_not_cleared              fixed FIXED exit=0   buggy DEFECT-REPRODUCED  as predicted VERDICT DEFECT-REPRODUCED the function-static filter label was freed but left set, so the next legacy entry passes storage that now belongs to another object  image aa22a09b8b4d2a4b
    07_8dc7d164dcdb_prefs_reset_not_idempotent                         fixed FIXED exit=0   buggy DEFECT-REPRODUCED  as predicted VERDICT DEFECT-REPRODUCED prefs_reset freed the version field without clearing it, so a second reset reaches storage that now belongs to another object  image d8044289d5d6e9eb
    08_48a00fd55671_ftype_string_freed_twice_by_caller_and_callee      fixed FIXED exit=0   buggy DEFECT-REPRODUCED  as predicted VERDICT DEFECT-REPRODUCED the value was released up front and again by the delegate, because string_fvalue_free does not clear the field it frees  image 970305adbb47ba3c
    09_cfc15838bdec_capture_ifinfo_out_parameter_left_unassigned       fixed FIXED exit=0   buggy DEFECT-REPRODUCED  as predicted VERDICT DEFECT-REPRODUCED the early return left *err_str unassigned, so the caller's global still named the message it had just freed and the next refresh frees it again  image 0e661c07dd65b714

## sublet

    platform  Capstone application VM on capstone-qemu 32e7c9754f9a, kernel e58613598c89, firmware 6f2b082cb677, capstone-exec 0752aa7c49c9, module a88ed2159b43; SDK heap=sublet runtime b73508e5005b; compiler 7d01722aab88
    env       {'CAPSTONE_GP_NONLIN': '1', 'CAPSTONE_REV_NODES': '65536'}
    tool      0383d3c2caa89fb6   predicted buggy outcome: CAUGHT
    control clean RETURNED  required RETURNED  CONTROL clean RETURNED 1
    control oob   CAUGHT    required CAUGHT    cause=7 pc=0xc020f70c address=0xc8802018 in ctl_write_probe+0x58 (the labelled probe)
    control uaf   CAUGHT    required CAUGHT    cause=24 pc=0xc020f758 address=0x0 in ctl_read_probe+0x38 (the labelled probe)
    00_f3c2e6087e7b_k12_callee_frees_its_own_argument                  fixed FIXED exit=0   buggy CAUGHT             as predicted cause=24 pc=0xc020fc58 address=0x0 in wst_read_probe+0x38 (the labelled probe)  image 7fce1b765eb2d3dd
    01_7dcf69480de8_peak_trc_callee_frees_state_struct                 fixed FIXED exit=0   buggy CAUGHT             as predicted cause=24 pc=0xc020fc58 address=0x0 in wst_read_probe+0x38 (the labelled probe)  image e72595ac09d380c4
    02_0fc7f3781351_wspstat_container_freed_before_its_contents        fixed FIXED exit=0   buggy CAUGHT             as predicted cause=24 pc=0xc020fd28 address=0x0 in wst_read_probe+0x38 (the labelled probe)  image 1e28f342805bfdb0
    03_012a179785ab_filesystem_alias_named_copy_is_not_a_copy          fixed FIXED exit=0   buggy CAUGHT             as predicted cause=24 pc=0xc020fcd8 address=0x0 in wst_read_probe+0x38 (the labelled probe)  image 61dc183f717b980f
    04_07ffcf90426b_extcap_one_help_string_stored_in_many_owners       fixed FIXED exit=0   buggy CAUGHT             as predicted cause=24 pc=0xc020fea0 address=0x0 in wst_read_probe+0x38 (the labelled probe)  image a91e504ccfcbff75
    05_fb46cda19602_wtap_close_inner_loop_rewinds_outer_cursor         fixed FIXED exit=0   buggy CAUGHT             as predicted cause=24 pc=0xc020fd40 address=0x0 in wst_read_probe+0x38 (the labelled probe)  image 5411133c2ff39438
    06_d3e3c00fbbe2_prefs_static_filter_label_not_cleared              fixed FIXED exit=0   buggy CAUGHT             as predicted cause=24 pc=0xc020fbfc address=0x0 in wst_read_probe+0x38 (the labelled probe)  image 7da448a0aa916107
    07_8dc7d164dcdb_prefs_reset_not_idempotent                         fixed FIXED exit=0   buggy CAUGHT             as predicted cause=24 pc=0xc020fbfc address=0x0 in wst_read_probe+0x38 (the labelled probe)  image e283b6066b586187
    08_48a00fd55671_ftype_string_freed_twice_by_caller_and_callee      fixed FIXED exit=0   buggy CAUGHT             as predicted cause=24 pc=0xc020fc30 address=0x0 in wst_read_probe+0x38 (the labelled probe)  image 71febe630db56f1a
    09_cfc15838bdec_capture_ifinfo_out_parameter_left_unassigned       fixed FIXED exit=0   buggy CAUGHT             as predicted cause=24 pc=0xc020fbfc address=0x0 in wst_read_probe+0x38 (the labelled probe)  image 0b299fcfde506464

