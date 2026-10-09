# wireshark/plain-heap-repros on the Capstone arm `capstone-subobject` -- 2026-10-09

Predictions committed before the run: 49498f856e87.
Platform: Capstone application VM on capstone-qemu 32e7c9754f9a, kernel e58613598c89, firmware 6f2b082cb677, capstone-exec 0752aa7c49c9, module a88ed2159b43; SDK heap=level0 heap_log=22 runtime 3c72f36acd5b; compiler 7d01722aab88.

Reproduce (one VM up, see ports/common/application/README.md):

    python3 capstone/bug-corpora/tools/run-capstone-domain.py \
      --corpus capstone/bug-corpora/wireshark/plain-heap-repros --arm capstone-subobject --sdk <SDK, heap=level0> \
      --state <capstone-vm state> --out <fresh dir> \
      --cc-arg=-Xclang \
      --cc-arg=-fcapstone-subobject-bounds

## Controls, from the same boot

    clean  RETURNED  required RETURNED  CONTROL clean RETURNED 1
    oob    CAUGHT    required CAUGHT    cause=7 pc=0xc670e5b8 address=0xc6751958 in ctl_write_probe+0x58 (the labelled probe)
    uaf    RETURNED  required RETURNED  CONTROL uaf RETURNED observed=0xaa reissued=1
    subobj CAUGHT    required CAUGHT    cause=7 pc=0xc670e5b8 address=0xc6751948 in ctl_write_probe+0x58 (the labelled probe)

## Cases: {'CAUGHT': 12}

    case  outcome  as-predicted  fixed  detail  image(sha256/16)
    00_19c51d27b9_netscaler_record_past_page
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=5 pc=0xc670f074 address=0xc6754000 in wsh_read_probe+0x38 (the labelled probe)  e99e9a95dfad8d9d
    01_373504f7c9_dfvm_error_message_wrong_index
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=5 pc=0xc670f160 address=0xc67521d0 in wsh_read_probe+0x38 (the labelled probe)  4e00c297abf38cf8
    02_381681583b_pcapng_nrb_custom_string_over_copy
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=5 pc=0xc670f254 address=0xc6752240 in wsh_read_probe+0x38 (the labelled probe)  04dbe02c972062c6
    03_c556b648aa_strptime_reads_past_null_timezone
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=5 pc=0xc670f094 address=0xc6752031 in wsh_read_probe+0x38 (the labelled probe)  28c09863c1167edb
    04_87803328179_blf_apptext_sized_without_terminator
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=5 pc=0xc670f200 address=0xc67521b0 in wsh_read_probe+0x38 (the labelled probe)  e1b6a9a435352301
    05_140aad08e081_nettrace_packet_buf_scanned_by_strstr
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=5 pc=0xc670f1f0 address=0xc67521a0 in wsh_read_probe+0x38 (the labelled probe)  0064da71a23b372e
    06_3aad1ef236e6_t61_accent_branch_reads_the_following_byte
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=5 pc=0xc670f28c address=0xc6752230 in wsh_read_probe+0x38 (the labelled probe)  7b1c676413ce359a
    07_e2ca71beaed2_uat_unbinstring_leaves_no_terminator
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=5 pc=0xc670f1d0 address=0xc6752180 in wsh_read_probe+0x38 (the labelled probe)  4b4225ebc7b069eb
    08_0cae98570ebc_cl_find_custom_memcmp_reads_the_prefix_length
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=5 pc=0xc670f22c address=0xc6752200 in wsh_read_probe+0x38 (the labelled probe)  f8bfbdb5a4459b31
    09_bf123efe154d_uat_oid_empty_field_underflows_the_index
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=5 pc=0xc670efb4 address=0xc6751f70 in wsh_read_probe+0x38 (the labelled probe)  6a184d1159fae573
    10_e9b933473e8f_address_to_str_buf_clamps_to_size_not_last_index
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=7 pc=0xc670f054 address=0xc6751fa0 in wsh_write_probe+0x58 (the labelled probe)  ed9e824fca1d45ef
    11_4b15bf76a7f7_oid_to_str_buf_tail_reservation_one_short
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=7 pc=0xc670f010 address=0xc6751f90 in wsh_write_probe+0x58 (the labelled probe)  8a4c6290f2e788b1
