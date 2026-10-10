# wireshark/plain-heap-repros on the Capstone application domain arms -- 2026-10-09

Predictions committed before the run: 60f1c5be25a5 (tools/run-capstone-domain.py, predicted()).

Reproduce (one VM up, see ports/common/application/README.md):

    python3 capstone/bug-corpora/tools/run-capstone-domain.py \
      --corpus capstone/bug-corpora/wireshark/plain-heap-repros --arm spatial|sublet \
      --sdk <application SDK built with CAPSTONE_APPLICATION_HEAP=level0|sublet> \
      --state <capstone-vm state> --out <fresh dir>

Result LINES only, from each arm's record.json.

## spatial

    platform  Capstone application VM on capstone-qemu 32e7c9754f9a, kernel e58613598c89, firmware 6f2b082cb677, capstone-exec 0752aa7c49c9, module a88ed2159b43; SDK heap=level0 runtime 3c72f36acd5b; compiler 7d01722aab88
    env       {'CAPSTONE_GP_NONLIN': '1', 'CAPSTONE_REV_NODES': '65536'}
    tool      0383d3c2caa89fb6   predicted buggy outcome: CAUGHT
    control clean RETURNED  required RETURNED  CONTROL clean RETURNED 1
    control oob   CAUGHT    required CAUGHT    cause=7 pc=0xc020e5b8 address=0xc0251768 in ctl_write_probe+0x58 (the labelled probe)
    control uaf   RETURNED  required RETURNED  CONTROL uaf RETURNED observed=0xaa reissued=1
    00_19c51d27b9_netscaler_record_past_page                           fixed FIXED exit=0   buggy CAUGHT             as predicted cause=5 pc=0xc020f074 address=0xc0254000 in wsh_read_probe+0x38 (the labelled probe)  image e99e9a95dfad8d9d
    01_373504f7c9_dfvm_error_message_wrong_index                       fixed FIXED exit=0   buggy CAUGHT             as predicted cause=5 pc=0xc020f160 address=0xc02521d0 in wsh_read_probe+0x38 (the labelled probe)  image 4e00c297abf38cf8
    02_381681583b_pcapng_nrb_custom_string_over_copy                   fixed FIXED exit=0   buggy CAUGHT             as predicted cause=5 pc=0xc020f254 address=0xc0252240 in wsh_read_probe+0x38 (the labelled probe)  image 04dbe02c972062c6
    03_c556b648aa_strptime_reads_past_null_timezone                    fixed FIXED exit=0   buggy CAUGHT             as predicted cause=5 pc=0xc020f094 address=0xc0252031 in wsh_read_probe+0x38 (the labelled probe)  image 28c09863c1167edb
    04_87803328179_blf_apptext_sized_without_terminator                fixed FIXED exit=0   buggy CAUGHT             as predicted cause=5 pc=0xc020f200 address=0xc02521b0 in wsh_read_probe+0x38 (the labelled probe)  image e1b6a9a435352301
    05_140aad08e081_nettrace_packet_buf_scanned_by_strstr              fixed FIXED exit=0   buggy CAUGHT             as predicted cause=5 pc=0xc020f1f0 address=0xc02521a0 in wsh_read_probe+0x38 (the labelled probe)  image 0064da71a23b372e
    06_3aad1ef236e6_t61_accent_branch_reads_the_following_byte         fixed FIXED exit=0   buggy CAUGHT             as predicted cause=5 pc=0xc020f28c address=0xc0252230 in wsh_read_probe+0x38 (the labelled probe)  image 7b1c676413ce359a
    07_e2ca71beaed2_uat_unbinstring_leaves_no_terminator               fixed FIXED exit=0   buggy CAUGHT             as predicted cause=5 pc=0xc020f1d0 address=0xc0252180 in wsh_read_probe+0x38 (the labelled probe)  image 4b4225ebc7b069eb
    08_0cae98570ebc_cl_find_custom_memcmp_reads_the_prefix_length      fixed FIXED exit=0   buggy CAUGHT             as predicted cause=5 pc=0xc020f22c address=0xc0252200 in wsh_read_probe+0x38 (the labelled probe)  image f8bfbdb5a4459b31
    09_bf123efe154d_uat_oid_empty_field_underflows_the_index           fixed FIXED exit=0   buggy CAUGHT             as predicted cause=5 pc=0xc020efb4 address=0xc0251f70 in wsh_read_probe+0x38 (the labelled probe)  image 6a184d1159fae573
    10_e9b933473e8f_address_to_str_buf_clamps_to_size_not_last_index   fixed FIXED exit=0   buggy CAUGHT             as predicted cause=7 pc=0xc020f054 address=0xc0251fa0 in wsh_write_probe+0x58 (the labelled probe)  image ed9e824fca1d45ef
    11_4b15bf76a7f7_oid_to_str_buf_tail_reservation_one_short          fixed FIXED exit=0   buggy CAUGHT             as predicted cause=7 pc=0xc020f010 address=0xc0251f90 in wsh_write_probe+0x58 (the labelled probe)  image 8a4c6290f2e788b1

## sublet

    platform  Capstone application VM on capstone-qemu 32e7c9754f9a, kernel e58613598c89, firmware 6f2b082cb677, capstone-exec 0752aa7c49c9, module a88ed2159b43; SDK heap=sublet runtime b73508e5005b; compiler 7d01722aab88
    env       {'CAPSTONE_GP_NONLIN': '1', 'CAPSTONE_REV_NODES': '65536'}
    tool      0383d3c2caa89fb6   predicted buggy outcome: CAUGHT
    control clean RETURNED  required RETURNED  CONTROL clean RETURNED 1
    control oob   CAUGHT    required CAUGHT    cause=7 pc=0xc020f70c address=0xc8802018 in ctl_write_probe+0x58 (the labelled probe)
    control uaf   CAUGHT    required CAUGHT    cause=24 pc=0xc020f758 address=0x0 in ctl_read_probe+0x38 (the labelled probe)
    00_19c51d27b9_netscaler_record_past_page                           fixed FIXED exit=0   buggy CAUGHT             as predicted cause=5 pc=0xc02101c8 address=0xc8804000 in wsh_read_probe+0x38 (the labelled probe)  image eefa9782a125e883
    01_373504f7c9_dfvm_error_message_wrong_index                       fixed FIXED exit=0   buggy CAUGHT             as predicted cause=5 pc=0xc02102b4 address=0xc8802130 in wsh_read_probe+0x38 (the labelled probe)  image df9cea140f4b9b91
    02_381681583b_pcapng_nrb_custom_string_over_copy                   fixed FIXED exit=0   buggy CAUGHT             as predicted cause=5 pc=0xc0210158 address=0xc8802010 in wsh_read_probe+0x38 (the labelled probe)  image 0357fa2678a5ae22
    03_c556b648aa_strptime_reads_past_null_timezone                    fixed FIXED exit=0   buggy CAUGHT             as predicted cause=5 pc=0xc020ff98 address=0xc8802001 in wsh_read_probe+0x38 (the labelled probe)  image 807e620d62dd0795
    04_87803328179_blf_apptext_sized_without_terminator                fixed FIXED exit=0   buggy CAUGHT             as predicted cause=5 pc=0xc0210104 address=0xc8802010 in wsh_read_probe+0x38 (the labelled probe)  image 06a8811f73a61fdb
    05_140aad08e081_nettrace_packet_buf_scanned_by_strstr              fixed FIXED exit=0   buggy CAUGHT             as predicted cause=5 pc=0xc02100f4 address=0xc8802010 in wsh_read_probe+0x38 (the labelled probe)  image 34da4de8771cb018
    06_3aad1ef236e6_t61_accent_branch_reads_the_following_byte         fixed FIXED exit=0   buggy CAUGHT             as predicted cause=5 pc=0xc0210190 address=0xc8802010 in wsh_read_probe+0x38 (the labelled probe)  image b5f0a2f057b164ed
    07_e2ca71beaed2_uat_unbinstring_leaves_no_terminator               fixed FIXED exit=0   buggy CAUGHT             as predicted cause=5 pc=0xc02100d4 address=0xc8802010 in wsh_read_probe+0x38 (the labelled probe)  image bc8fd465c9740ff2
    08_0cae98570ebc_cl_find_custom_memcmp_reads_the_prefix_length      fixed FIXED exit=0   buggy CAUGHT             as predicted cause=5 pc=0xc0210130 address=0xc8802010 in wsh_read_probe+0x38 (the labelled probe)  image 91dfa42fde323101
    09_bf123efe154d_uat_oid_empty_field_underflows_the_index           fixed FIXED exit=0   buggy CAUGHT             as predicted cause=5 pc=0xc020feb8 address=0xc8802010 in wsh_read_probe+0x38 (the labelled probe)  image 181d853b44896d80
    10_e9b933473e8f_address_to_str_buf_clamps_to_size_not_last_index   fixed FIXED exit=0   buggy CAUGHT             as predicted cause=7 pc=0xc020ff58 address=0xc8802010 in wsh_write_probe+0x58 (the labelled probe)  image 901d5b7b1e29f52b
    11_4b15bf76a7f7_oid_to_str_buf_tail_reservation_one_short          fixed FIXED exit=0   buggy CAUGHT             as predicted cause=7 pc=0xc020ff14 address=0xc8802010 in wsh_write_probe+0x58 (the labelled probe)  image 428f815ee161bc3a

