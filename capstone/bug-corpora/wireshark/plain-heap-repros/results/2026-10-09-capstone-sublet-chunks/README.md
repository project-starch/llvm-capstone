# wireshark/plain-heap-repros on the Capstone arm `sublet-chunks` -- 2026-10-09

Predictions committed before the run: 49498f856e87.
Platform: Capstone application VM on capstone-qemu 32e7c9754f9a, kernel e58613598c89, firmware 6f2b082cb677, capstone-exec 0752aa7c49c9, module a88ed2159b43; SDK heap=sublet heap_log=24 runtime cdfc061d5e9a; compiler 7d01722aab88.

Reproduce (one VM up, see ports/common/application/README.md):

    python3 capstone/bug-corpora/tools/run-capstone-domain.py \
      --corpus capstone/bug-corpora/wireshark/plain-heap-repros --arm sublet-chunks --sdk <SDK, heap=sublet> \
      --state <capstone-vm state> --out <fresh dir> \
      --cc-arg=<scratch>/capsdk/chunks-objs/chunks.o \
      --cc-arg=<scratch>/capsdk/chunks-objs/tsapp-wmem-chunks.o

## Controls, from the same boot

    clean  RETURNED  required RETURNED  CONTROL clean RETURNED 1
    oob    CAUGHT    required CAUGHT    cause=7 pc=0xc020f76c address=0xc5002018 in ctl_write_probe+0x58 (the labelled probe)
    uaf    CAUGHT    required CAUGHT    cause=24 pc=0xc020f7b8 address=0x0 in ctl_read_probe+0x38 (the labelled probe)
    subobj RETURNED  required RETURNED  CONTROL subobj RETURNED b0=0x5a

## Cases: {'CAUGHT': 12}

    case  outcome  as-predicted  fixed  detail  image(sha256/16)
    00_19c51d27b9_netscaler_record_past_page
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=5 pc=0xc02109c4 address=0xc5004000 in wsh_read_probe+0x38 (the labelled probe)  37cf95d3f4c6be78
    01_373504f7c9_dfvm_error_message_wrong_index
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=5 pc=0xc0210ab0 address=0xc5002130 in wsh_read_probe+0x38 (the labelled probe)  36434c4c61c09179
    02_381681583b_pcapng_nrb_custom_string_over_copy
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=5 pc=0xc0210ba4 address=0xc5002010 in wsh_read_probe+0x38 (the labelled probe)  bd779a3c8b2035f4
    03_c556b648aa_strptime_reads_past_null_timezone
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=5 pc=0xc02109e4 address=0xc5002001 in wsh_read_probe+0x38 (the labelled probe)  fb26cc4d954e0dc8
    04_87803328179_blf_apptext_sized_without_terminator
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=5 pc=0xc0210b50 address=0xc5002010 in wsh_read_probe+0x38 (the labelled probe)  5f30a44bd445fee6
    05_140aad08e081_nettrace_packet_buf_scanned_by_strstr
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=5 pc=0xc0210b40 address=0xc5002010 in wsh_read_probe+0x38 (the labelled probe)  c0d577922a03997e
    06_3aad1ef236e6_t61_accent_branch_reads_the_following_byte
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=5 pc=0xc0210bdc address=0xc5002010 in wsh_read_probe+0x38 (the labelled probe)  2a43f769bce9846a
    07_e2ca71beaed2_uat_unbinstring_leaves_no_terminator
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=5 pc=0xc0210b20 address=0xc5002010 in wsh_read_probe+0x38 (the labelled probe)  aa8a87164c76957a
    08_0cae98570ebc_cl_find_custom_memcmp_reads_the_prefix_length
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=5 pc=0xc0210b7c address=0xc5002010 in wsh_read_probe+0x38 (the labelled probe)  39e66d699355cee0
    09_bf123efe154d_uat_oid_empty_field_underflows_the_index
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=5 pc=0xc0210904 address=0xc5002010 in wsh_read_probe+0x38 (the labelled probe)  1b0925f011c93bef
    10_e9b933473e8f_address_to_str_buf_clamps_to_size_not_last_index
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=7 pc=0xc02109a4 address=0xc5002010 in wsh_write_probe+0x58 (the labelled probe)  fd54626709818f5b
    11_4b15bf76a7f7_oid_to_str_buf_tail_reservation_one_short
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=7 pc=0xc0210960 address=0xc5002010 in wsh_write_probe+0x58 (the labelled probe)  30cf6ee53dcd629b
