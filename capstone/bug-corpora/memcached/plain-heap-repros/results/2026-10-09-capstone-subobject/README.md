# memcached/plain-heap-repros on the Capstone arm `capstone-subobject` -- 2026-10-09

Predictions committed before the run: 49498f856e87.
Platform: Capstone application VM on capstone-qemu 32e7c9754f9a, kernel e58613598c89, firmware 6f2b082cb677, capstone-exec 0752aa7c49c9, module a88ed2159b43; SDK heap=level0 heap_log=22 runtime 3c72f36acd5b; compiler 7d01722aab88.

Reproduce (one VM up, see ports/common/application/README.md):

    python3 capstone/bug-corpora/tools/run-capstone-domain.py \
      --corpus capstone/bug-corpora/memcached/plain-heap-repros --arm capstone-subobject --sdk <SDK, heap=level0> \
      --state <capstone-vm state> --out <fresh dir> \
      --cc-arg=-Xclang \
      --cc-arg=-fcapstone-subobject-bounds

## Controls, from the same boot

    clean  RETURNED  required RETURNED  CONTROL clean RETURNED 1
    oob    CAUGHT    required CAUGHT    cause=7 pc=0xc670e5b8 address=0xc6751958 in ctl_write_probe+0x58 (the labelled probe)
    uaf    RETURNED  required RETURNED  CONTROL uaf RETURNED observed=0xaa reissued=1
    subobj CAUGHT    required CAUGHT    cause=7 pc=0xc670e5b8 address=0xc6751948 in ctl_write_probe+0x58 (the labelled probe)

## Cases: {'CAUGHT': 9}

    case  outcome  as-predicted  fixed  detail  image(sha256/16)
    00_ddee3e2_authfile_scan_past_calloc
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=7 pc=0xc670f040 address=0xc6751f99 in mch_write_probe+0x58 (the labelled probe)  36e178db06b6419c
    01_d5d9ff0_cachedump_end_marker_off_by_one
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=7 pc=0xc670f14c address=0xc6752140 in mch_write_probe+0x58 (the labelled probe)  767e14f155870752
    02_391f2e4762bf_freesuffix_realloc_sized_in_bytes
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=7 pc=0xc670f76c address=0xc6752730 in mch_write_probe+0x58 (the labelled probe)  73177c741354eab3
    03_16a809e2a062_cache_create_freelist_sized_by_object
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=7 pc=0xc670f014 address=0xc6752000 in mch_write_probe+0x58 (the labelled probe)  890befd830fa5991
    04_40aff8b0f113_stats_end_marker_past_exact_buffer
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=7 pc=0xc670efc4 address=0xc6751fb0 in mch_write_probe+0x58 (the labelled probe)  911b366cd9ad1f18
    05_49f3b0ca9b57_out_string_crlf_copied_with_its_nul
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=7 pc=0xc670f120 address=0xc67520b0 in mch_write_probe+0x58 (the labelled probe)  404b377593ab559b
    06_212c3820c7bb_key_hash_filter_tag_length_underflow
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=5 pc=0xc670f2d8 address=0xc6752260 in mch_read_probe+0x38 (the labelled probe)  01f07c71e945ea16
    07_0f605245cf3f_bin_delete_logs_unterminated_key
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=5 pc=0xc670f200 address=0xc6752160 in mch_read_probe+0x38 (the labelled probe)  eeee96d8119e1b84
    08_fa51ad8452d5_slab_list_shuffle_reads_one_past
        CAUGHT  predicted CAUGHT  held  fixed: FIXED exit=0
        cause=5 pc=0xc670f248 address=0xc67521c0 in mch_read_probe+0x38 (the labelled probe)  842b3ec9ef91aee9
