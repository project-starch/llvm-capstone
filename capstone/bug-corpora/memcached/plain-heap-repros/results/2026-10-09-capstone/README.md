# memcached/plain-heap-repros on the Capstone application domain arms -- 2026-10-09

Predictions committed before the run: 60f1c5be25a5 (tools/run-capstone-domain.py, predicted()).

Reproduce (one VM up, see ports/common/application/README.md):

    python3 capstone/bug-corpora/tools/run-capstone-domain.py \
      --corpus capstone/bug-corpora/memcached/plain-heap-repros --arm spatial|sublet \
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
    00_ddee3e2_authfile_scan_past_calloc                               fixed FIXED exit=0   buggy CAUGHT             as predicted cause=7 pc=0xc020f040 address=0xc0251f99 in mch_write_probe+0x58 (the labelled probe)  image 36e178db06b6419c
    01_d5d9ff0_cachedump_end_marker_off_by_one                         fixed FIXED exit=0   buggy CAUGHT             as predicted cause=7 pc=0xc020f14c address=0xc0252140 in mch_write_probe+0x58 (the labelled probe)  image 767e14f155870752
    02_391f2e4762bf_freesuffix_realloc_sized_in_bytes                  fixed FIXED exit=0   buggy CAUGHT             as predicted cause=7 pc=0xc020f76c address=0xc0252730 in mch_write_probe+0x58 (the labelled probe)  image 73177c741354eab3
    03_16a809e2a062_cache_create_freelist_sized_by_object              fixed FIXED exit=0   buggy CAUGHT             as predicted cause=7 pc=0xc020f014 address=0xc0252000 in mch_write_probe+0x58 (the labelled probe)  image 890befd830fa5991
    04_40aff8b0f113_stats_end_marker_past_exact_buffer                 fixed FIXED exit=0   buggy CAUGHT             as predicted cause=7 pc=0xc020efc4 address=0xc0251fb0 in mch_write_probe+0x58 (the labelled probe)  image 911b366cd9ad1f18
    05_49f3b0ca9b57_out_string_crlf_copied_with_its_nul                fixed FIXED exit=0   buggy CAUGHT             as predicted cause=7 pc=0xc020f120 address=0xc02520b0 in mch_write_probe+0x58 (the labelled probe)  image 404b377593ab559b
    06_212c3820c7bb_key_hash_filter_tag_length_underflow               fixed FIXED exit=0   buggy CAUGHT             as predicted cause=5 pc=0xc020f2d8 address=0xc0252260 in mch_read_probe+0x38 (the labelled probe)  image 01f07c71e945ea16
    07_0f605245cf3f_bin_delete_logs_unterminated_key                   fixed FIXED exit=0   buggy CAUGHT             as predicted cause=5 pc=0xc020f200 address=0xc0252160 in mch_read_probe+0x38 (the labelled probe)  image eeee96d8119e1b84
    08_fa51ad8452d5_slab_list_shuffle_reads_one_past                   fixed FIXED exit=0   buggy CAUGHT             as predicted cause=5 pc=0xc020f248 address=0xc02521c0 in mch_read_probe+0x38 (the labelled probe)  image 842b3ec9ef91aee9

## sublet

    platform  Capstone application VM on capstone-qemu 32e7c9754f9a, kernel e58613598c89, firmware 6f2b082cb677, capstone-exec 0752aa7c49c9, module a88ed2159b43; SDK heap=sublet runtime b73508e5005b; compiler 7d01722aab88
    env       {'CAPSTONE_GP_NONLIN': '1', 'CAPSTONE_REV_NODES': '65536'}
    tool      0383d3c2caa89fb6   predicted buggy outcome: CAUGHT
    control clean RETURNED  required RETURNED  CONTROL clean RETURNED 1
    control oob   CAUGHT    required CAUGHT    cause=7 pc=0xc020f70c address=0xc8802018 in ctl_write_probe+0x58 (the labelled probe)
    control uaf   CAUGHT    required CAUGHT    cause=24 pc=0xc020f758 address=0x0 in ctl_read_probe+0x38 (the labelled probe)
    00_ddee3e2_authfile_scan_past_calloc                               fixed FIXED exit=0   buggy CAUGHT             as predicted cause=7 pc=0xc020ff44 address=0xc8802009 in mch_write_probe+0x58 (the labelled probe)  image 7683a40580ae6ee1
    01_d5d9ff0_cachedump_end_marker_off_by_one                         fixed FIXED exit=0   buggy CAUGHT             as predicted cause=7 pc=0xc02102a0 address=0xc8802040 in mch_write_probe+0x58 (the labelled probe)  image 4db6d3d77c3b8887
    02_391f2e4762bf_freesuffix_realloc_sized_in_bytes                  fixed FIXED exit=0   buggy CAUGHT             as predicted cause=7 pc=0xc0210948 address=0xc8802110 in mch_write_probe+0x58 (the labelled probe)  image 690fa1571975c3de
    03_16a809e2a062_cache_create_freelist_sized_by_object              fixed FIXED exit=0   buggy CAUGHT             as predicted cause=7 pc=0xc020ff18 address=0xc8802040 in mch_write_probe+0x58 (the labelled probe)  image 9912709864d24957
    04_40aff8b0f113_stats_end_marker_past_exact_buffer                 fixed FIXED exit=0   buggy CAUGHT             as predicted cause=7 pc=0xc020fec8 address=0xc8802040 in mch_write_probe+0x58 (the labelled probe)  image d9ab911a43ae6dce
    05_49f3b0ca9b57_out_string_crlf_copied_with_its_nul                fixed FIXED exit=0   buggy CAUGHT             as predicted cause=7 pc=0xc0210024 address=0xc8802010 in mch_write_probe+0x58 (the labelled probe)  image 066981ea112faaa5
    06_212c3820c7bb_key_hash_filter_tag_length_underflow               fixed FIXED exit=0   buggy CAUGHT             as predicted cause=5 pc=0xc02101dc address=0xc8802010 in mch_read_probe+0x38 (the labelled probe)  image 150210a9f4758ee5
    07_0f605245cf3f_bin_delete_logs_unterminated_key                   fixed FIXED exit=0   buggy CAUGHT             as predicted cause=5 pc=0xc0210104 address=0xc8802010 in mch_read_probe+0x38 (the labelled probe)  image e446982140464e5b
    08_fa51ad8452d5_slab_list_shuffle_reads_one_past                   fixed FIXED exit=0   buggy CAUGHT             as predicted cause=5 pc=0xc021014c address=0xc8802020 in mch_read_probe+0x38 (the labelled probe)  image d3062a17282c95c6

