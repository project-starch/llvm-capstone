# memcached/allocator-repros under AddressSanitizer -- 2026-10-09

Predictions committed before the run: 3ef5169ae90b. Runner: tools/run-native-asan.py, built by
runners/run-asan.sh (the port library itself carries the sanitizer).

Build: cc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0; -fsanitize=address -fno-omit-frame-pointer -O1 -g; port library with the same sanitizer
ASAN_OPTIONS: detect_leaks=0:abort_on_error=0:halt_on_error=1:color=never

## Controls

    asan-control past 67108864: heap-buffer-overflow (required heap-buffer-overflow)
    asan-control uaf 67108864: heap-use-after-free (required heap-use-after-free)

## Cases

    00_7af02b0c87_rbuf_copied_after_cache_free                     fixed rc=0  buggy SILENT    the defect reproduced (VERDICT DEFECT-REPRODUCED) and ASan printed nothing
    01_0ad4de66ae_io_walk_reads_freed_link                         fixed rc=0  buggy SILENT    the defect reproduced (VERDICT DEFECT-REPRODUCED) and ASan printed nothing
    02_59bd02ce29_tail_repair_frees_referenced_item                fixed rc=0  buggy SILENT    the defect reproduced (VERDICT DEFECT-REPRODUCED) and ASan printed nothing
    03_a8c4a82787_refcount_overflow_frees_linked_item              fixed rc=0  buggy SILENT    the defect reproduced (VERDICT DEFECT-REPRODUCED) and ASan printed nothing
    04_152ddb68f7_unlocked_refcount_drift                          fixed rc=0  buggy SILENT    the defect reproduced (VERDICT DEFECT-REPRODUCED) and ASan printed nothing
    05_2d61f18_item_data_one_past                                  fixed rc=0  buggy SILENT    the defect reproduced (VERDICT DEFECT-REPRODUCED) and ASan printed nothing
    06_78eb770_suffix_write_no_space                               fixed rc=0  buggy SILENT    the defect reproduced (VERDICT DEFECT-REPRODUCED) and ASan printed nothing
    07_ecdb011_unterminated_key_read                               fixed rc=0  buggy SILENT    the defect reproduced (VERDICT DEFECT-REPRODUCED) and ASan printed nothing
    08_e8364b5_ascii_all_spaces_scan_past_rbuf                     fixed rc=0  buggy SILENT    the defect reproduced (VERDICT DEFECT-REPRODUCED) and ASan printed nothing
