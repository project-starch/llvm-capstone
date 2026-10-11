# memcached/plain-heap-repros, arm `poisoncap-spatial` -- 2026-10-09

Platform: PoisonCap mode 0: the rebuilt published PoisonCap platform (CHERI-PURECAP-QEMU-POISON kernel, libc with the mrs poison-retire fix), runtime revocation OFF (booted on the second host).
Predictions committed before the run: the corpus's committed predictions and 2852edcdd747 (size-class slack). Runtime revocation reported by the ABI control: 0.

Result LINES only; the boot capture is contaminated by construction.

## Controls, from the same boot

    cheribsd-abi: CHERI_ABI pointer_bytes=16 runtime_revocation=0
    cheribsd-bounds: CHERI_BOUNDARY_READY

## Cases: {'NOT CAUGHT': 1, 'CAUGHT': 8}

    00_ddee3e2_authfile_scan_past_calloc
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED fgets wrote its terminating NUL at offset sb.st_size of a calloc(1, sb.st_size), one byte past the allocation
    01_d5d9ff0_cachedump_end_marker_off_by_one
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x102400 INSIDE mch_write_probe [0x1023e8, 0x10240c)
        SUPERVISE fault signal=34 code=1 addr=0x102400 pc=0x102400
    02_391f2e4762bf_freesuffix_realloc_sized_in_bytes
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x10240e INSIDE mch_write_probe [0x1023f6, 0x10241a)
        SUPERVISE fault signal=34 code=1 addr=0x10240e pc=0x10240e
    03_16a809e2a062_cache_create_freelist_sized_by_object
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x102328 INSIDE mch_write_probe [0x102310, 0x102334)
        SUPERVISE fault signal=34 code=1 addr=0x102328 pc=0x102328
    04_40aff8b0f113_stats_end_marker_past_exact_buffer
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x102370 INSIDE mch_write_probe [0x102358, 0x10237c)
        SUPERVISE fault signal=34 code=1 addr=0x102370 pc=0x102370
    05_49f3b0ca9b57_out_string_crlf_copied_with_its_nul
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x1023a4 INSIDE mch_write_probe [0x10238c, 0x1023b0)
        SUPERVISE fault signal=34 code=1 addr=0x1023a4 pc=0x1023a4
    06_212c3820c7bb_key_hash_filter_tag_length_underflow
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x10252c OUTSIDE mch_write_probe [0x1024f8, 0x10251c)
        SUPERVISE fault signal=34 code=1 addr=0x10252c pc=0x10252c
    07_0f605245cf3f_bin_delete_logs_unterminated_key
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x1023b4 OUTSIDE mch_write_probe [0x102380, 0x1023a4)
        SUPERVISE fault signal=34 code=1 addr=0x1023b4 pc=0x1023b4
    08_fa51ad8452d5_slab_list_shuffle_reads_one_past
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x1023d8 OUTSIDE mch_write_probe [0x1023a4, 0x1023c8)
        SUPERVISE fault signal=34 code=1 addr=0x1023d8 pc=0x1023d8
