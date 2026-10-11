# memcached/plain-heap-repros, arm `cheribsd-subobject` -- 2026-10-09

Platform: stock CheriBSD purecap, kernel CHERI-PURECAP-QEMU, revocation on, every case built with -Xclang -cheri-bounds=subobject-safe (booted on the second host).
Predictions committed before the run: 886abf4cc336. Runtime revocation reported by the ABI control: 1.

Result LINES only; the boot capture is contaminated by construction.

## Controls, from the same boot

    cheribsd-abi: CHERI_ABI pointer_bytes=16 runtime_revocation=1
    cheribsd-bounds: CHERI_BOUNDARY_READY
    subobj-control: exit 162

## Cases: {'NOT CAUGHT': 1, 'CAUGHT': 8}

    00_ddee3e2_authfile_scan_past_calloc
        fixed: exit 0 VERDICT FIXED
        buggy: exit 0 -> NOT CAUGHT
        VERDICT DEFECT-REPRODUCED fgets wrote its terminating NUL at offset sb.st_size of a calloc(1, sb.st_size), one byte past the allocation
    01_d5d9ff0_cachedump_end_marker_off_by_one
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x1024bc INSIDE mch_write_probe [0x1024a4, 0x1024c8)
        SUPERVISE fault signal=34 code=1 addr=0x1024bc pc=0x1024bc
    02_391f2e4762bf_freesuffix_realloc_sized_in_bytes
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x1024ca INSIDE mch_write_probe [0x1024b2, 0x1024d6)
        SUPERVISE fault signal=34 code=1 addr=0x1024ca pc=0x1024ca
    03_16a809e2a062_cache_create_freelist_sized_by_object
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x1023e4 INSIDE mch_write_probe [0x1023cc, 0x1023f0)
        SUPERVISE fault signal=34 code=1 addr=0x1023e4 pc=0x1023e4
    04_40aff8b0f113_stats_end_marker_past_exact_buffer
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x10242c INSIDE mch_write_probe [0x102414, 0x102438)
        SUPERVISE fault signal=34 code=1 addr=0x10242c pc=0x10242c
    05_49f3b0ca9b57_out_string_crlf_copied_with_its_nul
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x102460 INSIDE mch_write_probe [0x102448, 0x10246c)
        SUPERVISE fault signal=34 code=1 addr=0x102460 pc=0x102460
    06_212c3820c7bb_key_hash_filter_tag_length_underflow
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x1025ec OUTSIDE mch_write_probe [0x1025b8, 0x1025dc)
        SUPERVISE fault signal=34 code=1 addr=0x1025ec pc=0x1025ec
    07_0f605245cf3f_bin_delete_logs_unterminated_key
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x102470 OUTSIDE mch_write_probe [0x10243c, 0x102460)
        SUPERVISE fault signal=34 code=1 addr=0x102470 pc=0x102470
    08_fa51ad8452d5_slab_list_shuffle_reads_one_past
        fixed: exit 0 VERDICT FIXED
        buggy: exit 162 -> CAUGHT; fault pc 0x102494 OUTSIDE mch_write_probe [0x102460, 0x102484)
        SUPERVISE fault signal=34 code=1 addr=0x102494 pc=0x102494
