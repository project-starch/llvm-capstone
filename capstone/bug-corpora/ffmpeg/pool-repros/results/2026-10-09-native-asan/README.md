# ffmpeg/pool-repros under AddressSanitizer -- 2026-10-09

Predictions committed before the run: 3ef5169ae90b. Runner: tools/run-native-asan.py, built by
runners/run-asan.sh (the port library itself carries the sanitizer).

Build: cc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0; -fsanitize=address -fno-omit-frame-pointer -g -O0; port library built with the same flags
ASAN_OPTIONS: detect_leaks=0:abort_on_error=0:halt_on_error=1:color=never

## Controls

    asan-control past 67108864: heap-buffer-overflow (required heap-buffer-overflow)
    asan-control uaf 67108864: heap-use-after-free (required heap-use-after-free)

## Cases

    00_461fb22053_af_join_dedup_bound                              fixed rc=0  buggy SILENT    the defect reproduced (VERDICT DEFECT-REPRODUCED) and ASan printed nothing
    01_1886c3269d_h264_refs_partial_clear                          fixed rc=0  buggy SILENT    the defect reproduced (VERDICT DEFECT-REPRODUCED) and ASan printed nothing
    02_316531e61c_vidstab_parked_plane_pointer                     fixed rc=0  buggy SILENT    the defect reproduced (VERDICT DEFECT-REPRODUCED) and ASan printed nothing
    03_5c66a3ab51_vvc_nonref_output_releases_tabs                  fixed rc=0  buggy SILENT    the defect reproduced (VERDICT DEFECT-REPRODUCED) and ASan printed nothing
