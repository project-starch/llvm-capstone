# native-detect (ASan) and the fix differential, all 12 cases -- 2026-10-10

    bash capstone/bug-corpora/wireshark/plain-heap-repros/runners/run-native.sh <fresh outdir>

Case 0 had the only earlier per-case record (`../20261006-native-plain-heap/`); 1-11 rested on an
aggregate in a lane commit message. **Result: 12 of 12 two-sided**, each buggy report at the labelled
probe (`wsh_read_probe` for 0-9, `wsh_write_probe` for 10-11, `at_probe=1`), each fixed arm silent.
