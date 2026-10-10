# native-detect (ASan) and the fix differential, all 9 cases -- 2026-10-10

    bash capstone/bug-corpora/memcached/plain-heap-repros/runners/run-native.sh <fresh outdir>

The only earlier per-case record (`../20261006-native-plain-heap/`) was case 0's; cases 1-8 read
"caught" through the table generator's ASan-text fallback. **Result: 9 of 9 two-sided**, each buggy
arm's report at the labelled probe (`mch_write_probe` for 0-5, `mch_read_probe` for 6-8, `at_probe=1`),
each fixed arm silent. Cases 6-8 are READ defects: this is also the attribution their CheriBSD
records lacked.
