# native-detect (ASan) and the fix differential, all 25 cases -- 2026-10-10

    bash capstone/bug-corpora/ffmpeg/plain-heap-repros/runners/run-native.sh <fresh outdir>

The arm's only earlier record (`../20261006-native-plain-heap/`) covered cases 0-3, from when the corpus
had four; cases 4-24 had `status: measured` with no record. This run covers all 25. The compiler is the
first line of `run-native.txt`.

**Result: 25 of 25 two-sided.** Every buggy arm DEFECT-REPRODUCED with an ASan `heap-buffer-overflow`
whose frame #0 is the case's labelled probe (`ffh_read_probe`, `ffh_read_probe_u8` or
`ffh_write_probe_u8`, `at_probe=1`), and every fixed arm VERDICT FIXED with ASan silent. The runner now
records the report's first two frames rather than only grepping for the report's name.

Case 13 is a duplicate of case 2 (the release/9.0 backport of the same fix, `git patch-id`
`03b9184e7cc8` for both); its line is a reading of the same defect, not a second one.
