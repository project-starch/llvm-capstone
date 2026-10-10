# ffmpeg/plane-repros on cheribsd-carve-bounds, the buggy arm supervised -- 2026-10-10 (R3b)

Pre-registered at `49e8e23f4122` (docs/history/10-10-2026_23-30-00_whole-corpus-audit-cross-program.md, R3b).
As predicted: the fixed arm completes (VERDICT FIXED, exit 0) and the buggy arm faults, SIGPROT si_code 1,
at `ffp_read_probe` -- now with the pc, which the 2026-10-09 run did not keep.

    CHERI_EXTRA_CFLAGS=-DFFP_CARVE_BOUNDS bash capstone/bug-corpora/ffmpeg/plane-repros/runners/run-cheribsd.sh <out>

Platform (stock CheriBSD, revocation on): qemu 16135483052dfdd6, firmware f0e1fe57b0f85075, kernel 8ab453f46dc76cf2, libc fdce2289224bb519, image 0cb16209c16c5edf.

| arm | exit | |
|---|---|---|
| cheribsd-abi | 0 | as predicted |
| cheribsd-bounds | 162 | as predicted |
| ffp-00-fixed | 0 | as predicted |
| ffp-00-buggy | 162 | as predicted |

Program sha256: ffp-00 5a9747b24d03a737

`attribution.tsv` beside this file is tools/attribute-cheribsd-faults.py's table for the buggy arm.
