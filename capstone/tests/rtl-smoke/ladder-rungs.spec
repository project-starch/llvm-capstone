# Silicon-ladder rung table — THE single source of truth for both halves.
#
# Format:  <rung>:<kernel header>:<compute fn>:<-O level>
#
# WHY THIS FILE EXISTS. The capability half and the baseline half of every overhead
# ratio must be built at the SAME -O level, or the ratio measures optimisation
# rather than capabilities. They used to be specified in two different places and
# two different ways: build-ladder-base-fpga.sh carried a per-rung table, while
# build-ladder-fpga.sh had a single global default overridden wholesale by
# LADDER_OPT. The consequence, found by audit on 2026-07-28, is that **no single
# invocation paired correctly for every rung** -- LADDER_OPT=-O1 mismatched the six
# -O0 baselines, and omitting it mismatched the eight -O1 ones. A full sweep was
# not runnable in one go without a silent mismatch somewhere.
#
# That is not a hypothetical failure. Two instances of it were caught the hard way:
#   * 2026-07-27 (I-1): a sweep rebuilt at -O0 against -O1 baselines, producing five
#     bogus "silicon failures" and a false refutation of R-1 that was nearly sent to
#     the board owner.
#   * 2026-07-28: beebs_recursion was listed -O0 in the baseline spec while its own
#     comment three lines below said it "has to be -O1", and the published pair is
#     -O1. The -O0 baseline returns roughly double.
#
# Both scripts now read THIS file, so the two halves cannot drift apart. The runner
# additionally cross-checks the recorded levels and hard-fails on a mismatch
# (optlevels.txt); that guard stays as a backstop, not as the primary defence.
#
# LADDER_OPT still overrides every rung, for deliberate A/B sweeps at one level.
#
# Choosing a level: -O0 is the historical default. Rungs are moved to -O1 when that
# is the level at which they compute correctly on silicon (beebs_recursion) or the
# level they were introduced and measured at (beebs_bs and later). coremark_matrix
# is -Os because it overflows the domain's 4 KiB PCC window (C-5) at -O0.

# UNIFORM -O1 (2026-07-28). Every rung is now -O1 except coremark_matrix, which needs
# -Os to fit the 4 KiB PCC window (C-5). Previously the table mixed -O0 and -O1 rows:
# each PAIR was internally consistent, so each ratio was valid, but comparing ACROSS
# rows mixed optimisation levels -- a reviewer's first question. All five measurable
# rungs were verified to pass the QEMU parity leg at -O1 before this change.

null:ladder_base_null_kernel.h:null_compute:-O1
matmult_int:matmult_int_kernel.h:mm_compute:-O1
coremark_matrix:coremark_matrix_kernel.h:coremark_matrix_compute:-Os
rv8_primes:rv8_primes_kernel.h:primes_compute:-O1
beebs_crc32:beebs_crc32_kernel.h:crc_compute:-O1
beebs_insertsort:beebs_insertsort_kernel.h:is_compute:-O1
beebs_prime:beebs_prime_kernel.h:prime_compute:-O1
beebs_recursion:beebs_recursion_kernel.h:rec_compute:-O1
beebs_bs:beebs_bs_kernel.h:bs_compute:-O1
beebs_janne:beebs_janne_kernel.h:jc_compute:-O1
beebs_fibcall:beebs_fibcall_kernel.h:fibcall_compute:-O1
beebs_fac:beebs_fac_kernel.h:fac_compute:-O1
beebs_cnt:beebs_cnt_kernel.h:cnt_compute:-O1
beebs_duff:beebs_duff_kernel.h:duff_compute:-O1
ctrsanity:ctrsanity_kernel.h:cs_compute:-O1
ctrsanity4:ctrsanity4_kernel.h:cs_compute:-O1
# Crypto/bitwise profile, added 2026-07-28. The rest of the RV8 set is blocked by
# known issues, not by this measurement: aes ~8 KB of tables (C-4/C-5), dhrystone 684
# lines (C-5), qsort sorts in place (R-1), miniz mixed-extend i128 logic (C-2).
# rv8_sha512 WORKS as of 2026-07-28, but needs TWO opt-in knobs (see below), so it is
# not enabled by the default build path yet -- the runner would build it at 4 KiB with
# the broken copy path. Build it with:
#   DOMAIN_WINDOW=32k LADDER_NO_RO_COPY=1 DOMAIN_OPT_LEVEL=-O1
# Verified: returns its oracle 1390718314, static gate cjalr=0 ldc-gp=3.
# Historic note (the two knobs exist because):
#   80-entry K table (640 B): fails at domain CREATION, QEMU asserts in helper_cssplit
#     (`rs1_v->tag && !rs2_v->tag`), loadable size 5088.
#   16-entry K table (128 B): domain creates, then a global access faults OOB --
#     cursor 0x101561000 against bounds (0x10157ffd0, 0x101580000).
#   Uncomment once C-4 is fixed; the kernel and oracle (1390718314) are ready.
# rv8_sha512:rv8_sha512_kernel.h:sha512_compute:-O1
