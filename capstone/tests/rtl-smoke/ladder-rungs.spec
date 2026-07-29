# Silicon-ladder rung table — THE single source of truth for both halves.
#
# Format:  <rung>:<kernel header>:<compute fn>:<-O level>[:<domain knobs>]
#
# Field 5 is OPTIONAL: space-separated KEY=VALUE env assignments applied to the
# CAPABILITY half's domain build only (build-ladder-fpga.sh). The baseline half
# discards it -- it is plain riscv64 with no cap-table glue for the knobs to affect.
# Two knobs are in use:
#   DOMAIN_WINDOW=32k     32 KiB code window instead of 4 KiB (issue C-5)
#   LADDER_NO_RO_COPY=1   unrolled li/sd initializer instead of the large-RO COPY
#                         path, which is broken (issue C-4b)
# They are per-rung and NOT global on purpose: changing the window changes image
# layout, and a 2026-07-26 A/B showed four added instructions flipping a passing rung.
# Every already-measured rung keeps its published number by staying at the default.
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
# rv8_sha512 WORKS as of 2026-07-28 but needs TWO opt-in knobs, which is why it sat
# QEMU-verified-but-unmeasured: a whole-set sweep could apply them to every rung
# (perturbing measured rows) or to none (this rung then builds at 4 KiB with the
# broken copy path and fails). Field 5 resolves that -- the knobs now travel WITH the
# rung, so a plain `build-ladder-fpga.sh` sweep builds it correctly.
# Verified: returns its oracle 1390718314, static gate cjalr=0 ldc-gp=3.
# Historic note (the two knobs exist because):
#   80-entry K table (640 B): fails at domain CREATION, QEMU asserts in helper_cssplit
#     (`rs1_v->tag && !rs2_v->tag`), loadable size 5088.
#   16-entry K table (128 B): domain creates, then a global access faults OOB --
#     cursor 0x101561000 against bounds (0x10157ffd0, 0x101580000).
#   Uncomment once C-4 is fixed; the kernel and oracle (1390718314) are ready.
# The BASELINE half is plain riscv64 with no code window and no cap-table glue, so it
# builds and runs unconditionally; only the CAPABILITY half needs the two knobs.
# KNOBS DROPPED 2026-07-28: C-4b is fixed, so the large-RO COPY path works and the
# 640 B K table no longer needs the 32 KiB window or the unrolled-path bypass. Re-gated
# at the DEFAULT window with no knobs: oracle 1390718314, cjalr=0 ldc-gp=3, copy-path=yes.
rv8_sha512:rv8_sha512_kernel.h:sha512_compute:-O1
# Selected against R-1's shape (no arrays at all), not for benchmark prestige --
# 6 of 6 silicon attempts today were predicted PASS and only 2 were.
beebs_expint:beebs_expint_kernel.h:expint_compute:-O1
# R-1 reproducer kept as a rung: a real crypto kernel that hangs at the DEFAULT
# 4 KiB window with no special flags. Baseline half builds and runs fine.
rv8_sha512s:rv8_sha512s_kernel.h:sha512s_compute:-O1
# Screened on shape over the WHOLE kernel: no initialized arrays (avoids C-4/C-5) and
# no array stores in any loop (avoids R-1's required shape). Control-flow dominated --
# 180 switch dispatches per call, a profile nothing else in the ladder has.
beebs_cover:beebs_cover_kernel.h:cover_compute:-O1
# Added 2026-07-28. These two are a deliberate PAIR, chosen to bracket the mechanism
# claim (overhead is a property of DATA ACCESS, not execution) from both ends in one
# board session rather than to add benchmark names:
#   aha_mont64 -- 24 B of globals, no arrays, no tables. Pure 64-bit multiply/shift
#                 carry chains: modul64 is a 64-iteration shift-and-subtract, xbinGCD
#                 a 64-iteration binary GCD. If the claim holds this lands near 1.00x
#                 for a completely different execution profile than beebs_cover, which
#                 is the other near-1.00x rung (and is control-flow, not arithmetic).
#   ns         -- almost nothing BUT memory: 500 four-level indexed read-only loads
#                 per call with essentially no arithmetic on the loaded value. The
#                 heaviest data-access profile in the ladder.
# Both QEMU-gated 2026-07-28 at -O1: mont64 retval 2185097489 (cjalr=0 ldc-gp=1),
# ns retval 1184999093 (cjalr=0 ldc-gp=2).
beebs_aha_mont64:beebs_aha_mont64_kernel.h:mont_compute:-O1
# KNOBS DROPPED 2026-07-28, same reason. Both 2,000 B tables are copy-eligible
# (file-scope symbol, size%8==0), so they take the copy path at the DEFAULT window:
# oracle 1184999093, cjalr=0 ldc-gp=2, copy-path=yes.
# WORTH RE-RUNNING ON THE BOARD: R-9 recorded this rung hanging, and the hypothesis on
# file was prologue SCALE -- ~500 words stored through the carving capability. The copy
# path replaces that with a 6-instruction loop, so the silicon binary is now a completely
# different shape. If prologue scale was the cause, this should now pass.
beebs_ns:beebs_ns_kernel.h:ns_compute:-O1
# R-9 DISCRIMINATORS (2026-07-28). beebs_ns hangs on silicon, R-1 does not predict it
# (neither table is ever written), and the prologue-scale hypothesis is refuted. Each
# of these changes exactly ONE property of the kernel so a board run says which one
# matters. Data is byte-identical to beebs_ns where present; all three QEMU-gated at -O1.
#   nskeys  -- reads ONE table, never a second. Is touching two distinct cap-table
#              globals in the same loop the trigger?           oracle 3914083333
#   nsflat  -- same 500 elements, FLAT, one index level. Is 4-level nested address
#              arithmetic the trigger?                          oracle 1184999093
#   nssmall -- 125 entries, same 4-level shape. The size test pre-registered under R-9.
#                                                               oracle 2711842293
# Run them in ONE boot with beebs_ns itself as the in-boot control.
beebs_nskeys:beebs_nskeys_kernel.h:nskeys_compute:-O1
beebs_nsflat:beebs_nsflat_kernel.h:nsflat_compute:-O1
beebs_nssmall:beebs_nssmall_kernel.h:nssmall_compute:-O1

# WINDOW CLIMB (2026-07-29). C-5 is validated on silicon only to 32 KiB; SQLite needs a
# 1.3 MB PCC and its first board run hung. Same kernel as beebs_prime (known-good,
# oracle 582955588) at larger code windows, to find where the window stops working
# before blaming SQLite for it.
beebs_prime256k:beebs_prime_kernel.h:prime_compute:-O1:DOMAIN_WINDOW=0x40000
beebs_prime1m:beebs_prime_kernel.h:prime_compute:-O1:DOMAIN_WINDOW=0x100000

# REPRODUCIBILITY SAMPLES (2026-07-29). Four IDENTICAL copies of beebs_prime, so one
# boot yields four pass/fail samples instead of one. C-13's bisection was invalidated by
# assuming the silicon failure was deterministic without checking: stage 2 passed once
# and failed on a re-run with no change at all. Nothing can be attributed until the
# failure RATE is known. They differ only in entry VA, an axis already validated as
# measurement-safe when R-3 was found address-keyed (0.03% across boot positions).
beebs_primer1:beebs_prime_kernel.h:prime_compute:-O1:INTERP_FAKE_COUNT=1 INTERP_DIAG_STAGE=7
beebs_primer2:beebs_prime_kernel.h:prime_compute:-O1:INTERP_FAKE_COUNT=1 INTERP_DIAG_STAGE=8
beebs_primer3:beebs_prime_kernel.h:prime_compute:-O1:INTERP_FAKE_COUNT=1 INTERP_DIAG_STAGE=8
beebs_primer4:beebs_prime_kernel.h:prime_compute:-O1:INTERP_FAKE_COUNT=1 INTERP_DIAG_STAGE=9
# DESCRIPTOR-STRESS rung (2026-07-29). Smallest domain exercising EVERY glue path
# SQLite needs: zero-fill, bulk copy, byte tail, a >2040 B global, and a private .L
# symbol -- none of which beebs_prime (1 zero-init global at -O1) reaches. Oracle
# 43662404; descriptor is 6 records. Bridges the 1-global bisection to SQLite's 1,059.
gpstress:gpstress_kernel.h:gpstress_compute:-O1
# BLOB-PEEK PROBE (2026-07-29). Returns the 64-bit word the DOMAIN actually reads from
# the monitor-copied blob at offset 8 -- the word stage 8 uses as `count`. Every C-13 step
# so far INFERRED the blob contents; this observes them. retval 1 = copy correct;
# retval 0 = blob zeroed/absent (fix the copy DESTINATION); anything else names the
# corruption. Requires INTERP_FAKE_COUNT=1 INTERP_DIAG_STAGE=11.
blobpeek:blobpeek_kernel.h:bp_compute:-O1:INTERP_FAKE_COUNT=1 INTERP_DIAG_STAGE=11
# Descriptor-header MAP: same probe, different offsets, so one boot shows whether ANY
# of the blob is present. Expected on a correct copy: +0=0, +8=1, +32=8, +48=-1.
# All zero across every offset => the blob is simply not there.
blobpeek0:blobpeek_kernel.h:bp_compute:-O1:INTERP_FAKE_COUNT=1 INTERP_DIAG_STAGE=11 INTERP_PEEK_OFF=0
blobpeek32:blobpeek_kernel.h:bp_compute:-O1:INTERP_FAKE_COUNT=1 INTERP_DIAG_STAGE=11 INTERP_PEEK_OFF=32
blobpeek48:blobpeek_kernel.h:bp_compute:-O1:INTERP_FAKE_COUNT=1 INTERP_DIAG_STAGE=11 INTERP_PEEK_OFF=48
