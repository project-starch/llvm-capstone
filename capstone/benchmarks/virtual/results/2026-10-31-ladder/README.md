# The microbenchmark ladder on vp2 silicon: Plain, Caps, ObjBounds, MapLife, ObjLife (2026-10-30/31)

All five configurations of the ladder on the 103 microbenchmarks (BEEBS 80 + 13 -libc, RV8 7, CoreMark, two controls).

| configuration | what it is | where it ran |
|---|---|---|
| Plain | the programs as plain RISC-V Linux processes, user-mode rdcycle/rdinstret | board 1, bitstream 3, kits p1..p6 (arms Plain, T1, T3) |
| Caps | allocator returns the group's authority, no per-object bounds (`CAPSTONE_HEAP_OBJECT_BOUNDS=0`) | board 1, bitstream 3, kits capsh1..3, capsa1..4 (Caps, MapLife, ObjLife) |
| ObjBounds | MapLife's images on bitstream 3 without the per-access lifetime check (capstone-ariane objbounds-b 603528b8b) | board 2, kits objbh1..3, objba1..4 (T1 = ObjBounds, T3 = ObjLife-v) |
| MapLife | per-object bounds, lifetime per mapping, hardware check on every access (T1) | bitstream-3 records `../2026-10-27-t3all-bitstream3`, `../2026-10-28-t3heap-bitstream3`, and inside every Plain and Caps boot |
| ObjLife | MapLife + a lifetime per malloc object (T3) | same |

Pre-registrations: `PREREG-t1-t2-sampling.md` sections "Plain" (perf-t0b 3e90e15c4ffc) with Plain addendum 1 (f2dd520270d6),
"Caps" (perf-t0a), "ObjBounds" (perf-objb afd4dacd48bb), all pushed before their boots. Analysis script `../../ladder.py`
(perf-objb bb355d87056d), committed before any cross-bitstream ratio was read.

## Validity

Every boot VALID by its own gate (`gates.txt`): Plain p1..p6 (249, 249, 249, 237, 228, 228 processes), Caps 7 of 7,
ObjBounds 7 of 7. The ObjBounds variant was confirmed on board 2 by the RTL lane's bare kit (CONFIRMED 10/10) and its
witness (WITNESSED: revocations take effect on the variant, so its zeros are the removed check).

Plain, addendum 1, implemented by `plain-exclude.py` and `plain-correct.py --k-ins-median` (perf-t0b 1cb365cae357):
- iterations with timer > 1.1 x median + 2 are dropped in all three arms (24, 33, 29, 45, 39, 49 on p1..p6);
- 23 of 101 programs had more than 2 of 5 iterations dropped in one process and are set aside from every Plain ratio
  (`plain/excl-plain-p*.txt`);
- k_ins 3,958..3,986 per interrupt (probe 3,817); P4 PASS on p1, p2, p4, p6. p5 FAILs on one ctl-loop iteration 15 %
  off. p3 FAILs because its probe was disturbed (0.94 M spin iterations against ~2.99 M, 89 gaps for 202 interrupts),
  so p3 takes k_cyc = 23,050, the other boots' probe median, flagged in `plain/xc-plain-p3.report`.

## Readings (`main/per-group.tsv`, geomeans; `main/per-program.tsv`, `main/ladder.tsv`)

| group (programs) | MapLife/Caps | ObjLife/MapLife | MapLife/ObjBounds | ObjLife/ObjBounds | ObjLife/Plain (raw) |
|---|---|---|---|---|---|
| BEEBS upstream (80) | 0.9996 | 1.0001 | 1.0992 | 1.0993 | 1.554 (1.408), 61 |
| BEEBS -libc (13) | 1.0032 | 1.0273 | 1.0604 | 1.0893 | 3.333 (2.916), 11 |
| RV8 (7) | 0.9998 | 0.9999 | 1.0438 | 1.0437 | 1.296 (1.105), 5 |
| CoreMark | 0.9997 | 1.0004 | 1.2743 | 1.2748 | 2.061 (1.868) |
| allocation-heavy (8) | 1.0045 | 1.0440 | 1.0509 | 1.0972 | 7.38 (6.63), 6 |

Suite intervals (two-level bootstrap) are within ±0.3 points; ObjLife/Plain's within ±0.6 %. ObjLife/Plain covers the
programs Plain kept (count after the reading). Ladder bars normalized to Plain (corrected), same programs:
BEEBS upstream Caps 1.555, ObjBounds 1.427, MapLife 1.554, ObjLife 1.554; -libc 3.251, 3.069, 3.261, 3.333; RV8 1.295,
1.234, 1.295, 1.296; CoreMark 2.062, 1.618, 2.061, 2.061. Caps ran on bitstream 3, so its bar includes the hardware check.

## Against the pre-registered predictions

ObjBounds:
- P1 correctness: every process rc 0 (the gates).
- P2 instret: holds. Within 0.1 % on 101 of 101 programs; the largest deviation is 0.02 %.
- P3 MapLife/ObjBounds bands: MISS. BEEBS upstream 1.099 against [1.000, 1.04], -libc 1.060 in [1.01, 1.10], RV8
  1.044 against [1.000, 1.04], CoreMark 1.274 against [1.000, 1.04].
  - Register-bound kernels are near 1.00, as predicted (rv8-sha512 1.0005, aha-mont64 1.008).
  - 12 programs exceed the 1.25 falsifier: sglib-listinsertsort 1.648, miniz 1.460, crc 1.439, fir 1.426,
    janne_complex 1.387, bs 1.387, sglib-listsort 1.386, fibcall 1.370, qrduino 1.319, dijkstra 1.314,
    sglib-dllist 1.285, coremark 1.274.
  - Two programs read below 0.99: nettle-arcfour 0.979 (CI 0.978-0.981) and duff 0.984 (CI 0.963-1.005).
- P4 ObjLife-v/ObjBounds below bitstream 3's ObjLife/MapLife: MISS. They are equal: -libc 1.0272 against 1.0273, and
  sglib-*-libc are not 0.5-3 points lower. The hardware check costs ObjLife's images what it costs MapLife's.
- P5 ObjLife/ObjBounds >= ObjLife/MapLife: holds except bubblesort, duff, nettle-arcfour and rv8-dhrystone. The
  allocation-heavy geomean is 1.097 (> 1.044).

Controls for the cross-bitstream reading:
- MapLife against MapLife across days on bitstream 3 reads 1.000 by suite: the Caps kits' and the Plain kits' MapLife
  (both board 1) against the records (boards 1 and 2) read 0.9998-1.0006 and 1.0000-1.0004.
- On board 2 alone (the records' board-2 boots t3a2, t3a4, t3k2 against ObjBounds), BEEBS upstream reads 1.098 over
  41 programs, against 1.099 for all 80. The other suites have 3-4 programs on board 2 and are not compared.
- A first version of this control was labelled "board 2 alone" while it compared the Caps kits' MapLife with
  ObjBounds. The Caps kits ran on board 1 (`perf/T0a/board/caps-b1.list:1`; every row has board=1), so that version
  was a cross-board comparison. The audit caught the mislabel before anything was published.

Caps: no program above 1.02. Eleven point estimates fall below 0.999 (crc32 0.991, lcdnum 0.990, ...), and every one
of their intervals contains 1. Order Caps <= MapLife <= ObjLife within 0.5 % holds except crc32 and lcdnum, both with
wide intervals.

Plain, ObjLife/Plain corrected:
- BEEBS-93: [1.40, 1.90] predicted, 1.746 read.
- BEEBS upstream: [1.30, 1.75] predicted, 1.554 read.
- RV8: [1.10, 1.45] predicted, 1.296 read.
- CoreMark: [1.10, 1.45] predicted, 2.061 read, a MISS.
- BEEBS -libc: [2.0, 3.3] predicted, 3.333 read, a MISS by 0.03.
- miniz-libc: [0.6, 1.3] predicted, 0.859 read.

## Mechanism

An adversarial audit (claim-auditor, 2026-10-31) checked the magnitude against the mechanism. Its arithmetic below
was re-checked here from the rows.
- **The per-op cost is as pre-registered.** The bare kit, board 2 against board 2 (`objb-bare-grade.py --compare`,
  9 of 9 hold), reads s6 (1,024 LDC) 9,232 -> 5,131 cycles, which is 4.005 per op, and s7 (1,024 STC) 11,269 -> 6,153,
  which is 4.996 per op. Plain loads (s1) are unchanged.
- **The count is what the model missed.** Compiled code loads and stores capabilities far more often than the
  pre-registration's class-based bands assumed: frame spills and reloads, and store-then-reload pairs through one stack
  slot.
  - fibcall: `benchmark()` is constant-folded, so an iteration is the harness frame: 2 STC + 2 LDC, 18 extra cycles
    predicted. Measured (32,096,450 - 23,422,304) / 479,488 = 18.09.
  - crc: also folded after the first call, 4 STC + 4 LDC per iteration (instret 37.09 per iteration). 36 predicted,
    36.16 measured.
  - sglib-listinsertsort: the list walk is 6 instructions with 1 STC and 2 LDC, so 13 extra cycles per step. The
    estimate of about n^2/4 steps for 100 nodes gives about 36k per call; measured 33.2k.
  So fibcall's 1.37 and crc's 1.44 measure call frames, not the named algorithms. These ratios measure the check
  together with this compiler's capability-copy pattern.
- **The cycle counter did not change rate.** The ns ratio tracks the cycle ratio (fibcall 1.370 cycles, 1.372 ns), and
  instret is equal.
- **A residual the mechanism does not explain.** nettle-arcfour 0.979 [0.978, 0.981] and rv8-dhrystone 0.994
  [0.990, 0.998] are slower without the check, with intervals excluding 1. Removing a check cannot slow code down by
  itself, so the variant changes some other timing as well. One unverified candidate is load/store ordering once the
  LSU verdict is gone; arcfour is load-load-store-store swaps. The effect is too small to move any geomean, but the
  ObjBounds rung is not purely "minus the check".
