# T3 (lifetimes on, scrub off) on the other 88 programs, on silicon, bitstream 3

Bitstream caplifive_vp2c_e7af48457, four boots, 2026-10-27. With `../2026-10-27-t3-bitstream2/` (the 15 programs that
allocate inside the timed region, bitstream 2) every one of the 103 programs now has all three arms. The two sets
are on different bitstreams and are never pooled into one geomean.

| kit | board | programs | processes | window |
|---|---|---|---|---|
| t3a1 | 1 | 22 | 285 | 19:08–20:36 |
| t3a2 | 2 | 22 | 285 | 19:08–20:43 |
| t3a3 | 1 | 22 | 285 | 20:36–22:07 |
| t3a4 | 2 | 22 | 264 | 20:43–22:11 |

Every one of the 1119 processes passed: rc 0, every BENCH line present, no VERIFY-FAILED.

**Pre-registration.** PREREG-t1-t2-sampling.md, section "T3 on the other 88 programs, bitstream 3", pushed at
92e59181f84d before any boot. The checks script, t3all-checks.py, was committed at a15f53b84ab9 while the kits ran,
before any transcript was read.

**Arms.** All three arms of a program run in one boot, ABC..CBA, at the main run's dimensions: P = 4 × (W = 1, I = 5)
per arm, and P = 11 × (0, 1) for crc32, huffbench and wikisort, which are not re-entrant.
- T1 and T2 are the main run's images (`perf/build2`).
- T3 images are `perf/build4`: the same tree, compiler and heap layout as the first T3 images.
- The layout gate passed 88 of 88 pairs.

**Two instrument fixes, made after t3a1 was read and before the analysis** (b729deaf33d2):
1. **parse-board.py rejoins guest lines split by the driver's heartbeat.** An `@@HB` line can land in the middle of
   a guest line.
   - It happened four times in this run. t3a1 lcdnum T1 p0 w0 was cut in `ns`. t3a2 rv8-dhrystone it=1 was cut in
     `check=`, where the old parser would have kept the row with the hash cut to `0x1`. t3a4 cut the tail of
     stringsearch1's `@@PERF-END`, which carries no data. t3a3 had none.
   - The parser now removes every heartbeat with its newline and counts the mid-line ones on stderr.
   - On bitstream 2 the only such cut was k3 frac w0, a warm-up row that the analysis drops. No committed ratio
     changes.
2. **t3all-checks.py keys the QEMU twin's table by (suite, bench name).** beebs/qsort and rv8/qsort share a bench
   name, and so do the two miniz. Keyed by name alone, beebs qsort was compared with rv8's hash.

**Files.**
- `runs-<kit>.tsv.gz`: parse-board.py's per-iteration rows, warm-ups included. `steps-<kit>.tsv` gives each kit's
  process order.
- `T1-T3/`, `T3-T2/`, `T1-T2/`: perf-analyze.py `report --arms` for each pair. T3/T1 is the per-object lifetimes'
  cost and T2/T3 is the scrub's.
- `pooled/`: the same three reports with every program in one suite, `all`. This gives one geomean over the 88.
- `t3all-checks.txt`: the pre-registered checks.

**Result.**

| pair | geomean over the 88 | 95 % CI |
|---|---|---|
| T3/T1, lifetimes | 1.0000 | 0.9997–1.0004 |
| T2/T3, scrub | 1.0002 | 0.9996–1.0010 |
| T2/T1, both | 1.0003 | 0.9996–1.0011 |

- Every T3/T1 and T2/T3 point lies within 1 %, except two noise cases, which are the misses below.
  - Beyond 0.5 % lie only crc32's T3/T1 (0.9926), duff's (1.0065 and 1.0054), and the two misses.
  - All of their intervals contain 1.
- Five programs have an interval that excludes 1 in some pair, none by more than 0.34 %:
  - sglib-rbtree and trio-snprintf, discussed next;
  - wikisort, with T3/T1 0.9998;
  - rv8-sha512, with T2/T3 1.0001 (T2/T1 1.0001);
  - stringsearch1, with T2/T1 1.0007.
- sglib-rbtree and trio-snprintf read T3/T1 1.0034 and 1.0017 with intervals above 1. Their T2/T3, 0.9967 and
  0.9981, cancels that, and their T2/T1 holds at 1.000.
  - So the 0.2–0.3 % belongs to the T3 images, not to a cost. instret is equal in all three arms.

**Pre-registered checks.**
1. **Correctness: holds.** check= is one value across the three arms for all 88 programs. It equals the QEMU twin's
   hash for all 50 programs that have one; the other 38 carry none.
2. **instret within 0.1 %: holds.** The largest spread is crc32 at 0.025 %.
3. **Cycles within 1 %: two misses.** Both geomeans fall in [0.995, 1.005].
   - **lcdnum, T2/T3 0.9846**, interval [0.964, 1.006].
     - lcdnum is the noisiest program in the set. In each of the three arms, single iterations read 7–17 % below
       their process's median.
     - Its main-run interval is [0.973, 1.028].
   - **ctl-empty, T2/T3 1.0315**, interval [0.935, 1.138].
     - One T2 process, the third, read 326,500–355,800 cycles in all five iterations. The other 11 processes in all
       three arms read 298,100–307,400.
     - Its instret and entries (22) equal the others'.
     - The timed region is 0.3 M cycles, so one disturbed process moves the mean by 3 %.
4. **Replicate: holds.** This run's T2/T1 interval overlaps the main run's for all 88 programs. For ludcmp, minver
   and st the comparison is with their bitstream-3 rows (`../2026-10-27-bitstream3-fpu/`).

**Regeneration.** Run from this directory. The output reproduces every file byte for byte; this was checked by
rerunning in a fresh copy.

    for k in t3a1 t3a2 t3a3 t3a4; do zcat runs-$k.tsv.gz | tail -n +$([ $k = t3a1 ] && echo 1 || echo 2); done > all.tsv
    awk -F'\t' 'BEGIN{OFS="\t"} {if (NR>1 && $3=="beebs" && $4 ~ /-libc$/) $3="beebs-libc"; print}' all.tsv > main.tsv
    awk -F'\t' 'BEGIN{OFS="\t"} {if (NR>1) $3="all"; print}' all.tsv > pooled.tsv
    for pr in T1,T3 T3,T2 T1,T2; do
      python3 ../../perf-analyze.py report main.tsv --out ${pr/,/-} --arms $pr --boot 10000 --seed 20261026
      python3 ../../perf-analyze.py report pooled.tsv --out pooled/${pr/,/-} --arms $pr --boot 10000 --seed 20261026
    done
    python3 ../../t3all-checks.py main.tsv ../2026-10-26-bitstream2/twin-checks.tsv T1-T3 T3-T2 T1-T2 \
      ../2026-10-26-bitstream2/per-program.tsv ../2026-10-27-bitstream3-fpu/per-program.tsv > t3all-checks.txt
