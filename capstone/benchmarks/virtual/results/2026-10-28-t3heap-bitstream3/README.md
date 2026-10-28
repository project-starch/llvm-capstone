# The 15 allocating programs on bitstream 3: every program now has T1, T3 and T2 on one bitstream

Bitstream caplifive_vp2c_e7af48457, three boots, 2026-10-28. The 15 programs that call the runtime's malloc/free
inside the timed region had their three arms only on bitstream 2 (`../2026-10-27-t3-bitstream2/`). With this run
and `../2026-10-27-t3all-bitstream3/` (the other 88), all 103 programs have T1, T3 and T2 on bitstream 3, and suite
geomeans no longer mix bitstreams.

| kit | board | processes | window |
|---|---|---|---|
| t3k1 | 1 | 81 | 04:42–05:15 |
| t3k3 | 1 | 60 | 05:15–05:53 |
| t3k2 | 2 | 60 | 06:26–07:03 |

Every one of the 201 processes passed: rc 0, every BENCH line present, no VERIFY-FAILED, and no mid-line heartbeat.

**Pre-registration.** PREREG-t1-t2-sampling.md, section "The 15 allocating programs on bitstream 3", pushed at
811dc7a43eb3 before any boot. The checks script, t3heap-checks.py, was written after the boots and before any
transcript was read. Each check was shown to fire on tampered inputs: a changed reference check=, a 1 % instret
shift, a disjoint reference interval and a shifted suite geomean each produced a miss.

**Kits.** `perf/main3b/t3k{1,2,3}` unchanged from the bitstream-2 T3 run: the same images, launcher 978c09d61f774315
and run scripts (0ecbbc7c4c9f5ec8, fc24bc09d77351fc, 37861b84fca1ac04). ABCCBA, P = 4 × (W = 1, I = 5) per arm,
huffbench-libc at P = 11 × (0, 1).

**Files.**
- `runs-<kit>.tsv.gz`: parse-board.py's per-iteration rows, warm-ups included. `steps-<kit>.tsv` gives each kit's
  process order.
- `T1-T3/`, `T3-T2/`, `T1-T2/`: perf-analyze.py `report --arms` on this run's 15 programs.
- `all103/`: the same three reports over this run's rows plus the T3-all run's: every program, bitstream 3, the
  usual suites (beebs, beebs-libc, rv8, coremark, ctl).
- `groups/`: the same with beebs-libc folded into beebs, so BEEBS is one group of 93 programs (the 80 upstream
  programs and the 13 that allocate through libc malloc), as the lead chose for the summary figure on 2026-10-28.
  The 13 are variants of upstream programs, so 13 programs enter the BEEBS group twice, once per allocator.
- `t3heap-checks.txt`: the pre-registered checks.
- Per-program figures cite `T1-T3/` etc. here and in the T3-all run, not `all103/`. In `all103/` and `groups/` the
  ratios and Welch intervals are identical to the per-run reports, but the bootstrap bounds (blo, bhi) differ by up
  to 0.0008, because the bootstrap draws from one stream across all programs of a report.

**Result. Every pre-registered prediction holds.**
- Correctness: check= is one value across the three arms in every program and equals the bitstream-2 T3 run's.
- instret equals the bitstream-2 T3 run's within 0.01 % for every program and arm (largest: miniz-libc, 0.009 %).
- All 45 program-pair intervals overlap their bitstream-2 intervals. Point estimates move by at most 0.008
  (huffbench-libc T3/T1 1.0049 against 0.9971). huffbench-libc's T2/T1 now reads 1.0069 [1.0017, 1.0121], an
  interval above 1, against 0.9999 [0.9887, 1.0081] on bitstream 2. The two intervals overlap: the readings agree,
  and this one is tighter.
- miniz-libc again reads T2/T1 2.1820 with T3/T1 1.0009: the scrub is almost all of it.

Suite geomeans over all 103 programs on bitstream 3 (`groups/` for BEEBS, `all103/` for the rest), 95 % CI:

| group | programs | T3/T1 | T2/T3 | T2/T1 |
|---|---|---|---|---|
| BEEBS (upstream + libc malloc) | 93 | 1.0038 [1.0034, 1.0042] | 1.0104 [1.0099, 1.0108] | 1.0142 [1.0137, 1.0147] |
| — of which upstream | 80 | 1.0001 [0.9997, 1.0004] | 0.9999 [0.9994, 1.0003] | 1.0000 [0.9995, 1.0004] |
| — of which libc malloc | 13 | 1.0273 [1.0257, 1.0288] | 1.0772 [1.0755, 1.0788] | 1.1065 [1.1050, 1.1081] |
| RV8 | 7 | 0.9999 [0.9984, 1.0014] | 1.0248 [1.0234, 1.0263] | 1.0247 [1.0231, 1.0263] |
| CoreMark | 1 | 1.0004 [0.9992, 1.0021] | 0.9998 [0.9980, 1.0019] | 1.0003 [0.9990, 1.0022] |
| controls (ctl) | 2 | 0.9980 [0.9946, 1.0018] | 1.0156 [0.9969, 1.0482] | 1.0136 [0.9951, 1.0464] |

CoreMark's three arms come from the T3-all run alone; it never calls malloc.

**Regeneration.** Run from this directory. The output reproduces every file byte for byte; this was checked by
rerunning in a fresh copy (55 files).

    R=..; V=../..
    lab='BEGIN{OFS="\t"} {if (NR>1 && $3=="beebs" && $4 ~ /-libc$/) $3="beebs-libc"; print}'
    for k in 1 2 3; do zcat runs-t3k$k.tsv.gz | tail -n +$([ $k = 1 ] && echo 1 || echo 2); done > run.tsv
    awk -F'\t' "$lab" run.tsv > main.tsv
    ( cat run.tsv; for k in 1 2 3 4; do zcat $R/2026-10-27-t3all-bitstream3/runs-t3a$k.tsv.gz | tail -n +2; done ) > all.tsv
    awk -F'\t' "$lab" all.tsv > all103.tsv
    awk -F'\t' 'BEGIN{OFS="\t"} {if (NR>1 && $3=="beebs-libc") $3="beebs"; print}' all103.tsv > groups.tsv
    for k in 1 2 3; do zcat $R/2026-10-27-t3-bitstream2/runs-t3k$k.tsv.gz | tail -n +$([ $k = 1 ] && echo 1 || echo 2); done > ref.tsv
    for pr in T1,T3 T3,T2 T1,T2; do d=${pr/,/-}
      python3 $V/perf-analyze.py report main.tsv --out $d --arms $pr --boot 10000 --seed 20261026
      python3 $V/perf-analyze.py report all103.tsv --out all103/$d --arms $pr --boot 10000 --seed 20261026
      python3 $V/perf-analyze.py report groups.tsv --out groups/$d --arms $pr --boot 10000 --seed 20261026
    done
    python3 $V/t3heap-checks.py run.tsv ref.tsv . $R/2026-10-27-t3-bitstream2 all103 > t3heap-checks.txt
