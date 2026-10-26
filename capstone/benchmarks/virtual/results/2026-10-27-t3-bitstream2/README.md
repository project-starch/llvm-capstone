# T3 (lifetimes on, scrub off) against T1 and T2 on silicon, bitstream 2

Bitstream caplifive_vp2c_dcc6daf6e, three boots, 2026-10-27.

| kit | board | processes | window |
|---|---|---|---|
| t3k1 | 1 | 81 | 02:15–02:48 |
| t3k2 | 2 | 60 | 02:15–02:52 |
| t3k3 | 1 | 60 | 02:48–03:26 |

Every one of the 201 processes passed: rc 0, every BENCH line present, no VERIFY-FAILED.

**Pre-registration.** The design and predictions are in PREREG-t1-t2-sampling.md, section "T3: lifetimes without
scrub". It was pushed at edea56432aaa before launch. Two amendments followed, both before any transcript was read:
- 49675ed8e504 replaced the replicate criterion with interval overlap;
- e15fe66ae894 added the instret tolerance.

**Arms.** All three arms run in the same boot, in ABCCBA order.
- T1: CAPSTONE_HEAP_LIFETIMES=0, CAPSTONE_HEAP_SCRUB_FREE=0.
- T3: lifetimes on, scrub off.
- T2: both on.

The dimensions are the main run's: P = 4 × (W = 1, I = 5) per arm. huffbench-libc, which is not re-entrant, runs at
P = 11 × (0, 1).

**Files.**
- `runs-<kit>.tsv.gz`: parse-board.py's per-iteration rows, warm-ups included. `steps-<kit>.tsv` gives each kit's
  process order.
- `T1-T3/`, `T3-T2/`, `T1-T2/`: perf-analyze.py `report --arms` for each pair.
  - T3/T1 is the per-object lifetimes' cost and T2/T3 is the scrub's.
  - The suite geomeans cover only the 15 allocating programs. They are not the main run's suite figures.
- `t3-checks.txt`: the pre-registered checks (t3-checks.py).

**Regeneration.** Run from this directory. The output reproduces every file byte for byte; this was checked by
rerunning in a fresh copy.

    for k in t3k1 t3k2 t3k3; do zcat runs-$k.tsv.gz | tail -n +$([ $k = t3k1 ] && echo 1 || echo 2); done > all.tsv
    awk -F'\t' 'BEGIN{OFS="\t"} {if (NR>1 && $3=="beebs" && $4 ~ /-libc$/) $3="beebs-libc"; print}' all.tsv > main.tsv
    for pr in T1,T3 T3,T2 T1,T2; do python3 ../../perf-analyze.py report main.tsv --out ${pr/,/-} --arms $pr --boot 10000 --seed 20261026; done
    python3 ../../t3-checks.py all.tsv ../2026-10-26-bitstream2/per-program.tsv T1-T3 T3-T2 T1-T2 > t3-checks.txt
