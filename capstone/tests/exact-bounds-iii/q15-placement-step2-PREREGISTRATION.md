# Q-15 placement repair (memcached 0006, postgres 0003): step 1 read, step 2 pre-registered (2026-10-19)

Run roots are on the build host under `/mnt/hdd/<user>/capstone/`. Comparator `q15-cmp.py`: per cell, verdict, cause,
symbol, controls, notes and infra.

## Step 1, exact bounds (qkit7-eb), ran 2026-10-19 00:32, judged 2026-10-19

The repaired images (t3-new8, this branch at 8f66b5fba146) against their exact-bounds records:

| arm | reference | repaired run | cells identical | image hashes differ |
|---|---|---|---|---|
| memcached allocator-repros, virtual-nested-pools | t3-eb-pb13 | t3-q15-eb-pb13 | 9 of 9 | 9 of 9 |
| postgres mmgr-repros, virtual-pg-pools | t3-pgm1 | t3-q15-eb-pgm1 | 5 of 5 | 5 of 5 |
| postgres sql-repros, virtual-pg-pools | t3-sql-eb-virtual-pg-pools | t3-q15-eb-sql | 9 of 9 | 9 of 9 |

- **Comparator control:** pb13's reference against pgm1's run reads 0 identical of 14.
- **Pad-masking:** no cell flips CAUGHT to MISSED, so no pad-masked candidate.

## Step 2, compact (qkit7), pre-registered before it runs

The same images and invocations on qkit7 (exact bounds stripped).

Prediction:
- every cell is identical to its step-1 reading;
- no console carries a `cause=29` attributed to `capstone_cap_derive`.

A cell that differs is reported with its console line. A 29 at `capstone_cap_derive` marks an unrepaired derive site.
