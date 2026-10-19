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

## Addendum, 2026-10-19, before any reading: what PAD-MASKED means (the semantics owner's ruling)

A case CAUGHT unrepaired and MISSED repaired because its overflow lands in the tail pad is a true MISSED of the
repaired configuration. On silicon the pad is accessible slack. It is recorded in verdicts.tsv as MISSED, with reason
`pad-masked` and the pad in bytes in its evidence. It is never excluded, never NO-READING, and never folded under the
unrepaired verdict. If the count of such cells is nonzero, it goes in the result line and in the commit subject, since
it is a cost of the Q-15 repair that the paper's tables must carry.

## Step 2 READ (2026-10-20): the prediction FAILS for memcached; the cache-path repair and its re-run, pre-registered

**Readings** on qkit7 (compact), against step 1:
- pgm1 5/5 and sql 9/9 are identical.
- pb13 is 7/9: cases 00 and 08 read NO-READING:control-failed. Both refuse 29, in their buggy AND fixed runs, at
  `capstone_cap_derive+0x9c` (word 0xa2e395db, CDERIVE).
- This pre-registration's check "no console carries a 29" was VOID as run: it grepped `raw/*`, which are directories,
  so it read no console. There are four such lines.

**Cause:** a caller census of `capstone_cap_derive` in case 00's image names `do_cache_alloc` (2 calls) and
`sublet_derive`. cache.c derives the object's lifetime and the caller's child over `bufsize` = object + 16 (the
lifetime slot). For the read buffer that is 16384 + 16 = 16400, which is not a multiple of its granule, 32.

**Repair**, in 0006 (regenerated from the patched tree; it applies with fuzz 0 and reproduces it):
- cache_create rounds bufsize to `capstone_repr_round` (a tail pad only; malloc's exact object puts the base on the
  granule) and keeps the pad in cache_t.q15_pad;
- do_cache_alloc counts it through slabs_q15_note(), a weak reference, so builds without slabs.c still link;
- the cache pad therefore appears in the Q15_PAD line.
- Images: t3-new8b/prebuilt/b13 (defect-00 6ab1d1d0148d, defect-08 47c236c38400). All differ from t3-new8's and carry
  slabs_q15_note.

**Re-run predictions** (pb13 only):
- step 1 (qkit7-eb): every cell identical to t3-eb-pb13 (9/9). A CAUGHT to MISSED flip is a pad-masked cell, counted.
- step 2 (qkit7): every cell identical to that step 1, and 0 `cause=29` lines, counted in the FILE
  `raw/guest/serial.log` with its line count printed beside the hit count.
