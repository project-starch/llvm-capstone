#ifndef BEEBS_NS_KERNEL_H
#define BEEBS_NS_KERNEL_H
/* Silicon-ladder rung: BEEBS `ns` -- deeply nested loops over a multi-D lookup table.
 *
 * Source: BEEBS `src/ns/libns.c` (Malardalen WCET suite via BEEBS, GPLv3). `foo()`
 * is VERBATIM apart from the outer bound: the same nested loops, the same linear
 * scan, the same `return answer[i][j][k][l] + keys[i][j][k][l]` early exit, the same
 * -1 fallthrough. Table VALUES are byte-for-byte upstream, dumped from a native
 * compile of libns.c rather than transcribed by hand.
 *
 * WHY THIS ONE. It is the opposite pole of `beebs_aha_mont64` on the one axis this
 * ladder exists to measure. mont64 touches essentially no memory; `ns` does almost
 * nothing BUT memory -- 500 four-level indexed loads per call, each needing the full
 * i*125 + j*25 + k*5 + l address computation, with essentially no arithmetic on the
 * loaded value. If overhead really is a property of DATA ACCESS rather than
 * execution, these two rungs bracket the claim from both ends in one board session.
 *
 * SLICE, and it is the only structural change: upstream's tables are [5][5][5][5]
 * (2,500 B each). The gp-captable glue materialises initialized globals with an
 * unrolled li/sd sequence whose store offsets are a 12-bit signed field, so a single
 * global must stay under 2,048 B, and the generator additionally requires size%8==0
 * (2,500 % 8 == 4). Both tables therefore keep upstream OUTER SLICES 1..4 and drop
 * slice 0 -- which is entirely zeros, so it is the least destructive cut available --
 * giving [4][5][5][5] = 2,000 B, a multiple of 8 and inside the offset field.
 * This was not a guess: the full [5][5][5][5] version was built and the generator
 * rejected it verbatim with "2512 B of *initialized* data overflows the 12-bit store
 * offset and is not copy-eligible (sym='ns_keys', size%8=4)". The alternative -- the
 * large-RO COPY path -- is C-4b, which is broken.
 * What the cut preserves is everything the rung is here for: four-level nesting,
 * 500 pure read-only indexed loads per miss, and 401 (upstream's only non-trivial
 * key, at the LAST element) still present so the early-exit path is reachable.
 *
 * SHAPE SCREEN (a shape argument, not a prediction -- grep-based screening was
 * measured NON-predictive on 2026-07-28; passes and fails land in the same bucket):
 *   - R-1 (register-indexed load with an intervening store to the same object):
 *     neither table is ever written. Same pure-read indexed shape as `beebs_bs`,
 *     which passes.
 *   - C-4/C-5: 2 x 2,000 B is still by far the largest initialized data of any rung
 *     (`rv8_sha512`'s K table is 640 B) and is well past the 256 B threshold at which
 *     the glue would switch to the broken copy path, so this rung REQUIRES both
 *     opt-in knobs -- DOMAIN_WINDOW=32k LADDER_NO_RO_COPY=1 -- and is registered in
 *     ladder-rungs.spec with exactly those.
 *
 * ADAPTATION of the driver: upstream `benchmark()` calls `foo(400)` once and discards
 * the result (`verify_benchmark` returns -1 -- upstream ships this one WITHOUT
 * verification). A single call is both too short to bracket and unverifiable, so this
 * runs NS_REPS calls and folds the returned values into an FNV hash. 400 is upstream's
 * argument and is deliberately NOT in `keys`, so it costs the full scan -- the WCET
 * worst case, which is the point of the benchmark. Every eighth call passes 401
 * instead, which IS present: that exercises the early-exit path and, critically, makes
 * `answer` live. Without it `answer` is never loaded and the compiler is free to drop
 * half the data this rung exists to exercise. */

#define NS_REPS 16

/* Upstream `keys`, outer slices 1..4 -- searched linearly; 400 is absent by design,
   401 is the final element. */
const int ns_keys[4][5][5][5] = {
  {
    {
      {1,1,1,1,1},
      {1,1,1,1,1},
      {1,1,1,1,1},
      {1,1,1,1,1},
      {1,1,1,1,1},
    },
    {
      {1,1,1,1,1},
      {1,1,1,1,1},
      {1,1,1,1,1},
      {1,1,1,1,1},
      {1,1,1,1,1},
    },
    {
      {1,1,1,1,1},
      {1,1,1,1,1},
      {1,1,1,1,1},
      {1,1,1,1,1},
      {1,1,1,1,1},
    },
    {
      {1,1,1,1,1},
      {1,1,1,1,1},
      {1,1,1,1,1},
      {1,1,1,1,1},
      {1,1,1,1,1},
    },
    {
      {1,1,1,1,1},
      {1,1,1,1,1},
      {1,1,1,1,1},
      {1,1,1,1,1},
      {1,1,1,1,1},
    },
  },
  {
    {
      {2,2,2,2,2},
      {2,2,2,2,2},
      {2,2,2,2,2},
      {2,2,2,2,2},
      {2,2,2,2,2},
    },
    {
      {2,2,2,2,2},
      {2,2,2,2,2},
      {2,2,2,2,2},
      {2,2,2,2,2},
      {2,2,2,2,2},
    },
    {
      {2,2,2,2,2},
      {2,2,2,2,2},
      {2,2,2,2,2},
      {2,2,2,2,2},
      {2,2,2,2,2},
    },
    {
      {2,2,2,2,2},
      {2,2,2,2,2},
      {2,2,2,2,2},
      {2,2,2,2,2},
      {2,2,2,2,2},
    },
    {
      {2,2,2,2,2},
      {2,2,2,2,2},
      {2,2,2,2,2},
      {2,2,2,2,2},
      {2,2,2,2,2},
    },
  },
  {
    {
      {3,3,3,3,3},
      {3,3,3,3,3},
      {3,3,3,3,3},
      {3,3,3,3,3},
      {3,3,3,3,3},
    },
    {
      {3,3,3,3,3},
      {3,3,3,3,3},
      {3,3,3,3,3},
      {3,3,3,3,3},
      {3,3,3,3,3},
    },
    {
      {3,3,3,3,3},
      {3,3,3,3,3},
      {3,3,3,3,3},
      {3,3,3,3,3},
      {3,3,3,3,3},
    },
    {
      {3,3,3,3,3},
      {3,3,3,3,3},
      {3,3,3,3,3},
      {3,3,3,3,3},
      {3,3,3,3,3},
    },
    {
      {3,3,3,3,3},
      {3,3,3,3,3},
      {3,3,3,3,3},
      {3,3,3,3,3},
      {3,3,3,3,3},
    },
  },
  {
    {
      {4,4,4,4,4},
      {4,4,4,4,4},
      {4,4,4,4,4},
      {4,4,4,4,4},
      {4,4,4,4,4},
    },
    {
      {4,4,4,4,4},
      {4,4,4,4,4},
      {4,4,4,4,4},
      {4,4,4,4,4},
      {4,4,4,4,4},
    },
    {
      {4,4,4,4,4},
      {4,4,4,4,4},
      {4,4,4,4,4},
      {4,4,4,4,4},
      {4,4,4,4,4},
    },
    {
      {4,4,4,4,4},
      {4,4,4,4,4},
      {4,4,4,4,4},
      {4,4,4,4,4},
      {4,4,4,4,4},
    },
    {
      {4,4,4,4,4},
      {4,4,4,4,4},
      {4,4,4,4,4},
      {4,4,4,4,4},
      {4,4,4,4,401},
    },
  },
};

/* Upstream `answer`, outer slices 1..4 -- read only on a hit. NOT const upstream. */
int ns_answer[4][5][5][5] = {
  {
    {
      {234,234,234,234,234},
      {234,234,234,234,234},
      {234,234,234,234,234},
      {234,234,234,234,234},
      {234,234,234,234,234},
    },
    {
      {234,234,234,234,234},
      {234,234,234,234,234},
      {234,234,234,234,234},
      {234,234,234,234,234},
      {234,234,234,234,234},
    },
    {
      {234,234,234,234,234},
      {234,234,234,234,234},
      {234,234,234,234,234},
      {234,234,234,234,234},
      {234,234,234,234,234},
    },
    {
      {234,234,234,234,234},
      {234,234,234,234,234},
      {234,234,234,234,234},
      {234,234,234,234,234},
      {234,234,234,234,234},
    },
    {
      {234,234,234,234,234},
      {234,234,234,234,234},
      {234,234,234,234,234},
      {234,234,234,234,234},
      {234,234,234,234,234},
    },
  },
  {
    {
      {345,345,345,345,0},
      {345,345,345,345,0},
      {345,345,345,345,0},
      {345,345,345,345,0},
      {345,345,345,345,0},
    },
    {
      {345,345,345,345,0},
      {345,345,345,345,0},
      {345,345,345,345,0},
      {345,345,345,345,0},
      {345,345,345,345,0},
    },
    {
      {345,345,345,345,0},
      {345,345,345,345,0},
      {345,345,345,345,0},
      {345,345,345,345,0},
      {345,345,345,345,0},
    },
    {
      {345,345,345,345,0},
      {345,345,345,345,0},
      {345,345,345,345,0},
      {345,345,345,345,0},
      {345,345,345,345,0},
    },
    {
      {345,345,345,345,0},
      {345,345,345,345,0},
      {345,345,345,345,0},
      {345,345,345,345,0},
      {345,345,345,345,0},
    },
  },
  {
    {
      {456,456,456,456,456},
      {456,456,456,456,456},
      {456,456,456,456,456},
      {456,456,456,456,456},
      {456,456,456,456,456},
    },
    {
      {456,456,456,456,456},
      {456,456,456,456,456},
      {456,456,456,456,456},
      {456,456,456,456,456},
      {456,456,456,456,456},
    },
    {
      {456,456,456,456,456},
      {456,456,456,456,456},
      {456,456,456,456,456},
      {456,456,456,456,456},
      {456,456,456,456,456},
    },
    {
      {456,456,456,456,456},
      {456,456,456,456,456},
      {456,456,456,456,456},
      {456,456,456,456,456},
      {456,456,456,456,456},
    },
    {
      {456,456,456,456,456},
      {456,456,456,456,456},
      {456,456,456,456,456},
      {456,456,456,456,456},
      {456,456,456,456,456},
    },
  },
  {
    {
      {567,567,567,567,567},
      {567,567,567,567,567},
      {567,567,567,567,567},
      {567,567,567,567,567},
      {567,567,567,567,567},
    },
    {
      {567,567,567,567,567},
      {567,567,567,567,567},
      {567,567,567,567,567},
      {567,567,567,567,567},
      {567,567,567,567,567},
    },
    {
      {567,567,567,567,567},
      {567,567,567,567,567},
      {567,567,567,567,567},
      {567,567,567,567,567},
      {567,567,567,567,567},
    },
    {
      {567,567,567,567,567},
      {567,567,567,567,567},
      {567,567,567,567,567},
      {567,567,567,567,567},
      {567,567,567,567,567},
    },
    {
      {567,567,567,567,567},
      {567,567,567,567,567},
      {567,567,567,567,567},
      {567,567,567,567,567},
      {567,567,567,567,1111},
    },
  },
};

/* Upstream foo(), outer bound 4 instead of 5 (see SLICE above); otherwise verbatim. */
static int ns_foo(int x) {
  int i, j, k, l;

  for (i = 0; i < 4; i++)
    for (j = 0; j < 5; j++)
      for (k = 0; k < 5; k++)
        for (l = 0; l < 5; l++) {
          if (ns_keys[i][j][k][l] == x) {
            return ns_answer[i][j][k][l] + ns_keys[i][j][k][l];
          }
        }
  return -1;
}

static unsigned ns_compute(void) {
  unsigned h = 2166136261u;
  for (int rep = 0; rep < NS_REPS; rep++) {
    /* 400 = upstream's argument, absent from keys -> full 500-element scan.
       401 IS present (last element) -> exercises the early exit and makes
       `answer` live. */
    int x = ((rep & 7) == 7) ? 401 : 400;
    h ^= (unsigned)ns_foo(x);
    h *= 16777619u;
  }
  return h;
}
#endif
