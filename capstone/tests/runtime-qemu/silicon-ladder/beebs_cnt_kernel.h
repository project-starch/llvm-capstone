#ifndef BEEBS_CNT_KERNEL_H
#define BEEBS_CNT_KERNEL_H
/* Silicon-ladder rung: BEEBS cnt -- seed a 10x10 matrix, then sum and count it.
 *
 * Source: Bristol/Embecosm BEEBS `cnt` (MAXSIZE reduced to 10 upstream already).
 * Verbatim compute, including the LCG and the four result globals.
 *
 * SHAPE PREDICTION under issue R-1 (ref/ISSUES.md): PASS -- and this rung is
 * chosen because the prediction is NOT trivial. It is the sharpest available
 * test of R-1's *same object* clause.
 *
 *   R-1 as characterised: a load through capability register X misses a store
 *   through a DIFFERENT capability register Y **into the same object**.
 *
 * cnt's seeding loop does `Array[i][j] = RandomInteger()`, and RandomInteger
 * loads and stores the global `Seed` on every call. So there are two live
 * capability registers with a store outstanding through each -- but they name
 * two DIFFERENT globals, hence two different cap-table entries and two
 * different objects. The summing loop then re-reads the whole matrix.
 *
 *   - If it PASSES, the same-object clause is real and R-1 stays narrow. That
 *     matters practically: most benchmark code stores to one object while
 *     reading another, so a narrow R-1 leaves far more of a benchmark suite
 *     measurable than a wide one.
 *   - If it FAILS, R-1 is wider than we wrote it down -- any two derived
 *     capability registers, not just two into one object -- and both the
 *     registry entry and the repro README have to be corrected before they go
 *     to the board owner.
 *
 * `matmult_int` (a known failure) is the same-object case: C[i*N+j] += ...
 * reads and writes ONE array through two derived registers. cnt is the
 * cross-object control that has been missing from the whole investigation. */

#define CNT_MAXSIZE 10

static int cnt_Seed;
static int cnt_Array[CNT_MAXSIZE][CNT_MAXSIZE];
static int cnt_Postotal, cnt_Negtotal, cnt_Poscnt, cnt_Negcnt;

static int cnt_RandomInteger(void) {
  cnt_Seed = ((cnt_Seed * 133) + 81) % 8095;
  return cnt_Seed;
}

static void cnt_InitSeed(void) { cnt_Seed = 0; }

static void cnt_Initialize(void) {
  int outer, inner;
  for (outer = 0; outer < CNT_MAXSIZE; outer++)
    for (inner = 0; inner < CNT_MAXSIZE; inner++)
      cnt_Array[outer][inner] = cnt_RandomInteger();
}

static int cnt_Sum(void) {
  int outer, inner;
  int ptotal = 0, ntotal = 0, pcnt = 0, ncnt = 0;
  for (outer = 0; outer < CNT_MAXSIZE; outer++)
    for (inner = 0; inner < CNT_MAXSIZE; inner++)
      if (cnt_Array[outer][inner] < 0) {
        ptotal += cnt_Array[outer][inner];
        pcnt++;
      } else {
        ntotal += cnt_Array[outer][inner];
        ncnt++;
      }
  cnt_Postotal = ptotal;
  cnt_Poscnt = pcnt;
  cnt_Negtotal = ntotal;
  cnt_Negcnt = ncnt;
  return cnt_Negtotal;
}

static unsigned cnt_compute(void) {
  unsigned h = 2166136261u;
  for (int rep = 0; rep < 32; rep++) {
    cnt_InitSeed();
    cnt_Initialize();
    h ^= (unsigned)cnt_Sum();
    h *= 16777619u;
    /* Fold the other three result globals in too, so a partial miscompute
       cannot hide behind the one value upstream happens to return. */
    h ^= (unsigned)cnt_Postotal; h *= 16777619u;
    h ^= (unsigned)cnt_Poscnt;   h *= 16777619u;
    h ^= (unsigned)cnt_Negcnt;   h *= 16777619u;
  }
  return h;
}
#endif
