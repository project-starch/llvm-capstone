#ifndef BEEBS_INSERTSORT_KERNEL_H
#define BEEBS_INSERTSORT_KERNEL_H
/* Silicon-ladder rung 2: BEEBS insertsort (a *found* benchmark, not hand-written).
 *
 * Source: Bristol/Embecosm BEEBS `insertsort` (SNU-RT suite). Kept faithful --
 * `is_a[11]` is the benchmark's global integer array, `is_sort` is the original
 * `benchmark()` insertion sort verbatim, `is_init`/`is_verify` are the original
 * initialise/verify. This rung is single-TU (no RISK-A per-module index issue)
 * and exercises: a global array reached via `ldc gp[i]`, in-loop array stores
 * (shrink-off config), and an init->sort->verify call graph. The array is filled
 * in code (.bss), so like rung 1 it needs only zero-carved storage from the gp
 * cap-table builder. A native host and the domain fold the same FNV-1a checksum
 * over the sorted array + the verify result, giving one deterministic oracle. */

static unsigned int is_a[11]; /* assume all data is positive */

/* verbatim BEEBS insertsort benchmark() */
static int is_sort(void) {
  int i, j;
  unsigned int temp;
  i = 2;
  while (i <= 10) {
    j = i;
    while (is_a[j] < is_a[j - 1]) {
      temp = is_a[j];
      is_a[j] = is_a[j - 1];
      is_a[j - 1] = temp;
      j--;
    }
    i++;
  }
  return 0;
}

/* verbatim BEEBS insertsort initialise_benchmark() */
static void is_init(void) {
  is_a[0] = 0;  is_a[1] = 11; is_a[2] = 10; is_a[3] = 9;
  is_a[4] = 8;  is_a[5] = 7;  is_a[6] = 6;  is_a[7] = 5;
  is_a[8] = 4;  is_a[9] = 3;  is_a[10] = 2;
}

/* verbatim BEEBS insertsort verify_benchmark() (expected[] is a stack local) */
static int is_verify(void) {
  int i;
  int expected[] = {0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  for (i = 0; i < 11; i++)
    if (is_a[i] != (unsigned int)expected[i])
      return 0;
  return 1;
}

static unsigned is_compute(void) {
  is_init();
  is_sort();
  int ok = is_verify();
  unsigned h = 2166136261u; /* FNV-1a over the sorted array, LE bytes */
  for (int i = 0; i < 11; i++) {
    unsigned v = is_a[i];
    for (int b = 0; b < 4; b++) { h ^= (v >> (8 * b)) & 0xffu; h *= 16777619u; }
  }
  h ^= (unsigned)ok; h *= 16777619u; /* fold correctness into the oracle */
  return h;
}
#endif
