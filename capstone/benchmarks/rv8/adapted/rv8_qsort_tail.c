/*
 * Capstone adapted oracle for RV8 `qsort`.
 *
 * rv8-bench's qsort ships its own in-place BSD qsort; main() mallocs a 50M-int
 * (200 MB) array, fills it with a deterministic sequence, sorts ascending, and
 * prints the max. That array is far larger than the domain's memory, so this
 * tail uses a small static array (RV8_QSORT_N ints in .bss -- no malloc) filled
 * with the same recurrence, calls the upstream qsort/compare, and validates with
 * a self-contained oracle:
 *   - the result is sorted non-decreasing (signed, matching `compare`), and
 *   - the element sum is preserved (a sort is a permutation, so the post-sort
 *     sum must equal the pre-sort sum).
 * Together these catch both ordering and content (lost/duplicated element) bugs
 * without needing a host reference.
 */
#include "rv8_capstone_preamble.h"

#ifndef RV8_QSORT_N
#define RV8_QSORT_N 8192
#endif

typedef int cmp_t(const void *, const void *);
extern void qsort(void *a, size_t n, size_t es, cmp_t *cmp);
extern int compare(const void *a, const void *b);
extern void rv8_arena_init(void);

static int arr[RV8_QSORT_N];

void initialise_benchmark(void) { rv8_arena_init(); }

int benchmark(void) {
  int val = 1;
  long in_sum = 0;
  for (int i = 0; i < RV8_QSORT_N; i++) {
    arr[i] = val;
    in_sum += (long)val;
    val = ((val * 8191) << 7) ^ val; /* same recurrence as upstream main() */
  }

  qsort(arr, (size_t)RV8_QSORT_N, sizeof(int), compare);

  long out_sum = 0;
  int sorted = 1;
  for (int i = 0; i < RV8_QSORT_N; i++) {
    out_sum += (long)arr[i];
    if (i > 0 && arr[i - 1] > arr[i])
      sorted = 0;
  }
  return (sorted && out_sum == in_sum) ? 1 : 0;
}

int verify_benchmark(int result) { return result == 1; }
