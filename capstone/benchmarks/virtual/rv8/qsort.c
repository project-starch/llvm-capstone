/* RV8 qsort: upstream main() with ARRAY_SIZE = RV8_N ints, sorted by upstream's own BSD qsort.
 * Upstream prints arr[ARRAY_SIZE-1] and never frees; here the array is checked (non-decreasing,
 * sum preserved) and freed after the bracket. */
#include RV8_UPSTREAM
#include "rv8-common.h"

static int *arr;

static void unit(void) {
  arr = malloc(RV8_N * sizeof(int));
  int val = 1;
  for (size_t i = 0; i < RV8_N; i++) {
    arr[i] = val;
    val = ((val * 8191) << 7) ^ val;
  }
  qsort(arr, RV8_N, sizeof(int), compare);
}

static int post(unsigned long long *check) {
  long long in_sum = 0, out_sum = 0;
  int val = 1, sorted = 1;
  for (size_t i = 0; i < RV8_N; i++) {
    in_sum += val;
    val = ((val * 8191) << 7) ^ val;
    out_sum += arr[i];
    if (i && arr[i - 1] > arr[i])
      sorted = 0;
  }
  *check = ((unsigned long long)(unsigned)arr[RV8_N - 1] << 32) ^ (unsigned long long)out_sum;
  free(arr);
  return sorted && in_sum == out_sum;
}

int main(int argc, char **argv) { return bench_run("rv8/qsort", argc, argv, unit, post); }
