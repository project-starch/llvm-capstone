#define CAPSTONE_DELIN(rd) \
  __asm__ volatile (".insn r 0x5b, 0x1, 0x3, %0, x0, x0" : "+r"(rd))

static __attribute__((noinline)) unsigned int *beebs_insertsort_a_ptr(void) {
  unsigned int *p = a;
  CAPSTONE_DELIN(p);
  return p;
}

static __attribute__((noinline)) int beebs_insertsort_get(int idx) {
  unsigned int *p = beebs_insertsort_a_ptr();
  return (int)p[idx];
}

static __attribute__((noinline)) void beebs_insertsort_set(int idx, int value) {
  unsigned int *p = beebs_insertsort_a_ptr();
  p[idx] = (unsigned int)value;
}

int benchmark(void) {
  int i = 2;
  while (i <= 10) {
    int j = i;
    while (beebs_insertsort_get(j) < beebs_insertsort_get(j - 1)) {
      int temp = beebs_insertsort_get(j);
      beebs_insertsort_set(j, beebs_insertsort_get(j - 1));
      beebs_insertsort_set(j - 1, temp);
      j--;
    }
    i++;
  }
  return 0;
}

void initialise_benchmark(void) {
  beebs_insertsort_set(0, 0);
  beebs_insertsort_set(1, 11);
  beebs_insertsort_set(2, 10);
  beebs_insertsort_set(3, 9);
  beebs_insertsort_set(4, 8);
  beebs_insertsort_set(5, 7);
  beebs_insertsort_set(6, 6);
  beebs_insertsort_set(7, 5);
  beebs_insertsort_set(8, 4);
  beebs_insertsort_set(9, 3);
  beebs_insertsort_set(10, 2);
}

int verify_benchmark(int unused) {
  (void)unused;
  if (beebs_insertsort_get(0) != 0) return 0;
  if (beebs_insertsort_get(1) != 2) return 0;
  if (beebs_insertsort_get(2) != 3) return 0;
  if (beebs_insertsort_get(3) != 4) return 0;
  if (beebs_insertsort_get(4) != 5) return 0;
  if (beebs_insertsort_get(5) != 6) return 0;
  if (beebs_insertsort_get(6) != 7) return 0;
  if (beebs_insertsort_get(7) != 8) return 0;
  if (beebs_insertsort_get(8) != 9) return 0;
  if (beebs_insertsort_get(9) != 10) return 0;
  if (beebs_insertsort_get(10) != 11) return 0;
  return 1;
}
