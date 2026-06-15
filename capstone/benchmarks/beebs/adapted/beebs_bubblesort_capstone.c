#define FALSE 0
#define TRUE 1
#define NUMELEMS 100
#define MAXDIM (NUMELEMS + 1)

int Array[MAXDIM], Seed;
int factor;

#define CAPSTONE_DELIN(rd) \
  __asm__ volatile (".insn r 0x5b, 0x1, 0x3, %0, x0, x0" : "+r"(rd))

static __attribute__((noinline)) int *beebs_bubblesort_array_ptr(void) {
  int *p = Array;
  CAPSTONE_DELIN(p);
  return p;
}

static __attribute__((noinline)) int *beebs_bubblesort_factor_ptr(void) {
  int *p = &factor;
  CAPSTONE_DELIN(p);
  return p;
}

static __attribute__((noinline)) int beebs_bubblesort_get(int idx) {
  int *p = beebs_bubblesort_array_ptr();
  return p[idx];
}

static __attribute__((noinline)) void beebs_bubblesort_set(int idx, int value) {
  int *p = beebs_bubblesort_array_ptr();
  p[idx] = value;
}

static __attribute__((noinline)) void beebs_bubblesort_factor_set(int value) {
  int *p = beebs_bubblesort_factor_ptr();
  *p = value;
}

static __attribute__((noinline)) int beebs_bubblesort_factor_get(void) {
  int *p = beebs_bubblesort_factor_ptr();
  return *p;
}

void BubbleSort(int unused[]) {
  (void)unused;
  int sorted = FALSE;
  int temp, index, i;

  for (i = 0; i < NUMELEMS; i++) {
    sorted = TRUE;
    for (index = 0; index < NUMELEMS; index++) {
      if (index >= NUMELEMS - i)
        break;
      if (beebs_bubblesort_get(index) > beebs_bubblesort_get(index + 1)) {
        temp = beebs_bubblesort_get(index);
        beebs_bubblesort_set(index, beebs_bubblesort_get(index + 1));
        beebs_bubblesort_set(index + 1, temp);
        sorted = FALSE;
      }
    }

    if (sorted)
      break;
  }
}

int benchmark(void) {
  BubbleSort(Array);
  return 0;
}

void initialise_benchmark(void) {
  int index;

  beebs_bubblesort_factor_set(-1);
  int fact = beebs_bubblesort_factor_get();
  for (index = 0; index < NUMELEMS; index++)
    beebs_bubblesort_set(index, index * fact);
}

int verify_benchmark(int result) {
  (void)result;
  for (int i = 0; i < NUMELEMS; i++) {
    int expected = i - (NUMELEMS - 1);
    if (beebs_bubblesort_get(i) != expected)
      return 0;
  }
  return 1;
}
