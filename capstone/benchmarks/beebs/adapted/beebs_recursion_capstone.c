volatile int In;
static int n;

#define CAPSTONE_DELIN(rd) \
  __asm__ volatile (".insn r 0x5b, 0x1, 0x3, %0, x0, x0" : "+r"(rd))

static __attribute__((noinline)) volatile int *beebs_recursion_in_ptr(void) {
  volatile int *p = &In;
  CAPSTONE_DELIN(p);
  return p;
}

static __attribute__((noinline)) int *beebs_recursion_n_ptr(void) {
  int *p = &n;
  CAPSTONE_DELIN(p);
  return p;
}

static __attribute__((noinline)) int beebs_recursion_in_get(void) {
  volatile int *p = beebs_recursion_in_ptr();
  return *p;
}

static __attribute__((noinline)) void beebs_recursion_in_set(int value) {
  volatile int *p = beebs_recursion_in_ptr();
  *p = value;
}

static __attribute__((noinline)) int beebs_recursion_n_get(void) {
  int *p = beebs_recursion_n_ptr();
  return *p;
}

static __attribute__((noinline)) void beebs_recursion_n_set(int value) {
  int *p = beebs_recursion_n_ptr();
  *p = value;
}

int fib(int i) {
  if (i == 0)
    return 1;
  if (i == 1)
    return 1;
  return fib(i - 1) + fib(i - 2);
}

int anka(int i);

int kalle(int i) {
  if (i <= 0)
    return 0;
  return anka(i - 1);
}

int anka(int i) {
  if (i <= 0)
    return 1;
  return kalle(i - 1);
}

int benchmark(void) {
  beebs_recursion_in_set(fib(beebs_recursion_n_get()));
  return beebs_recursion_in_get();
}

void initialise_benchmark(void) {
  beebs_recursion_n_set(10);
}

int verify_benchmark(int r) {
  return r == 89;
}
