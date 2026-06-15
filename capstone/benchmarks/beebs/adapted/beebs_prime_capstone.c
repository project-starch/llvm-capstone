typedef unsigned char bool;
typedef unsigned long ulong;

volatile int result = 0;
ulong x;
ulong y;

#define CAPSTONE_DELIN(rd) \
  __asm__ volatile (".insn r 0x5b, 0x1, 0x3, %0, x0, x0" : "+r"(rd))

static __attribute__((noinline)) volatile int *beebs_prime_result_ptr(void) {
  volatile int *p = &result;
  CAPSTONE_DELIN(p);
  return p;
}

static __attribute__((noinline)) ulong *beebs_prime_x_ptr(void) {
  ulong *p = &x;
  CAPSTONE_DELIN(p);
  return p;
}

static __attribute__((noinline)) ulong *beebs_prime_y_ptr(void) {
  ulong *p = &y;
  CAPSTONE_DELIN(p);
  return p;
}

static __attribute__((noinline)) int beebs_prime_result_get(void) {
  volatile int *p = beebs_prime_result_ptr();
  return *p;
}

static __attribute__((noinline)) void beebs_prime_result_set(int value) {
  volatile int *p = beebs_prime_result_ptr();
  *p = value;
}

static __attribute__((noinline)) ulong beebs_prime_x_get(void) {
  ulong *p = beebs_prime_x_ptr();
  return *p;
}

static __attribute__((noinline)) void beebs_prime_x_set(ulong value) {
  ulong *p = beebs_prime_x_ptr();
  *p = value;
}

static __attribute__((noinline)) ulong beebs_prime_y_get(void) {
  ulong *p = beebs_prime_y_ptr();
  return *p;
}

static __attribute__((noinline)) void beebs_prime_y_set(ulong value) {
  ulong *p = beebs_prime_y_ptr();
  *p = value;
}

bool divides(ulong n, ulong m) {
  return (m % n == 0);
}

bool even(ulong n) {
  return divides(2, n);
}

bool prime(ulong n) {
  ulong i;
  if (even(n))
    return (n == 2);
  for (i = 3; i * i <= n; i += 2) {
    if (divides(i, n))
      return 0;
  }
  return (n > 1);
}

static void beebs_prime_swap_globals(void) {
  ulong tmp = beebs_prime_x_get();
  beebs_prime_x_set(beebs_prime_y_get());
  beebs_prime_y_set(tmp);
}

void swap(ulong *a, ulong *b) {
  ulong tmp = *a;
  *a = *b;
  *b = tmp;
}

int benchmark(void) {
  beebs_prime_swap_globals();
  beebs_prime_result_set(!(prime(beebs_prime_x_get()) &&
                           prime(beebs_prime_y_get())));
  return 0;
}

void initialise_benchmark(void) {
  beebs_prime_x_set(21649L);
  beebs_prime_y_set(513239L);
}

int verify_benchmark(int unused) {
  (void)unused;
  return beebs_prime_result_get() == 0;
}
