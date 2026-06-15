static int a, b;

#define CAPSTONE_DELIN(rd) \
  __asm__ volatile (".insn r 0x5b, 0x1, 0x3, %0, x0, x0" : "+r"(rd))

static __attribute__((noinline)) int *beebs_janne_complex_a_ptr(void) {
  int *p = &a;
  CAPSTONE_DELIN(p);
  return p;
}

static __attribute__((noinline)) int *beebs_janne_complex_b_ptr(void) {
  int *p = &b;
  CAPSTONE_DELIN(p);
  return p;
}

static __attribute__((noinline)) int beebs_janne_complex_a_get(void) {
  int *p = beebs_janne_complex_a_ptr();
  return *p;
}

static __attribute__((noinline)) void beebs_janne_complex_a_set(int value) {
  int *p = beebs_janne_complex_a_ptr();
  *p = value;
}

static __attribute__((noinline)) int beebs_janne_complex_b_get(void) {
  int *p = beebs_janne_complex_b_ptr();
  return *p;
}

static __attribute__((noinline)) void beebs_janne_complex_b_set(int value) {
  int *p = beebs_janne_complex_b_ptr();
  *p = value;
}

int complex(int a_arg, int b_arg) {
  while (a_arg < 30) {
    while (b_arg < a_arg) {
      if (b_arg > 5)
        b_arg = b_arg * 3;
      else
        b_arg = b_arg + 2;

      if (b_arg >= 10 && b_arg <= 12)
        a_arg = a_arg + 10;
      else
        a_arg = a_arg + 1;
    }

    a_arg = a_arg + 2;
    b_arg = b_arg - 10;
  }

  return 1;
}

int benchmark(void) {
  return complex(beebs_janne_complex_a_get(), beebs_janne_complex_b_get());
}

void initialise_benchmark(void) {
  beebs_janne_complex_a_set(1);
  beebs_janne_complex_b_set(1);
}

int verify_benchmark(int r) {
  return r == 1;
}
