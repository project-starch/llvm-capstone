#define CAPSTONE_DELIN(rd) \
  __asm__ volatile (".insn r 0x5b, 0x1, 0x3, %0, x0, x0" : "+r"(rd))

static __attribute__((noinline)) short int *beebs_fdct_block_ptr(void) {
  short int *p = block;
  CAPSTONE_DELIN(p);
  return p;
}

static __attribute__((noinline)) const short int *beebs_fdct_block_ref_ptr(void) {
  const short int *p = block_ref;
  CAPSTONE_DELIN(p);
  return p;
}

static __attribute__((noinline)) const short int *beebs_fdct_exp_res_ptr(void) {
  const short int *p = exp_res;
  CAPSTONE_DELIN(p);
  return p;
}

void initialise_benchmark(void) {
}

int benchmark(void) {
  short int *dst = beebs_fdct_block_ptr();
  const short int *src = beebs_fdct_block_ref_ptr();

  for (long i = 0; i < 64; ++i)
    dst[i] = src[i];

  fdct(dst, 8);
  return 0;
}

int verify_benchmark(int unused) {
  (void)unused;
  short int *actual = beebs_fdct_block_ptr();
  const short int *expected = beebs_fdct_exp_res_ptr();

  for (long i = 0; i < 64; ++i)
    if (actual[i] != expected[i])
      return 0;
  return 1;
}
