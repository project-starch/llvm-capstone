#define ARRAYSIZE 100
#define INVOCATION_COUNT 43

char source[ARRAYSIZE];
char target[ARRAYSIZE];

#define CAPSTONE_DELIN(rd) \
  __asm__ volatile (".insn r 0x5b, 0x1, 0x3, %0, x0, x0" : "+r"(rd))

static __attribute__((noinline)) char *beebs_duff_source_ptr(void) {
  char *p = source;
  CAPSTONE_DELIN(p);
  return p;
}

static __attribute__((noinline)) char *beebs_duff_target_ptr(void) {
  char *p = target;
  CAPSTONE_DELIN(p);
  return p;
}

static __attribute__((noinline)) char beebs_duff_source_get(long idx) {
  char *p = beebs_duff_source_ptr();
  return p[idx];
}

static __attribute__((noinline)) void beebs_duff_source_set(long idx,
                                                            char value) {
  char *p = beebs_duff_source_ptr();
  p[idx] = value;
}

static __attribute__((noinline)) char beebs_duff_target_get(long idx) {
  char *p = beebs_duff_target_ptr();
  return p[idx];
}

static __attribute__((noinline)) void beebs_duff_target_set(long idx,
                                                            char value) {
  char *p = beebs_duff_target_ptr();
  p[idx] = value;
}

void duffcopy(int count) {
  int n = (count + 7) / 8;
  long idx = 0;

  switch (count % 8) {
  case 0:
    do {
      beebs_duff_target_set(idx, beebs_duff_source_get(idx));
      idx++;
  case 7:
      beebs_duff_target_set(idx, beebs_duff_source_get(idx));
      idx++;
  case 6:
      beebs_duff_target_set(idx, beebs_duff_source_get(idx));
      idx++;
  case 5:
      beebs_duff_target_set(idx, beebs_duff_source_get(idx));
      idx++;
  case 4:
      beebs_duff_target_set(idx, beebs_duff_source_get(idx));
      idx++;
  case 3:
      beebs_duff_target_set(idx, beebs_duff_source_get(idx));
      idx++;
  case 2:
      beebs_duff_target_set(idx, beebs_duff_source_get(idx));
      idx++;
  case 1:
      beebs_duff_target_set(idx, beebs_duff_source_get(idx));
      idx++;
    } while (--n > 0);
  }
}

int benchmark(void) {
  duffcopy(INVOCATION_COUNT);
  return 0;
}

void initialise_benchmark(void) {
  for (long i = 0; i < ARRAYSIZE; i++) {
    beebs_duff_source_set(i, (char)(ARRAYSIZE - i));
    beebs_duff_target_set(i, 0);
  }
}

int verify_benchmark(int unused) {
  (void)unused;
  for (long i = 0; i < INVOCATION_COUNT; i++)
    if (beebs_duff_source_get(i) != beebs_duff_target_get(i))
      return 0;
  return 1;
}
