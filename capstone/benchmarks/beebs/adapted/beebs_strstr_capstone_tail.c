#define CAPSTONE_DELIN(rd) \
  __asm__ volatile (".insn r 0x5b, 0x1, 0x3, %0, x0, x0" : "+r"(rd))

static char text_data[] =
    "abbaabbaababadcsdabbacasdaabbbaabbadabbacbbbaabbadabbacasdaabbbaabba";
static char substr_data[] = "abba";

static __attribute__((noinline)) char *beebs_strstr_text_ptr(void) {
  char *p = text_data;
  CAPSTONE_DELIN(p);
  return p;
}

static __attribute__((noinline)) char *beebs_strstr_substr_ptr(void) {
  char *p = substr_data;
  CAPSTONE_DELIN(p);
  return p;
}

void initialise_benchmark(void) {
}

int benchmark(void) {
  char *substr = beebs_strstr_substr_ptr();
  char *f = beebs_strstr_text_ptr();
  int n = 0;

  do {
    f = strstr(f + 1, substr);
    n++;
  } while (f);

  return n;
}

int verify_benchmark(int r) {
  return r == 8;
}
