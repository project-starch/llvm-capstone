#define NUM_STRINGS 5
#define MAX_STRING_LEN 10

static const char string0[] = "srrjngre";
static const char string1[] = "asfcjnsdkj";
static const char string2[] = "string";
static const char string3[] = "msd";
static const char string4[] = "strings";

#define CAPSTONE_DELIN(rd) \
  __asm__ volatile (".insn r 0x5b, 0x1, 0x3, %0, x0, x0" : "+r"(rd))

static int min(int x, int y) {
  return x < y ? x : y;
}

static __attribute__((noinline)) const char *beebs_levenshtein_string_ptr(
    long idx) {
  const char *p = string4;
  switch (idx) {
  case 0:
    p = string0;
    break;
  case 1:
    p = string1;
    break;
  case 2:
    p = string2;
    break;
  case 3:
    p = string3;
    break;
  default:
    p = string4;
    break;
  }
  CAPSTONE_DELIN(p);
  return p;
}

static __attribute__((noinline)) long beebs_levenshtein_strlen(
    const char *s) {
  long len = 0;
  while (s[len] != 0)
    len++;
  return len;
}

int levenshtein_distance(const char *s, const char *t) {
  long sl = beebs_levenshtein_strlen(s);
  long tl = beebs_levenshtein_strlen(t);
  int d[(MAX_STRING_LEN + 1) * (MAX_STRING_LEN + 1)];

  for (long i = 0; i <= sl; i++)
    d[i * (MAX_STRING_LEN + 1)] = (int)i;

  for (long j = 0; j <= tl; j++)
    d[j] = (int)j;

  for (long j = 1; j <= tl; j++) {
    for (long i = 1; i <= sl; i++) {
      long cur = i * (MAX_STRING_LEN + 1) + j;
      long prev_row = (i - 1) * (MAX_STRING_LEN + 1);
      long row = i * (MAX_STRING_LEN + 1);
      if (s[i - 1] == t[j - 1]) {
        d[cur] = d[prev_row + j - 1];
      } else {
        d[cur] = min(d[prev_row + j] + 1,
                     min(d[row + j - 1] + 1, d[prev_row + j - 1] + 1));
      }
    }
  }

  return d[sl * (MAX_STRING_LEN + 1) + tl];
}

void initialise_benchmark(void) {}

int benchmark(void) {
  volatile unsigned sum = 0;

  for (long i = 0; i < NUM_STRINGS; i++) {
    for (long j = 0; j < NUM_STRINGS; j++) {
      const char *s = beebs_levenshtein_string_ptr(i);
      const char *t = beebs_levenshtein_string_ptr(j);
      sum += levenshtein_distance(s, t);
    }
  }

  return sum;
}

int verify_benchmark(int r) {
  return r == 122;
}
