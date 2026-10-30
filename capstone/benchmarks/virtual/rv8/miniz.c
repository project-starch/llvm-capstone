/* RV8 miniz: upstream main() (step 0: no fuzzing) with DATA_SIZE = RV8_N bytes. Upstream frees
 * pCmp and pUncomp but never pData; here pData is freed after the bracket. Deviations: the test
 * pattern comes from rv8_rand, a copy of musl's rand() (6364136223846793005*s+1, >>33), so the
 * native glibc build generates the same bytes; pData is zeroed after malloc (see aes.c). */
#include RV8_UPSTREAM
#include "rv8-common.h"

static uint64_t rv8_seed;
static void rv8_srand(unsigned s) { rv8_seed = s - 1; }
static int rv8_rand(void) {
  rv8_seed = 6364136223846793005ULL * rv8_seed + 1;
  return rv8_seed >> 33;
}

static unsigned char *pData;
static int status, ok;
static uLong cmp_len_out;

static void unit(void) {
  const size_t DATA_SIZE = RV8_N;
  int cmp_status;
  uLong src_len = DATA_SIZE;
  uLong cmp_len = compressBound(src_len);
  uLong uncomp_len = src_len;
  unsigned char *pCmp, *pUncomp;
  ok = 0;
  pData = malloc(DATA_SIZE);
  memset(pData, 0, DATA_SIZE);
  rv8_srand(0);
  for (size_t j = 0; j < DATA_SIZE; j += sizeof(int))
    pData[j] = rv8_rand();
  pCmp = (mz_uint8 *)malloc((size_t)cmp_len);
  pUncomp = (mz_uint8 *)malloc((size_t)src_len);
  if (!pCmp || !pUncomp) {
    status = -1;
    return;
  }
  cmp_status = compress(pCmp, &cmp_len, (const unsigned char *)pData, src_len);
  cmp_len_out = cmp_len;
  if (cmp_status != Z_OK) {
    status = -2;
    free(pCmp);
    free(pUncomp);
    return;
  }
  cmp_status = uncompress(pUncomp, &uncomp_len, pCmp, cmp_len);
  status = cmp_status;
  ok = cmp_status == Z_OK && uncomp_len == src_len && !memcmp(pUncomp, pData, (size_t)src_len);
  free(pCmp);
  free(pUncomp);
}

static int post(unsigned long long *check) {
  *check = ((unsigned long long)cmp_len_out << 8) ^ (unsigned long long)(unsigned)status;
  free(pData);
  return ok && cmp_len_out < RV8_N;
}

int main(int argc, char **argv) { return bench_run("rv8/miniz", argc, argv, unit, post); }
