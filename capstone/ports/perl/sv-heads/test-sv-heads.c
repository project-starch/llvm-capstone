/* Native unit test of the SV-head adapter core through its native backend.
 * Checks upstream Perl's allocation policy (ascending issue from a new arena,
 * LIFO reuse, a new arena only when the free list is empty), S_visit's order,
 * the released-head predicate, the delayed-publication mode and the observer.
 *
 *   cc -fsanitize=address,undefined -I. test-sv-heads.c -o test && \
 *     PERL_SVH_NATIVE_MODE=0 ./test && PERL_SVH_NATIVE_MODE=1 ./test */
#include "native.c"

#include <assert.h>

enum { BYTES = 48, PER_PAGE = 3 };

static void check(int ok, const char *what) {
  if (!ok) {
    fprintf(stderr, "FAIL %s\n", what);
    exit(1);
  }
}

int main(void) {
  check(!perl_svh_released(&svh), "foreign pointer before init");
  void *h[8];
  for (int i = 0; i < 3; ++i)
    h[i] = perl_svh_new(BYTES, PER_PAGE);
  check((char *)h[1] - (char *)h[0] == BYTES && (char *)h[2] - (char *)h[1] == BYTES,
        "a new arena issues ascending");
  check(perl_svh_extent() == 1, "one arena");
  h[3] = perl_svh_new(BYTES, PER_PAGE);
  check(perl_svh_extent() == 2 && (char *)h[3] - (char *)h[2] == BYTES,
        "the next arena only when the list is empty");
  check(!perl_svh_released(h[1]), "a live head is not released");
  perl_svh_del(h[1], 1);
  check(perl_svh_released(h[1]), "a released head is released");
  check(!perl_svh_released((char *)h[1] + 1), "a misaligned pointer is not a head");
  perl_svh_del(h[2], 1);
  if (!svh.mode) {
    check(perl_svh_new(BYTES, PER_PAGE) == h[2], "LIFO: the last released first");
    check(perl_svh_new(BYTES, PER_PAGE) == h[1], "LIFO: then the one before");
  } else {
    /* Both are delayed, so the rest of arena 2 is issued first. */
    void *next = perl_svh_new(BYTES, PER_PAGE);
    check((char *)next - (char *)h[3] == BYTES, "delayed heads are not reissued");
    check(perl_svh_released(h[1]) && perl_svh_released(h[2]), "delayed heads stay released");
    perl_svh_del(next, 1);
  }
  /* S_visit's order: newest arena first, ascending within an arena, live only. */
  size_t cursor = 0, limit = perl_svh_extent();
  void *seen[8];
  size_t n = 0;
  for (void *p; (p = perl_svh_next(&cursor, limit)) && n < 8;)
    seen[n++] = p;
  if (!svh.mode) {
    check(n == 4 && seen[0] == h[3] && seen[1] == h[0] && seen[2] == h[1] && seen[3] == h[2],
          "visit order");
  } else {
    check(n == 2 && seen[0] == h[3] && seen[1] == h[0], "visit skips released heads");
  }
  /* An SVf_BREAK head is released for good. */
  perl_svh_del(h[0], 0);
  check(perl_svh_released(h[0]) && svh.retired == 1, "retired head");
  for (int i = 0; i < 6; ++i)
    check(perl_svh_new(BYTES, PER_PAGE) != h[0], "a retired head is never reissued");
  check(svh_reuse.error == 0 && svh_reuse.issues == svh.issues &&
        svh_reuse.releases == svh.releases, "observer agrees with the core");
  uint64_t binned = 0;
  for (int i = 0; i < 32; ++i)
    binned += svh_reuse.bins[i];
  check(binned == svh_reuse.reuses, "observer histogram reconciles");
  check(svh_reuse.reuses == (svh.mode ? 0 : 2), "reuse count");
  printf("PASS mode=%u issues=%llu releases=%llu reuses=%llu\n", svh.mode,
         (unsigned long long)svh.issues, (unsigned long long)svh.releases,
         (unsigned long long)svh_reuse.reuses);
  return 0;
}
