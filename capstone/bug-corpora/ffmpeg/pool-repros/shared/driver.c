/* main(), the arenas and the pool. One program per case, as the contract says:
 * a capability fault ends a domain, so a case that provokes one cannot also
 * report results beside it. */
#include "corpus.h"

AVBufferPool *g_pool;

_Noreturn void ff2_fail(unsigned code) {
  fprintf(stderr, "CONTROL-FAILED %u\n", code);
  exit(75); /* an infrastructure failure is never a verdict */
}
void ff2_sink(const struct ff2_event *event) { (void)event; }
void ff2_lock(void) {}
void ff2_unlock(int *guard) { (void)guard; }

int main(int argc, char **argv) {
  int fixed = argc > 1 && !strcmp(argv[1], "fixed");
  if (argc > 2 && atoi(argv[2]) != ff2_case_number) {
    fprintf(stderr, "CONTROL-FAILED fixture is case %d, run asked for %s\n",
            ff2_case_number, argv[2]);
    return 75;
  }
  void *metadata = aligned_alloc(64, FF2_META_BYTES);
  void *payload = aligned_alloc(64, FF2_PAYLOAD_BYTES);
  if (!metadata || !payload)
    ff2_fail(604);
  ff2_memory_init(metadata, FF2_META_BYTES);
  ff2_payload_init(payload, FF2_PAYLOAD_BYTES);
  ff2_set_mode(0); /* the native arms carry no protection; that is the point */
  ff2_reset();
  g_pool = av_buffer_pool_init(POOL_BYTES, NULL);
  if (!g_pool)
    ff2_fail(605);
  printf("case=%d arm=%s\n", ff2_case_number, fixed ? "fixed" : "buggy");
  int rc = ff2_case_run(fixed);
  av_buffer_pool_uninit(&g_pool);
  free(metadata);
  free(payload);
  return rc;
}
