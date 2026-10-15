/* main() and the pool for the nested-lifetime controls. One program per control, as the corpus
 * contract says: a capability fault ends the process, so the stale access of control 93 runs in
 * its own process (`stale`), separate from the permitted accesses (`run`).
 *
 *   ctl-NN run NN      the control's permitted accesses
 *   ctl-NN stale NN    control 93 only: the same sequence, then the read through A's old pointer
 */
#include "ctl.h"

AVBufferPool *g_pool;

_Noreturn void ff2_fail(unsigned code) {
  fflush(stdout);
  fprintf(stderr, "CONTROL-FAILED %u\n", code);
  exit(75); /* an infrastructure failure is never a reading */
}
/* The library's event hooks (buffer-pool patch 0001, which 0003 is written against). The
 * controls observe nothing through them. */
void ff2_sink(const struct ff2_event *event) { (void)event; }
void ff2_lock(void) {}
void ff2_unlock(int *guard) { (void)guard; }

int main(int argc, char **argv) {
  if (argc != 3 || (strcmp(argv[1], "run") && strcmp(argv[1], "stale"))) {
    fprintf(stderr, "CONTROL-FAILED usage: %s run|stale NN\n", argv[0]);
    return 75;
  }
  if (atoi(argv[2]) != ctl_case_number) {
    fprintf(stderr, "CONTROL-FAILED program is control %d, run asked for %s\n", ctl_case_number,
            argv[2]);
    return 75;
  }
  int stale = !strcmp(argv[1], "stale");
  ff2_reset();
  g_pool = av_buffer_pool_init(POOL_BYTES, NULL);
  if (!g_pool)
    ff2_fail(605);
  printf("case=%d control=%s arm=%s mode=%s\n", ctl_case_number, ctl_case_name,
         CTL_PROTECTED ? "protected" : "release", stale ? "stale" : "run");
  fflush(stdout);
  int rc = ctl_case_run(stale);
  av_buffer_pool_uninit(&g_pool);
  if (rc == 0)
    printf("CTL-RESULT completed\n");
  fflush(stdout);
  return rc;
}
