/* Control: whether the C library revokes for this process. Prints libc_revoke_enabled, which is
 * malloc_revoke_enabled() (malloc_np.h): 1 when the process was exec'd while
 * security.cheri.runtime_revocation_default was 1. run-cases.py gates on it: 1 with
 * --libc-revocation run, 0 otherwise. Runs through the same command as the cases, so the cases
 * were exec'd with the same setting. */
#include "repro322_common.h"
#include <malloc_np.h>

static int run_case(void) {
  int rc = repro_init();
  if (rc) return rc;
  out_text("repro libc_revoke_enabled="); out_uint(malloc_revoke_enabled() ? 1 : 0); out_text("\n");
  return 0;
}
REPRO322_MAIN("ctllibc")
