/* Reports whether CHERI heap temporal safety (quarantine+revoke) is actually
 * active for THIS process — so the baseline records config reality, not just the
 * sysctl we asked for. Prints one line the host driver greps. */
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <malloc_np.h>

int main(void) {
  bool en = malloc_revoke_enabled();
  printf("REVOKE_ENABLED=%d\n", en ? 1 : 0);
  return 0;
}
