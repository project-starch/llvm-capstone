#include "port.h"
#include <stddef.h>
/* The pinned wmem core parses an allocator override from the environment;
 * this port selects allocators explicitly, so no override ever exists. */
char *wm_getenv(const char *name) {
  (void)name;
  return NULL;
}
/* wmem_map.c is not part of this port; the core's one call into it is a
 * hashing seed that no extracted unit consumes. */
void wmem_init_hashing(void) {}
