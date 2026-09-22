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
#ifdef WM_DOMAIN
/* The freestanding string library the domain links has no strncmp. */
int strncmp(const char *a, const char *b, size_t n) {
  for (; n; --n, ++a, ++b) {
    if (*a != *b || !*a)
      return (unsigned char)*a - (unsigned char)*b;
  }
  return 0;
}
#endif
