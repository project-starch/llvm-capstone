/* Included by the patched upstream allocators under WMEM_PORT_HOOKS only. */
#ifndef WMEM_PORT_HOOKS_H
#define WMEM_PORT_HOOKS_H
#include "port.h"
/* Narrow a returned object to its request. wmem never hands out zero-byte
 * objects, so SHRINK always has a nonempty interval to encode. */
static inline void *wm_narrow(void *p, size_t n) {
#ifdef WM_DOMAIN
  uintptr_t base = (uintptr_t)p;
  return __builtin_capstone_cap_shrink(p, base, base + n);
#else
  (void)n;
  return p;
#endif
}
#endif
