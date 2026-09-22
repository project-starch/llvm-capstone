/* Included by the patched upstream allocators under WMEM_PORT_HOOKS only. */
#ifndef WMEM_PORT_HOOKS_H
#define WMEM_PORT_HOOKS_H
#include "port.h"
#if defined(WM_POISONCAP)
void *wm_publish(void *p, size_t n);
void wm_release_chunk(void *p, size_t n);
#endif
/* Narrow a returned object to its request. wmem never hands out zero-byte
 * objects, so SHRINK always has a nonempty interval to encode. */
static inline void *wm_narrow(void *p, size_t n) {
#if defined(WM_DOMAIN)
  uintptr_t base = (uintptr_t)p;
  return __builtin_capstone_cap_shrink(p, base, base + n);
#elif defined(WM_POISONCAP)
  return wm_publish(p, n);
#else
  (void)n;
  return p;
#endif
}
/* An individual free in the recycler allocator. Sublet lends whole regions
 * and ends no epoch here; a mechanism that acts per chunk can. */
static inline void wm_release(void *p, size_t n) {
#if defined(WM_POISONCAP)
  wm_release_chunk(p, n);
#else
  (void)p;
  (void)n;
#endif
}
#endif
