#include "port.h"
#include <stdlib.h>
/* g_malloc, g_free and g_realloc are the process's malloc, free and realloc,
 * as a stock wmem build gets them from GLib: native, AddressSanitizer,
 * CheriBSD's libc with its revocation, or virtual mallocng on Capstone. The
 * counts are the replay's report of what wmem asked the system for. */
static uint64_t allocs, live, peak;
void *wm_sys_alloc(size_t n) {
  void *p = malloc(n);
  if (!p)
    wm_fail(205);
  ++allocs;
  if (++live > peak)
    peak = live;
  return p;
}
void wm_sys_free(void *p) {
  if (!p)
    return;
  --live;
  free(p);
}
void *wm_sys_realloc(void *p, size_t n) {
  if (!p)
    return wm_sys_alloc(n);
  if (!n) {
    wm_sys_free(p);
    return NULL;
  }
  void *q = realloc(p, n);
  if (!q)
    wm_fail(208);
  return q;
}
void wm_system_stats(struct wm_header *out) {
  out->system_allocs = allocs;
  out->system_peak = peak;
}
