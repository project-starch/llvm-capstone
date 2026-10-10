#ifndef WIRESHARK_WMEM_PORT_H
#define WIRESHARK_WMEM_PORT_H
#include <stddef.h>
#include <stdint.h>
#define WM_MAGIC UINT64_C(0x31304d454d575357)
/* The largest object a trace may request, and the replay's capacities. */
#define WM_OBJECT_BYTES (384UL << 20)
#define WM_TRACE_BYTES (16UL << 20)
#define WM_ALLOCATORS 64
#define WM_OBJECTS 8192
enum { WM_NEW = 1, WM_ALLOC, WM_FREE, WM_REALLOC, WM_FREE_ALL, WM_GC, WM_DESTROY, WM_END };
struct wm_header {
  uint64_t magic, count, mode, status, completed, news, allocs, frees;
  uint64_t reallocs, free_alls, gcs, destroys, checksum, live_allocators,
      system_allocs, system_peak;
};
struct wm_event {
  uint64_t op, allocator, object, size, type, arg;
};
_Noreturn void wm_fail(unsigned code);
/* What wmem asks of the system allocator -- whole blocks, jumbo objects and
 * its own descriptors -- is the process's malloc, counted. */
void *wm_sys_alloc(size_t n);
void wm_sys_free(void *p);
void *wm_sys_realloc(void *p, size_t n);
void wm_system_stats(struct wm_header *out);
char *wm_getenv(const char *name);
void wm_replay(const struct wm_header *in, struct wm_header *out);
#endif
