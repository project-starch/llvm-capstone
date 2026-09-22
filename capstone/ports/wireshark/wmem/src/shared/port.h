#ifndef WIRESHARK_WMEM_PORT_H
#define WIRESHARK_WMEM_PORT_H
#include <stddef.h>
#include <stdint.h>
#define WM_MAGIC UINT64_C(0x31304d454d575357)
#define WM_PAYLOAD_BYTES (384UL << 20)
#define WM_META_BYTES (8UL << 20)
#define WM_TRACE_BYTES (16UL << 20)
#define WM_ALLOCATORS 64
#define WM_OBJECTS 8192
#define WM_REGIONS 4096
enum { WM_NEW = 1, WM_ALLOC, WM_FREE, WM_REALLOC, WM_FREE_ALL, WM_GC, WM_DESTROY, WM_END };
struct wm_header {
  uint64_t magic, count, mode, status, completed, news, allocs, frees;
  uint64_t reallocs, free_alls, gcs, destroys, checksum, live_allocators,
      regions_created, regions_peak;
};
struct wm_event {
  uint64_t op, allocator, object, size, type, arg;
};
_Noreturn void wm_fail(unsigned code);
void wm_init_backing(void *metadata, void *payload, unsigned mode);
/* What wmem asks of the system allocator: whole blocks, jumbo objects and
 * its own descriptors. Every request is one region with its own authority. */
void *wm_sys_alloc(size_t n);
void wm_sys_free(void *p);
void *wm_sys_realloc(void *p, size_t n);
/* A retained block starts a new epoch: the same storage under fresh authority. */
void *wm_epoch(void *p);
/* Recover block-wide authority from an object pointer the allocator is handed
 * back, after proving that pointer still carries authority of its own. */
void *wm_widen(void *p);
void wm_backing_stats(struct wm_header *out);
char *wm_getenv(const char *name);
void wm_replay(const struct wm_header *in, struct wm_header *out);
#endif
