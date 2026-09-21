#include "scopes.h"
static wmem_allocator_t *file_scope, *epan_scope, *packet_cache;
void wm_scopes_init(void) {
  if (file_scope || epan_scope || packet_cache)
    wm_fail(220);
  wmem_init();
  file_scope = wmem_allocator_new(WMEM_ALLOCATOR_BLOCK);
  epan_scope = wmem_allocator_new(WMEM_ALLOCATOR_BLOCK);
  /* Pools start in scope; the file scope opens with the first capture. */
  wmem_leave_scope(file_scope);
}
wmem_allocator_t *wm_file_scope(void) {
  if (!file_scope)
    wm_fail(221);
  return file_scope;
}
wmem_allocator_t *wm_epan_scope(void) {
  if (!epan_scope)
    wm_fail(221);
  return epan_scope;
}
void wm_enter_file_scope(void) {
  if (!file_scope || wmem_in_scope(file_scope))
    wm_fail(222);
  wmem_enter_scope(file_scope);
}
void wm_leave_file_scope(void) {
  if (!file_scope || !wmem_in_scope(file_scope))
    wm_fail(223);
  wmem_leave_scope(file_scope);
  /* Upstream collects here, returning wholly unused blocks to the system. */
  wmem_gc(file_scope);
}
wmem_allocator_t *wm_packet_pool_acquire(void) {
  wmem_allocator_t *pool = packet_cache;
  if (pool)
    packet_cache = NULL;
  else
    pool = wmem_allocator_new(WMEM_ALLOCATOR_BLOCK_FAST);
  return pool;
}
void wm_packet_pool_reset(wmem_allocator_t *pool) { wmem_free_all(pool); }
void wm_packet_pool_release(wmem_allocator_t *pool) {
  if (!packet_cache) {
    wmem_free_all(pool);
    packet_cache = pool;
  } else {
    wmem_destroy_allocator(pool);
  }
}
void wm_scopes_cleanup(void) {
  if (!file_scope || !epan_scope || wmem_in_scope(file_scope))
    wm_fail(224);
  wmem_destroy_allocator(file_scope);
  wmem_destroy_allocator(epan_scope);
  if (packet_cache)
    wmem_destroy_allocator(packet_cache);
  wmem_cleanup();
  file_scope = epan_scope = packet_cache = NULL;
}
