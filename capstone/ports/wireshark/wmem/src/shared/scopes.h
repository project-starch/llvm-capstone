/* The pool lifecycle Wireshark's dissection loop drives: the per-dissection
 * packet pool (block_fast, recycled through a one-entry cache), the file
 * scope (block, garbage-collected when a capture closes) and the epan scope.
 * Modelled on epan/wmem_scopes.c and the epan_dissect_t pool handling in
 * epan/epan.c at the pinned release; not extracted from them. */
#ifndef WM_SCOPES_H
#define WM_SCOPES_H
#include "wmem_core.h"
void wm_scopes_init(void);
void wm_scopes_cleanup(void);
wmem_allocator_t *wm_file_scope(void);
wmem_allocator_t *wm_epan_scope(void);
void wm_enter_file_scope(void);
void wm_leave_file_scope(void);
wmem_allocator_t *wm_packet_pool_acquire(void);
void wm_packet_pool_reset(wmem_allocator_t *pool);
void wm_packet_pool_release(wmem_allocator_t *pool);
#endif
