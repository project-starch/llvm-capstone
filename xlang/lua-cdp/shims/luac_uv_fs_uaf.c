/* luv #696 — Lua userdata ⟷ libuv uv_fs_t cross-thread use-after-free.
 * Source: ../../luv-696/boundary.md. ASan: SEGV on a freed uv_fs_t (the free
 * side is luv_fs_gc, fs.c:40); no labelled region size in the trace.
 *
 * Two allocations: the userdata wrapping the scandir request and the libuv
 * uv_fs_t request the worker thread reads via uv_fs_scandir_next.
 *   Free-site (fs.c:40): main thread: req = nil; collectgarbage() -> luv_fs_gc
 *     -> uv_fs_req_cleanup frees/cleans the request.
 *   Stale-use: worker thread: uv_fs_scandir_next derefs the freed uv_fs_t -> SEGV.
 * The cross-thread race is collapsed to sequential free-then-read here — the
 * allocator-visible memory event (alloc -> free -> deref) is identical, and the
 * domain is single-threaded. READ at OFFSET 0 -> plain load through the revoked
 * capability (clean cause-25 route). Control: the read returns; row reports.
 */
#include "luac_shim.h"
#include <stdint.h>

#define UV_FS_BYTES 64 /* size not named in the SEGV trace; a 16-multiple */

static volatile uint64_t sink;

int main(void) {
  void *req = malloc(UV_FS_BYTES); /* uv_fs_scandir request */
  if (!req)
    abort();
  memset(req, 0, UV_FS_BYTES);

  void *worker_req = req; /* the pointer the threadpool worker holds */

  free(req); /* luv_fs_gc -> uv_fs_req_cleanup -> REVOKE */

  /* uv_fs_scandir_next derefs the freed request. */
  sink = *(volatile uint64_t *)worker_req; /* fs.c worker-side read */

  mock_report("luac_uv_fs_uaf", "use-after-free-survived");
  return 0;
}
