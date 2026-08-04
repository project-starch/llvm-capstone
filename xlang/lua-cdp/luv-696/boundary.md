# Boundary annotation — luv #696

### The object that crosses the boundary
A libuv `uv_fs_t` scandir request, wrapped by a Lua userdata, handed to a work
request and read on a libuv threadpool worker thread via `uv_fs_scandir_next`.

### Owner vs. borrower
- **Lua (managed) owns the request's lifetime** through the wrapping userdata; its
  `__gc` (`luv_fs_gc`) calls `uv_fs_req_cleanup`.
- **The worker thread borrows** the `uv_fs_t` (passed via `work:queue(req)`) with
  no synchronisation against the main thread's GC.

### Free site
Main thread: `req = nil; collectgarbage()` → `luv_fs_gc` (`src/fs.c:40`) →
`uv_fs_req_cleanup` frees/cleans the request.

### Stale-use site (concurrent, one crossing away)
Worker thread: `uv_fs_scandir_next(_entries)` derefs the cleaned-up `uv_fs_t` →
SEGV (`uv_fs_req_cleanup` internals on freed state).

### The lifetime rule that is violated
A native handle handed to another thread must be pinned until that thread is
done. `0e4a895` made the req GC-able (fixing a leak) without pinning it for the
worker; `3e39f98` restores correct GC management of scandir reqs.

### Capability note (revoke-on-free)
Revoke-on-free revokes the `uv_fs_t` capability at `luv_fs_gc`; the worker
thread's copy is revoked too, so `uv_fs_scandir_next` faults at the boundary
rather than corrupting threadpool state.
