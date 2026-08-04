# Boundary annotation — Lua-cURLv3 easy ⟷ multi dangling pointer

### The object that crosses the boundary

A raw C pointer `lcurl_multi_t *` (the field `lcurl_easy_t::multi`), set inside
the easy's C struct by `multi:add_handle(easy)`. It points at the **multi**
userdata — a *separate* Lua-GC object whose `lcurl_multi_t` struct wraps a
native `CURLM*`. The pointer crosses from the easy's C domain into the multi's.

### Owner vs. borrower

- **Lua (the GC) owns both userdata lifetimes.** The multi userdata is freed
  when it becomes unreachable and its finalizer has run.
- **The easy borrows** a raw C back-pointer to the multi (`e->multi`) with **no
  Lua-visible edge** keeping the multi reachable from the easy. `add_handle`
  installs the reverse edge only (multi → easy, in the multi's weak handle
  table), so the multi can be collected first while the easy still lives.
- The bug: on the vulnerable tree the multi's finalizer does not invalidate the
  easies' `->multi` back-pointers, so a live easy is left pointing at freed
  memory.

### Free site

`m = nil; collectgarbage()` makes the multi userdata unreachable. Its finalizer
`lcurl_multi_cleanup` (`src/lcmulti.c`) runs `curl_multi_cleanup(p->curl)` and
unrefs its handle table, then Lua frees the 96-byte `lcurl_multi_t` userdata via
`l_alloc`. On the vulnerable tree it does **not** null `e->multi` for the still
attached easy.

### Stale-use site (one crossing later)

`easy:close()` → `lcurl_easy_cleanup` (`src/lceasy.c:87`) executes
`if(p->multi){ p->multi->L = L; }`, a **write through the dangling `e->multi`**
into the freed multi userdata → ASan `heap-use-after-free` (WRITE of size 8).

### The lifetime rule that is violated

If object A stores a raw C back-pointer to a separately-GC'd object B, then B's
finalizer must invalidate every such back-pointer before B's memory is freed
(or A must hold a Lua reference that keeps B alive). Here neither held: the easy
kept no Lua reference to the multi, and (pre-fix) the multi's finalizer left the
back-pointers dangling. The fix (`56b4d05`) makes `lcurl_multi_cleanup` walk the
handle table and set every `e->multi = NULL` before the userdata is freed.

### Capability note (revoke-on-free)

On a revoke-on-free allocator, collecting the multi userdata **revokes** the
capability to that `lcurl_multi_t` block. The easy's `e->multi` is a derived
capability to the same block, so it is revoked too: the `easy:close()` write
`p->multi->L = L` faults at the store through a revoked capability — at the
exact C↔Lua handoff — instead of silently corrupting a reused userdata. This is
a boundary-only case (the dangling deref is one crossing after the free).
