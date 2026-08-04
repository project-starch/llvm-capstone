# Boundary annotation — tarantool #7657

### The object that crosses the boundary

A `struct merge_source *` (a refcounted C merge source, with a `vtab` of
`next`/`destroy` function pointers), stored inside a LuaJIT cdata of ctype
`CTID_STRUCT_MERGE_SOURCE_REF`. The cdata is the Lua-visible handle; the
`struct merge_source *` is what crosses.

### Owner vs. borrower

- **The C side (merger module) owns the memory.** The source constructor
  allocates the `struct merge_source` (refcount 1); `merge_source_unref` runs the
  vtab `destroy` and frees it when the refcount reaches 0.
- **LuaJIT (managed) owns each handle.** Every cdata over the source carries a
  `__gc` = `lbox_merge_source_gc`, which calls `merge_source_unref`.
- The bug: `source:pairs()` returns the luafun triple
  `(gen=lbox_merge_source_gen, param=nil, state=<source cdata>)`, and each
  `lbox_merge_source_gen` step pushes back a **fresh** cdata wrapping the SAME
  `merge_source *` as the next `state` — **without** a matching
  `merge_source_ref`, yet with a finalizer that will `unref`. So a GC that fires
  mid-iteration collects an intermediate cdata, drives the refcount to 0, and
  frees the still-in-use `struct merge_source`.

### Free site

GC of an intermediate iterator cdata → `lbox_merge_source_gc`
(`src/box/lua/merger.c`) → `merge_source_unref` → vtab `destroy` → the
`struct merge_source` is freed (refcount hit 0 one step too early).

### Stale-use site (one crossing later)

The next `lbox_merge_source_gen` (`src/box/lua/merger.c`) does
`luaT_check_merge_source(L, 2)` on the freed struct and calls
`source->vtab->next(...)`. On the release build LuaJIT's sweep has zeroed the
freed slab, so `vtab`/`vtab->next` reads as NULL and the indirect call jumps to
address 0 → `SEGV_MAPERR` at `addr 0` (`rip=0x0`, `cr2=0x0`).

### The lifetime rule that is violated

If two handles may independently free a refcounted resource, every handle that
is created must take its own reference before it can be collected. Minting a new
cdata over the same `merge_source *` without `merge_source_ref`, while its `__gc`
will `merge_source_unref`, makes unrefs outnumber refs and frees the resource
while an active iterator still points at it. The fix (`e52fabf9`) makes
`lbox_merge_source_gen` return the **same** cdata it was passed instead of a new
one, so ref/unref stay balanced across the whole iteration.

### Capability note (revoke-on-free)

On a revoke-on-free allocator, `merge_source_unref` freeing the struct **revokes**
the capability to the `struct merge_source`. The next `lbox_merge_source_gen`
then holds a revoked capability, so the deref of `source->vtab` faults at the
contract point — the delivered fault the capability model promises, in place of
the accidental null-vtable SIGSEGV that a stock allocator produces only because
the slab happened to be zeroed.
