# Boundary annotation — cffi-lua #57

### The object that crosses the boundary

A raw pointer to a heap `closure_data` C++ struct (holding a libffi
`ffi_closure` + a registry ref), stored inside the Lua `callback` cdata returned
by `cffi.cast(...)`. The cdata is the Lua-visible handle; the `closure_data`
pointer is what crosses.

### Owner vs. borrower

- **The C side (cffi-lua/libffi) owns the memory.** `make_cdata_func`
  (`ffi.cc:271`) `new[]`s the `closure_data`; `destroy_closure` (`ffi.cc:127`)
  `delete[]`s it.
- **Lua (managed) owns the handle.** The cdata's lifetime is the GC's; its
  stored `fd.cd` pointer is the coupling.
- The bug: `:free()` frees the `closure_data` but leaves `fd.cd` dangling, so a
  later `:set()` (or a second `:free()`) operates on freed memory.

### Free site

`callback:free()` → `cdata_meta::cb_free` (`src/ffilib.cc:268`) →
`ffi::destroy_closure` (`src/ffi.cc:127`) → `delete[]`.

### Stale-use site (one crossing later)

`callback:set(fn)` → `cdata_meta::cb_set` (`src/ffilib.cc:281`) reads
`fd.cd->fref` on the freed block → ASan heap-use-after-free (READ of size 4).
A second `callback:free()` re-`delete[]`s the same pointer (double free).

### The lifetime rule that is violated

A handle that owns a native resource must invalidate its stored pointer the
moment that resource is freed, so later methods fault cleanly instead of
touching freed memory. The fix (`ced2cba79`) nulls `fd.cd` and guards
`cb_set`/`cb_free` with `if (!fd.cd) luaL_error(...)`.

### Capability note (revoke-on-free)

On a revoke-on-free allocator the `:free()` at `ffi.cc:127` **revokes** the
capability to the `closure_data`. `:set()` then holds a revoked capability, so
the `READ` at `ffilib.cc:281` faults at the contract point — exactly the
delivered fault the capability model promises, in place of the ASan-detected UAF.
