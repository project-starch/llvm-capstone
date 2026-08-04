# Boundary annotation — tarantool #1955

### The object that crosses the boundary

A `struct error *` (a refcounted C error object — here a `FiberIsCancelled`
exception), stored inside a LuaJIT cdata produced by `luaL_pusherror`. The cdata
is the Lua-visible handle; the `struct error *` (and the `error_ref` count read
at `diag.c:100`) is what crosses.

### Owner vs. borrower

- **The C side (diag/exception) owns the memory.** `BuildFiberIsCancelled`
  (`src/exception.cc:264`) allocates the `struct error`; `exception_destroy`
  (`src/exception.cc:45`), reached via `error_unref`, frees it when the refcount
  reaches 0. The fiber's diagnostics area (`struct diag`) holds the owning
  reference.
- **LuaJIT (managed) owns the handle.** `lbox_error` / `luaL_pusherror`
  (`src/lua/utils.c:913,924`) takes an `error_ref` and wraps the pointer in a
  cdata.
- The bug: the diagnostics area is mutated out from under the Lua side. A GC pass
  runs the `lbox_fiber_cancel` (`src/lua/fiber.c:459`) finalizer, which clears the
  fiber diag; `diag_clear` `error_unref`s the current error to 0 and
  `exception_destroy` frees it — while `luaL_pusherror` is mid-flight taking its
  own `error_ref` on the very same, now-freed, `struct error`.

### Free site

GC → `lbox_fiber_cancel` (`src/lua/fiber.c:459`) → clears fiber diagnostics →
`diag_clear` → `error_unref` → `exception_destroy` (`src/exception.cc:45`) →
the `struct error` is freed.

### Stale-use site (one crossing later)

`lbox_error` (`src/lua/utils.c:924`) → `luaL_pusherror` (`src/lua/utils.c:913`) →
`error_ref` (`src/diag.c:100`) reads `error->refs` on the freed block → ASan
**heap-use-after-free, READ of size 4**.

### The lifetime rule that is violated

A handle that borrows a refcounted native object must hold its own reference for
the whole window it touches that object — including across any point where a GC
finalizer may run. Here the read of the diag's "last error" and the `error_ref`
that pins it are not atomic with respect to the cancel finalizer, so a
`diag_clear` between them frees the error before it is ref'd. The 1.10.3-era fix
tightens this ordering (ref before any collection point / re-check the diag) so
the error cannot be freed between select and ref.

### Capability note (revoke-on-free)

On a revoke-on-free allocator, `exception_destroy` freeing the `struct error`
**revokes** the capability to it. The subsequent `error_ref` in `luaL_pusherror`
then holds a revoked capability, so the READ at `diag.c:100` faults at the
contract point — the delivered fault the capability model promises, in place of
the ASan-detected UAF (which on a stock non-ASan build is a silent read of
freed-but-mapped memory, precisely why this case needs an ASan build to observe).
