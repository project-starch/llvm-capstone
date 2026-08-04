# Boundary annotation — LuaBridge #319

### The object that crosses the boundary
A non-owning C++ reference (`A&`, returned by `A::fn()`) that LuaBridge pushes to
Lua as `a1`, pointing into an `A` owned by a *different* Lua userdata (the one
`getA()` created).

### Owner vs. borrower
- **The `getA()` userdata (Lua-GC) owns the `A`** by value; the GC frees it (and
  runs `~A`) when it becomes unreachable.
- **`a1` borrows a reference into it**, with no lifetime tie to the owning
  userdata — LuaBridge pushes `A&` as a raw non-owning handle.

### Free site
`collectgarbage("collect")` collects the intermediate `getA()` userdata; its
`__gc` runs `~A`, which sets `i = -1`, then Lua frees the block.

### Stale-use site (one crossing later)
`a1:getI()` dereferences the freed `A` and returns `-1` (the sentinel).

### The lifetime rule that is violated
A reference handed to Lua must pin the object it references (or copy it). Pushing
`A&` for a chained temporary pins nothing, so the reference outlives the owner.

### Capability note (revoke-on-free)
Revoke-on-free revokes the capability to the owning userdata's `A` at GC. `a1`'s
derived capability is revoked with it, so `getI()` faults at the read instead of
returning the `-1` of a destroyed object.
