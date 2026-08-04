# Boundary annotation — sol2 #1373

### The object that crosses the boundary

A sol2 interior-reference wrapper (`bb`) that points into the storage of a
`ComplexStructC` value owned by a Lua-GC userdata. The interior pointer is what
crosses; `bb` is a live Lua global holding it.

### Owner vs. borrower

- **Lua (managed) owns the storage.** The `ComplexStructC` lives inside a Lua
  userdata; the GC frees it when `csc` becomes unreachable.
- **`bb` (also Lua-side, but a distinct handle) borrows an interior pointer**
  into that storage, with no lifetime tie to the parent userdata.
- sol2 returns member/interior accesses **by reference, not copy** (owner-
  confirmed), so `bb` aliases the parent's memory.

### Free site

`csc = nil` followed by `collect_garbage()` (twice) frees the parent
`ComplexStructC` userdata storage.

### Stale-use site (one crossing later)

`return bb.a.a` → sol2 pushes the interior `ComplexStructA::a` via
`sol::stack::unqualified_pusher<int>::push` (`stack_push.hpp:316`) →
heap-use-after-free READ of the freed storage.

### The lifetime rule that is violated

An interior reference into a GC-owned aggregate must either copy out or pin the
parent for its own lifetime. sol2 does neither by default — "the C++-isms leak
through" — so the interior wrapper outlives the storage it points into.

### Capability note (revoke-on-free)

Revoke-on-free revokes the capability to the `ComplexStructC` block at the GC
free. `bb`'s interior capability (derived from the parent allocation) is then
revoked too, so the `bb.a.a` read faults at the contract point instead of
returning freed bytes.
