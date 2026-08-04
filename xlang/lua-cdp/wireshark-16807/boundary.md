# Boundary annotation — Wireshark #16807

### The object that crosses the boundary

A raw pointer to a C `tvbuff` (the dissection engine's testy-virtual-buffer for a
packet's bytes), stored inside the Lua `TvbRange` userdata that `buffer(0,16)`
returns. The `TvbRange` is the Lua-visible handle; the `tvbuff` pointer is what
crosses into Lua and is retained past its owner's lifetime.

### Owner vs. borrower

- **The C side (`libwireshark`) owns the memory.** `tvb_new_subset_remaining`
  (`g_malloc`, via `dissect_tcp_payload`) creates the `tvbuff`;
  `epan_dissect_reset` → `tvb_free_chain` frees it when the current dissection is
  torn down (between passes, or on the next GUI re-dissection).
- **Lua (managed) owns the handle.** The `TvbRange` userdata's lifetime is the
  Lua GC's; because `handle_packet()` stores it in the global `ProtocolState`
  table, it long outlives the `tvbuff` it wraps.
- The bug: a `TvbRange`/`Tvb` is only valid during the dissection call that
  produced it, but nothing invalidates the stored handle when its `tvbuff` is
  freed, so the next dissection reuses a dangling pointer.

### Free site

`epan_dissect_reset` → `tvb_free_chain` (`epan/tvbuff.c`) — the engine frees the
packet's tvbuff chain when resetting the `edt` between the two analysis passes
(GUI: when re-dissecting the newly-selected packet). Native-frees.

### Stale-use site (one crossing later)

Re-dissection of the same `pinfo.number`: `ProtocolState[id]` is already set, so
`handle_packet()` is skipped and the stale `TvbRange` is passed to
`subtree:add(foo_field, staleRange)` → wslua `TreeItem:add` →
`proto_tree_add_item_new` → `tvb_ensure_bytes_exist` reads the freed `tvbuff`
(valgrind: Invalid read of size 1 and 4). In the issue's 3.2.6 the same add path
reached `tvb_offset_from_real_beginning`.

### The lifetime rule that is violated

A managed handle that wraps a native resource is valid only while that resource
lives; it must not be retained past the callback that produced it, and the
runtime must fault cleanly rather than dereference a freed pointer if it is.
wslua *does* guard `TvbRange` **accessor methods** (`:bytes()`, `:uint()`, …)
with an "expired tvb" check, but the `TreeItem:add(field, range)` fast-path
observed here hands the raw `tvbuff` straight to `proto_tree_add_item_new`
without that check — so the guard is bypassed and the UAF still fires on 4.6.4.
The correct fix at the *script* level (our `trigger_fixed.lua`) is to copy the
bytes out or re-derive the range from the live `buffer` each dissection.

### Capability note (revoke-on-free)

On a revoke-on-free allocator, `tvb_free_chain` at `epan_dissect_reset`
**revokes** the capability to the `tvbuff`. The stored `TvbRange` then holds a
revoked capability, so `tvb_ensure_bytes_exist` faults at the contract point on
the next dissection — the delivered fault the capability model promises, in place
of the valgrind-detected (and on stock hardware, silent-until-corrupted) UAF.
