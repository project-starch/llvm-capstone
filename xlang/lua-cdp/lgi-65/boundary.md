# Boundary annotation — lgi #65

### The object that crosses the boundary
A `GArray` (from `g_array_sized_new`) marshalled from a Lua table by
`marshal_2c_array` (`marshal.c:383`). Its data segment is installed into the C
struct field `DBusInterfaceInfo.methods`; the `GArray` itself is owned by a Lua
guard userdata.

### Owner vs. borrower
- **The C struct borrows** the array: `iface.methods` is set with
  `GI_TRANSFER_EVERYTHING` (`lgi_marshal_field`, `marshal.c:1555`), i.e. the field
  is meant to take ownership of the data.
- **Lua (managed) owns the `GArray`** through the guard userdata; on the
  vulnerable tree the guard `__gc` (`guard_gc`, `core.c:256`) `g_array_unref`s the
  whole array — container *and* data — leaving the struct field dangling.

### Free site
Main thread, during `collectgarbage()`: the unanchored guard is collected →
`guard_gc` (`core.c:256`) → `g_array_unref` frees the `GArray` and its data
segment (the same memory `iface.methods` points at).

### Stale-use site (one crossing later)
`print(iface.methods)` → `record_field` → `lgi_marshal_field` (`marshal.c:1549`,
get mode) → `marshal_2lua_array` (`marshal.c:562`) walks the freed `GArray` (reads
`->len` and each element pointer) → invalid read of freed memory. (The freed field
is also re-derefed at teardown by `g_dbus_interface_info_unref`.)

### The lifetime rule that is violated
An array whose ownership is transferred into a C struct field must not also be
freed by the Lua guard. lgi's fix (`358371fd`) introduces `array_detach`
(`g_array_free(array, FALSE)`) for `GI_TRANSFER_EVERYTHING` field sets, so the
guard releases only the container and the data stays owned by the struct.

### Capability note (revoke-on-free)
Revoke-on-free revokes the `GArray` data capability at `guard_gc`'s free; the copy
installed in `DBusInterfaceInfo.methods` is revoked too, so `marshal_2lua_array`
faults at the field read instead of walking a freed array — the fault the fix
avoids by transferring, rather than freeing, the data.
