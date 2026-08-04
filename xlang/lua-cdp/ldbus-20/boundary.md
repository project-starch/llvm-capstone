# Boundary annotation — ldbus #20

### The object that crosses the boundary
A raw pointer into a C `DBusMessage` (a reply), embedded in a `DBusMessageIter`
that a Lua userdata wraps. Per libdbus, `dbus_message_iter_init` makes the
iterator reference the message, but **without** a refcount.

### Owner vs. borrower
- **The reply `DBusMessage` (its own Lua wrapper, Lua-GC) owns the C object;** its
  `__gc` calls `dbus_message_unref`.
- **The iterator userdata borrows** a pointer into it, with no refcount and no Lua
  reference to the message wrapper — so the message can be collected first.

### Free site
The unstored reply wrapper becomes unreachable; `collectgarbage` runs its `__gc`
→ `dbus_message_unref` → the C `DBusMessage` is freed (returned to libdbus's pool).

### Stale-use site (one crossing later)
`iter:get_arg_type()` → `ldbus_message_iter_get_arg_type` (`src/message_iter.c`)
→ `dbus_message_iter_get_arg_type(iter)` reads the freed/reused message.

### The lifetime rule that is violated
An iterator that references a foreign object must keep that object alive for its
own lifetime. The fix adds `lDBusMessageIter { DBusMessageIter; DBusMessage*; }`
and refs/unrefs the message in `__gc`.

### Capability note (revoke-on-free)
Revoke-on-free revokes the capability to the `DBusMessage` block at
`dbus_message_unref`. The iterator's derived capability is revoked too, so
`get_arg_type` faults at the read rather than returning pooled/garbage data — a
strictly sharper signal than the pool-masked behaviour seen here.
