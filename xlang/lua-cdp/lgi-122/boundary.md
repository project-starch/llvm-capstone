# Boundary annotation — lgi #122

### The object that crosses the boundary
A boxed `cairo_region_t`, owned and wrapped by a `cairo.Region` lgi record
userdata. `r:get_extents()` marshals the record's stored `record->addr` (the
boxed pointer) across the Lua→C boundary into `cairo_region_get_extents`.

### Owner vs. borrower
- **Lua (managed) owns the region's lifetime** through the record userdata; its
  `__gc` (`record_gc`, `record.c:422`) frees the owned boxed value via
  `record_free` → `g_boxed_free` → `cairo_region_destroy`.
- **The second finaliser borrows** the same record (captured as the upvalue `r`
  in the `{}`-proxy's `__gc`), with no ordering guarantee against the region's
  own finalisation.

### Free site
Main thread, during `collectgarbage("collect")`: the region record is finalised
first → `record_gc` (`record.c:438`) → `record_free` (`record.c:139`) →
`g_boxed_free` → `cairo_region_destroy` frees the `cairo_region_t`.

### Stale-use site (one crossing later)
The `{}`-proxy's `__gc` (`f`) runs next → `r:get_extents()` →
`callable_call` (`callable.c:943`) → `ffi_call` → `cairo_region_get_extents`
reads the freed region's `extents` fields → invalid read of freed memory.

### The lifetime rule that is violated
A finalised record's owned boxed value must not be dereferenced afterwards. lgi's
fix (`94f970d8`) makes the record *unusable* in the Lua VM the moment it is
finalised — `record_gc` nils its metatable — so any later `record:method()`
raises a Lua error instead of marshalling the dangling `record->addr` into cairo.

### Capability note (revoke-on-free)
Revoke-on-free revokes the `cairo_region_t` capability at `record_gc`'s
`g_boxed_free`; the record's stored pointer is revoked too, so
`cairo_region_get_extents` faults at the boundary crossing rather than reading a
freed region — exactly what lgi approximates in software by nilling the metatable.
