# Language Boundary Violation — rlua #19 (Row 1)

### The object that crosses the FFI

A Rust `Userdata` value — owning a heap buffer through its `String` field — is
handed to Lua as **full userdata** by `LuaTable::set`. Lua receives a block it
allocated itself, into which rlua moves the Rust value; the Rust value's heap
buffer stays in the Rust allocator. So two allocations are in play, with split
responsibility: Lua owns the *userdata block*, Rust owns the *buffer inside it*.

### Owner vs. borrower

- **Lua (managed) owns the lifetime decision.** Its collector decides when the
  userdata is unreachable and when to run the `__gc` metamethod. It also decides
  whether to honour a resurrection performed *inside* that metamethod.
- **Rust (native) owns the memory.** `Drop for Userdata` frees the `String`
  buffer, and rlua's `destructor<T>` is what the collector invokes to do it.
- The borrow that dangles is the one Lua keeps: after `__gc` has run the Rust
  destructor, Lua still holds a live handle to the userdata block and will hand
  it to any Rust method the script calls.

### Free site

`rlua::util::destructor::<RefCell<Userdata>>` — `rlua/src/util.rs:279`

Reached from the Lua collector: `runafewfinalizers` → `GCTM` (`lua/lgc.c:822`)
→ `luaD_precall` (`lua/ldo.c:434`) → `destructor` → `drop_glue::<RawVec<u8>>` →
`free`. The 43-byte `String` buffer is released here.

### Stale-use site

`<Userdata as LuaUserDataType>::add_methods::{closure#0}` — `src/main.rs:35`

Reached back across the boundary: `Lua::eval` → `lua_pcallk` (`lua/lapi.c:968`)
→ `luaV_execute` (`lua/lvm.c:1134`) → `luaD_precall` → rlua's
`callback_call_impl` (`rlua/src/lua.rs:1318`) → the `access` method, which reads
`this.payload` and faults. ASan: `heap-use-after-free`, `READ of size 43`, at
offset 0 of the freed 43-byte region.

### The lifetime rule that is violated

Lua's `__gc` finalizer may resurrect the object it is finalizing by storing it
somewhere reachable, and Lua then keeps the userdata block alive — but rlua has
already run the Rust destructor, so the resurrected handle exposes a value whose
heap memory is freed, and any subsequent method call reads through it.

Equivalently, in the terms rlua's maintainers used: userdata handed to Lua must
be `'static` and its destruction must be driven by something Rust can see,
because Lua's collection time is unknowable from Rust — and a finalizer that can
resurrect makes "the destructor has run" and "the object is gone" two different
facts.

### Note on the capability phase (not implemented here)

The free site and the stale-use site are one boundary crossing apart, which is
what makes this row a clean revocation target: a capability held by the Lua
userdata block, derived from the Rust allocation and revoked at `util.rs:279`,
would make the `main.rs:35` read fault instead of returning freed bytes.
Resurrection would then yield an unusable handle rather than a live one.
