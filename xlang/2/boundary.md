# Language Boundary Violation — rlua #97 (Row 2)

### The object that crosses the FFI

A **Lua handle** — here a `Table` — created inside the Lua state and passed into a
Rust callback as an argument. Concretely it is a reference-counted slot in rlua's
Lua-side registry plus a borrow of the `Lua` context that owns it. It crosses
N→M→N: Rust registers a callback, Lua calls it, and the handle arrives on the Rust
side as a typed value.

### Owner vs. borrower

- **Lua (managed) owns the object.** The table lives in the Lua state; closing the
  state destroys it and the registry slot with it.
- **Rust (native) is the borrower.** The callback receives the handle for the
  duration of the call, and nothing more — that is the contract the type system is
  supposed to encode.
- The type system fails to encode it. Because `'callback` is unconstrained, the
  borrow can be typed `'static`, so Rust believes it may keep the handle forever
  while Lua's state is free to disappear.

### Free site

`drop(lua)` — `src/main.rs`

Dropping the `Lua` closes the Lua state and releases the registry, invalidating
every outstanding handle. rlua's `Lua::drop_ref` / `src/lua.rs` teardown is what
runs. Nothing here consults the escaped `Table`, because nothing knows it exists.

### Stale-use site

`<rlua::table::Table>::len` — reached from `src/main.rs` via the thread-local
`BAD_TIME`

The escaped handle dereferences state belonging to the closed `Lua`. ASan:
`stack-use-after-return`, `READ of size 8`.

### The lifetime rule that is violated

A Lua handle handed to a callback is valid only for that call, because the Lua
state may be closed at any point afterwards — but the vulnerable signature's
unconstrained `'callback` lifetime lets the caller instantiate it as `'static`, so
the compiler permits the handle to be stored beyond the call and used after the
state is gone.

This is the mirror image of Row 1. Row 1 is the *managed collector* freeing
Rust-owned memory the Rust side still holds; Row 2 is the *Rust side* retaining a
managed handle past the managed object's death. Same boundary, opposite direction,
and neither language's own guarantees see across it: Lua has no idea Rust kept a
reference, and Rust's borrow checker was handed a lifetime that lied.

### Note on the capability phase (not implemented here)

This is a clean revocation target. Model each handle handed across the boundary as
a capability derived from the Lua state, and revoke the whole derivation tree when
the state closes at `drop(lua)`. The `Table::len` read then faults instead of
walking freed state — and, unlike a purely static fix, it holds even though the
type system was told the wrong lifetime. That is the interesting part for the
benchmark: the *compiler* was lied to, so only a runtime mechanism catches it.
