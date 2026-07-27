# Language Boundary Violation — GHSA-f56g-chqp-22m9 (Row 3)

### The object that crosses the FFI

A **C `pa_proplist`** — a PulseAudio property-list object (internally a
`pa_hashmap`), allocated by `pa_proplist_new()` inside `libpulse.so`. The Rust side
holds it as a raw `*mut ProplistInternal` inside a `Proplist` wrapper, and hands a
*copy of that raw pointer* to a second Rust object, `proplist::Iterator`.

This is the N→N′ case the other rows do not cover: both participants are Rust
types, but the object they contend over is owned by a C library, and the stale
dereference executes inside that library.

### Owner vs. borrower

- **Rust owns the lifetime.** `Proplist` has sole responsibility for destruction via
  its `Drop`, which calls `pa_proplist_free()`.
- **C owns the memory.** libpulse allocated the block (`pa_xmalloc`) and frees it
  (`pa_xfree`); Rust only decides *when*.
- **`proplist::Iterator` is the borrower** — it holds a raw copy of the same pointer
  and, at 2.4.0, carries no lifetime relating it to the `Proplist`. Nothing in the
  type system prevents the owner from being destroyed first.

Note the inversion relative to Rows 1 and 2. There, a *managed* runtime's collector
decided lifetimes Rust could not predict. Here Rust is fully in charge of the
lifetime and still gets it wrong, because the pointer was copied out from behind a
`&self` borrow into a struct with no lifetime parameter — so the borrow checker,
which would have caught this, was never given anything to check.

### Free site

`<Proplist as Drop>::drop` — `pulse-binding/src/proplist.rs:453`

```rust
if !self.weak { unsafe { capi::pa_proplist_free(self.ptr) }; }
```

reached from `<Proplist as IntoIterator>::into_iter` (`proplist.rs:186`), which
takes `self` by value and drops it on return. Valgrind attributes the free to
`pa_xfree` inside `libpulsecommon-17.0.so`, a 1,072-byte block.

### Stale-use site

`<proplist::Iterator as Iterator>::next` — `proplist.rs:171`

which calls `pa_proplist_iterate()` (`libpulse.so.0.24.3`) →
`pa_hashmap_iterate()` (`libpulsecommon-17.0.so`). Valgrind: `Invalid read of
size 8`, 32 bytes inside the freed block.

### The lifetime rule that is violated

An iterator over a C object must not outlive the Rust wrapper that owns and frees
that object — but `iter()` copies the raw pointer out from behind its `&self`
borrow into an `Iterator` with no lifetime parameter, erasing the association. The
`IntoIterator` impl then frees the owner while handing the iterator back, so the
iterator is dangling on return.

### Note on the capability phase (not implemented here)

Clean revocation target, and an instructive one because the fault executes in
**prebuilt C code**. A capability for the `pa_proplist` allocation, derived into
the iterator and revoked at `proplist.rs:453`, would make `pa_hashmap_iterate`'s
load fault — without recompiling PulseAudio, since the check rides on the pointer
rather than on instrumentation at the access site.

That contrast is the reason this row is worth keeping: it is exactly the case where
the sanitizer-based tooling is blind (see `target.md`, "Why the evidence is
valgrind") and a hardware capability would not be.
