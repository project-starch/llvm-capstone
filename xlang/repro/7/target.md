# target.md — Row 7 (RUSTSEC-2022-0070 / GHSA-969w-q74q-9j8v)

* **Defect:** heap use-after-free, READ, reachable from **entirely safe Rust**
* **Product:** `secp256k1` (Rust) over `libsecp256k1` (C) — Rust↔C FFI
* **Advisory:** RUSTSEC-2022-0070, alias GHSA-969w-q74q-9j8v
* **Affected:** `< 0.22.2`, `>= 0.23.0 < 0.23.5`, `>= 0.24.0 < 0.24.2`
* **Pin used:** `secp256k1 = "=0.24.0"`
* **Status:** **REPRODUCES** — valgrind, invalid read inside a freed 208-byte block
* **Instrument:** valgrind (**not** ASan — see below)

## The defect

`Secp256k1::preallocated_gen_new` takes the storage for the context as a mutable
reference and returns a context carrying that reference. The lifetime bound on
the returned type was wrong, so the context could be inferred as `'static` and
outlive the buffer it borrows:

```rust
fn escaped() -> Secp256k1<AllPreallocated<'static>> {
    let mut buf = vec![AlignedType::zeroed(); Secp256k1::preallocate_size()];
    Secp256k1::preallocated_gen_new(&mut buf).unwrap()
}   // `buf` is dropped here; the returned context still points into it
```

Every later use reads the freed context. `PublicKey::from_secret_key` is used in
the trigger because it consults the preallocated generator tables; `sign_ecdsa`
alone does not touch them and produces no report.

**No `unsafe` appears in the reproducer.** The unsoundness is entirely in the
crate's API bounds, which is what makes this a cross-language defect rather than
a Rust bug: the crate's job is to hold a C object's lifetime correctly, and it
does not.

## The version pin is load-bearing

`secp256k1 = "0.24.1"` **silently resolves to 0.24.3**, which is patched — and
there the borrow checker rejects the reproducer with `E0515: cannot return value
referencing local variable`. That compile error IS the fix. The pin must be
exact.

0.24.1, 0.23.4 and 0.22.1 are yanked from crates.io, so **0.24.0 is the only
fetchable vulnerable release**.

## ASan is structurally blind here — verified, not assumed

| build | result |
|---|---|
| plain | completes, prints a valid-looking signature, exit 0 |
| **ASan** (`-Zsanitizer=address`) | **completes, exit 0, no report** |
| **valgrind** | **Invalid read of size 4 and 8, inside a 208-byte block free'd — exit 9** |

ASan sees the *free* (the `Vec` is a Rust allocation) but the *read* executes
inside `libsecp256k1`, C compiled by the `cc` crate and carrying no
instrumentation. Same structural blindness as Row 3, and it generalises the same
way: **for Rust→C rows where the stale dereference lands in C, ASan has nothing
to check even though it poisoned the region.**

## Why this row replaces the original Row 7

The original ("mruby #6701 / `mrb_bint_reduce`") does not exist — its issue
number belongs to Row 6, the function is absent from the versions the spec
assigns, and the GC hazard it describes is closed by the allocation arena. That
analysis is kept at `../7-old-sortbang/target.md`, which also documents a second
rejected candidate (CVE-2025-13120, `Array#sort!`) and why: its "free" is a
shrink-in-place, so there is no free for revocation to observe, and both ASan and
valgrind mask it by replacing `realloc` with an always-moving implementation.

This row was chosen because it has what those lacked: **a real free** (a
208-byte block) that a revocation mechanism can act on, which is the property
the CHERI and Capstone columns measure.

## Boundary

See `boundary.md`. The free is `Vec::drop` on the Rust side; the offending read
is a load inside C, through a pointer the C context still holds.
