# boundary.md — Row 7 (RUSTSEC-2022-0070)

**Boundary:** Rust → C. `secp256k1` (safe Rust API) over `libsecp256k1` (C).

| | |
|---|---|
| Allocated by | Rust — `Vec<AlignedType>`, 208 bytes, in `escaped()` |
| Handed across | `Secp256k1::preallocated_gen_new(&mut buf)` — the C context is constructed *in* that Rust buffer |
| Freed by | Rust — `Vec::drop` at the end of `escaped()` |
| Offending access | a load **inside C** (`libsecp256k1`), through the pointer the context still holds |
| Detected by | valgrind (ASan is blind: the read is in uninstrumented C) |

## Why it crosses the boundary

The C library never allocates or frees here — it is handed storage owned by
Rust and told to treat it as its context. Ownership therefore lives entirely on
the Rust side, and the crate's type signature is the only thing preventing the C
object from outliving its storage. The wrong lifetime bound removes that
protection, and the result is reachable **without writing any `unsafe`**.

This is the same failure class as Rows 1–3: a safe-language wrapper losing track
of a foreign object's lifetime. It differs from Row 2 in being a *heap*
use-after-free rather than a stack use-after-return, and from Row 3 in that the
freed memory is owned by Rust rather than by the C library.

## What a capability mechanism would do at the free site

The free is an ordinary heap deallocation of a 208-byte block, so it is exactly
the event a revocation scheme observes. On a revoke-on-free allocator, the
capability the C context holds is invalidated at `Vec::drop`, and the first read
through it faults at the contract point rather than returning plausible garbage
— which is what the plain run does today.

Nothing is implemented here; per task spec §9 this row uses a stock toolchain
only. The measurement lives in `../../cheri/` and `../../capstone/`.
