# RUSTSEC-2022-0070 (Row 7) — use-after-free across the Rust↔C secp256k1 boundary

Minimal, deterministic reproduction of `RUSTSEC-2022-0070`: `secp256k1`'s
`preallocated_gen_new` had an incorrect lifetime bound, so the C context it
returns could outlive the Rust buffer it is constructed in. Every later use
reads freed memory — **from entirely safe Rust, with no `unsafe` in the
reproducer**.

## Run it

```bash
./build.sh && ./run.sh
```

`run.sh` shows both legs. The plain run **completes and prints a signature** —
that is expected, and is evidence about detectability rather than about the
defect. The valgrind leg is the oracle:

```
==307573== Invalid read of size 4
==307573==  Address 0x4b43ce0 is 0 bytes inside a block of size 208 free'd
```

## Requirements

`cargo`, and **`valgrind`** — which is this row's instrument, not an optional
extra. AddressSanitizer is structurally blind here: it sees the Rust-side free
but the offending read executes inside `libsecp256k1`, uninstrumented C. An ASan
build exits 0.

## Notes

The dependency pin is exact and matters. `"0.24.1"` resolves to a **patched**
0.24.3 where the borrow checker rejects the reproducer outright — that compile
error is the fix. 0.24.1, 0.23.4 and 0.22.1 are yanked, so 0.24.0 is the only
fetchable vulnerable release.

Full argument, including why this row replaced the original Row 7 and why a
second candidate was rejected, is in `target.md`.
