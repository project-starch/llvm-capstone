# GHSA-f56g-chqp-22m9 (Row 3) — Rust→C use-after-free in libpulse-binding

Minimal, deterministic reproduction of the `libpulse-binding` **2.4.0**
`proplist::Iterator` use-after-free: the iterator holds a raw copy of the C
`pa_proplist*` with no lifetime tie to the `Proplist` that owns it, so calling
`into_iter()` frees the C object and hands back an already-dangling iterator.

## Vulnerability overview

```rust
pub struct Iterator {                  // no lifetime parameter
    ptr: *const ProplistInternal,      // raw copy of the C pointer
    ...
}
pub fn iter(&self) -> Iterator { Iterator::new(self.ptr) }   // borrow not tracked

impl IntoIterator for Proplist {
    fn into_iter(self) -> Self::IntoIter { self.iter() }     // `self` dropped here
}                                                            // -> pa_proplist_free()
```

Upstream's advisory: *"There was no actual lifetime association … linking the
lifetime of the `Iterator` object to the `Proplist` object … it was possible for the
`Proplist` object to be destroyed first, leaving the `Iterator` object working on a
freed C object … trivial to achieve, including simply by using the `into_iter()`
function."* Fixed in 2.5.0 by adding `Iterator<'a>` with
`PhantomData<&'a ProplistInner>`.

Unlike Rows 1 and 2, no garbage collector is involved: **Rust** owns the lifetime
and still gets it wrong, because the pointer was copied out from behind a `&self`
borrow into a struct with no lifetime — so the borrow checker had nothing to check.

No PulseAudio server is needed; `pa_proplist` is a standalone data structure.

## Contents

| File | What it is |
|---|---|
| `target.md` | Pinned commit, mechanism, both crash sites, and why valgrind is the tool |
| `build.sh` | Clean checkout of `b6b1010` → native build (+ libpulse check) |
| `src/main.rs` | **The trigger**: populate a `Proplist`, `into_iter()`, then `next()` |
| `Cargo.toml` | Path dependency on the pinned `libpulse-binding` |
| `run.sh` | Runs ASan (clean, by design) then valgrind (catches it); asserts |
| `asan.txt` | Captured **valgrind** report — see below |
| `boundary.md` | Boundary annotation per task spec §8 |

## Requirements

- **Rust nightly** — used for the ASan leg of `run.sh`.
- **`valgrind`** — required; this is the tool that actually detects the bug.
- **libpulse development files** (`libpulse-dev`). Verified at libpulse 17.0.

## How to build and run

```bash
chmod +x build.sh run.sh
./build.sh
./run.sh
```

## Expected outcome

```
==NNNN== Invalid read of size 8
==NNNN==    at 0x...: pa_hashmap_iterate (in .../libpulsecommon-17.0.so)
==NNNN==    by 0x...: pa_proplist_iterate (in .../libpulse.so.0.24.3)
==NNNN==    by 0x...: <...proplist::Iterator as ...Iterator>::next (proplist.rs:171)
==NNNN==    by 0x...: libpulse_repro::main (main.rs:51)
==NNNN==  Address 0x... is 32 bytes inside a block of size 1,072 free'd
==NNNN==    by 0x...: <...Proplist as ...Drop>::drop (proplist.rs:453)
==NNNN==    by 0x...: <...Proplist as ...IntoIterator>::into_iter (proplist.rs:186)
==NNNN==    by 0x...: libpulse_repro::main (main.rs:47)
```

valgrind exits **9**. Deterministic — 10/10 runs.

`PASS = valgrind reports an invalid read in pa_proplist_iterate, with the free
attributed to Proplist::drop reached from into_iter`

## Why valgrind and not AddressSanitizer

**The ASan build exits 0, and that is expected.** The stale dereference executes
inside `libpulse.so` / `libpulsecommon.so` — distro shared libraries with no
sanitizer instrumentation. ASan checks only the loads and stores it instrumented;
it poisons the region on `free()` but never sees `pa_proplist_iterate()` read it.
Valgrind instruments at the machine level and catches the access wherever it comes
from.

`run.sh` deliberately runs both legs so the clean ASan result is visible and not
mistaken for a failed reproduction. This generalises: for Rust→C rows where the
stale dereference lands in a prebuilt C library, ASan is structurally blind.

## RISC-V QEMU

**Not built for this row.** It would need a cross-compiled Rust toolchain plus a
riscv64 build of PulseAudio to link against. Task spec §9 accepts the native
reproduction plus `boundary.md` as a complete artifact; the mruby rows carry the
RISC-V evidence for the corpus.
