# rlua #19 (Row 1) — Use-After-Free across the Lua/Rust boundary

Minimal, deterministic reproduction of `rlua #19`: a Lua `__gc` finalizer
resurrects the userdata it is finalizing, after rlua has already run the Rust
destructor. The resurrected handle then exposes a Rust value whose heap buffer is
freed, and calling a method on it reads through freed memory.

## Vulnerability overview

A Rust value is handed to Lua as full userdata. When Lua collects it, the `__gc`
metamethod calls rlua's `destructor<T>`, which drops the Rust value — freeing the
heap buffer its `String` field owns. But `__gc` receives the object being
finalized, so the script can store it somewhere reachable again (`hatch =
self.userdata`). Lua honours the resurrection and keeps the userdata block alive,
while the Rust value inside it is already dropped. `hatch:access()` then crosses
back into Rust and reads the freed buffer.

The split is the whole bug: **Lua decides the lifetime, Rust owns the memory**,
and a resurrecting finalizer makes "the destructor has run" and "the object is
gone" two different facts. See `boundary.md`.

## Contents

| File | What it is |
|---|---|
| `target.md` | Pinned commit, crash sites, ASan verdict, toolchain note |
| `rlua-modern-rustc.patch` | Two-line shim so rlua builds on rustc ≥1.48; header explains why it does not mask the bug |
| `build.sh` | Clean checkout → patched → native+ASan binary |
| `trigger.lua` | **The trigger.** Resurrection via `__gc`, then the stale access |
| `src/main.rs` | Minimal Rust host: the userdata type and `Lua::eval` |
| `Cargo.toml` | Path dependency on the pinned `rlua` |
| `run.sh` | Runs it, asserts the ASan abort |
| `asan.txt` | Captured ASan report (scrubbed) |
| `boundary.md` | Boundary annotation per task spec §8 |

## Requirements

- **Rust nightly** — `-Zsanitizer=address` is nightly-only.
  `rustup toolchain install nightly`
- `llvm-symbolizer` — without it ASan prints bare hex addresses. `run.sh` finds it
  on `PATH` or falls back to `/usr/lib/llvm-21/bin/llvm-symbolizer`.
- A C compiler, for the vendored Lua that rlua builds.

## How to build and run

```bash
chmod +x build.sh run.sh
./build.sh
./run.sh
```

## Expected outcome

```
collecting...
dropping 123                                  <- Rust destructor ran, buffer freed
hatch =        userdata: 0x...                <- resurrected by __gc
==NNNN==ERROR: AddressSanitizer: heap-use-after-free
READ of size 43 ...
```

Freed at `rlua/src/util.rs:279` (`rlua::util::destructor`, from Lua's `GCTM`);
read at `src/main.rs:35` (the `access` method, from rlua's
`callback_call_impl`). Exit code 1. Deterministic — 10/10 runs.

`PASS = ASan reports heap-use-after-free with the read in Userdata::add_methods::{closure#0}`

## RISC-V QEMU

**Not built for this row.** The rv64 leg would need a cross-compiled Rust
toolchain (`riscv64gc-unknown-linux-gnu`) *and* the vendored Lua C library
cross-built and linked through rlua's `gcc`-crate build script — a stack of
cross-build plumbing well beyond the "stock toolchain" bar the task sets, and
none of it changes what the defect is.

Task spec §9 covers this case: where the QEMU leg is impractical, the native ASan
reproduction plus `boundary.md` is a complete artifact. The mruby rows carry the
RISC-V evidence for the corpus.
