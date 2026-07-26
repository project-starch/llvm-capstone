# rlua #97 (Row 2) — callback handle escapes its lifetime, used after the state dies

Minimal, deterministic reproduction of `rlua #97`: at the pinned commit
`Lua::create_function` has an **unconstrained callback lifetime**, so a Lua handle
passed into a Rust callback can be typed `'static` and stashed somewhere that
outlives the Lua state. Dropping the state and then reading the handle is a
use-after-lifetime of Lua-owned memory.

## Vulnerability overview

The vulnerable signature:

```rust
pub fn create_function<'lua, 'callback, A, R, F>(&'lua self, func: F) -> Result<Function<'lua>>
where
    A: FromLuaMulti<'callback>,          // 'callback is unconstrained --
    R: ToLuaMulti<'callback>,            // the CALLER picks it, including 'static
    F: 'static + Send + Fn(&'callback Lua, A) -> Result<R>,
```

Callback arguments are handles into the Lua state, but their lifetime is no longer
tied to the `&'lua self` borrow that produced them. The fix (`0f5a9a3`) removes
`'callback` and ties everything to `'lua`.

This is the mirror image of Row 1: there, Lua's collector freed Rust-owned memory
Rust still held; here, Rust retains a Lua handle past the Lua object's death.

## The trigger is upstream's own compile-fail test

`src/main.rs` is `tests/compile-fail/static_callback_args_tls.rs`, the test the fix
commit added — kept verbatim apart from comments. **That it compiles is half the
bug**: against a fixed rlua it must fail with `lua does not live long enough`. The
runtime abort is the other half.

## Contents

| File | What it is |
|---|---|
| `target.md` | Pinned commit, mechanism, ASan verdict, and the corrected-pin note |
| `rlua-modern-rustc.patch` | Removes a trailing semicolon from `rlua_panic!` so the crate compiles on current rustc; header explains why it is far from the defect |
| `build.sh` | Clean checkout of `78021185` → patched → nightly ASan release build |
| `src/main.rs` | **The trigger** (upstream PoC): TLS escape, `drop(lua)`, stale read |
| `Cargo.toml` | Path dependency on the pinned `rlua` |
| `run.sh` | Runs it, asserts the ASan abort |
| `asan.txt` | Captured ASan report (scrubbed) |
| `boundary.md` | Boundary annotation per task spec §8 |

## Requirements

- **Rust nightly** — `-Zsanitizer=address` is nightly-only.
- `llvm-symbolizer` — otherwise ASan prints bare hex. `run.sh` finds it on `PATH`
  or falls back to `/usr/lib/llvm-21/bin/llvm-symbolizer`.
- A C compiler, for the vendored Lua that rlua builds.

## How to build and run

```bash
chmod +x build.sh run.sh
./build.sh
./run.sh
```

## Expected outcome

```
==NNNN==ERROR: AddressSanitizer: stack-use-after-return
READ of size 8 ...
SUMMARY: AddressSanitizer: stack-use-after-return ... in <rlua::table::Table>::len
```

Exit code 1. Deterministic — 10/10 runs.

`PASS = ASan aborts with the stale read in <rlua::table::Table>::len`

**Why `--release`:** a debug build trips rlua's internal reference-leak assertion
(`src/lua.rs:46`) before reaching the read, burying the trace. Upstream noted the
same thing in the PoC. `-g` keeps symbols so the frame is still named.

**On the ASan class:** the report says `stack-use-after-return`, not
`heap-use-after-free` — the escaped handle refers into state whose frame has
returned. It is still a *temporal* violation, which is the class the benchmark
cares about. `Box::new(Lua::new())` was tried and reports the same class, so this
is not an artifact of where the `Lua` value lived.

## RISC-V QEMU

**Not built for this row**, for the same reason as Row 1: the rv64 leg needs a
cross-compiled Rust toolchain plus rlua's vendored Lua C library cross-built
through its `gcc`-crate build script — well beyond the stock-toolchain bar, and it
would not change what the defect is. Task spec §9 accepts the native ASan
reproduction plus `boundary.md` as complete; the mruby rows carry the RISC-V
evidence for the corpus.
