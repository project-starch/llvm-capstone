# target.md — rlua #97 (Row 2)

* **CVE/Issue:** rlua #97
* **Product:** rlua (Lua-in-Rust binding), crate version `0.15.4-alpha.0`
* **Vulnerability Type:** Unsound public API — an unconstrained callback lifetime
  lets a Lua handle escape the callback and be used after the `Lua` state is
  dropped (temporal borrow violation)
* **Status:** REPRODUCED (native ASan)
* **Vulnerable Commit:** `78021185d55c562e48a7bd76582f42c8b3326cbc`
* **Fix Commit:** `0f5a9a3ea8a2819380c061dcd03ca907ab6ea2c5`
  ("Fix terrible soundness issue... barely")
* **Stale-Use Site:** `<rlua::table::Table>::len` — the escaped handle, read after
  `drop(lua)` in `src/main.rs`
* **ASan Verdict:** `stack-use-after-return`, `READ of size 8`
* **Determinism:** aborts on 10/10 consecutive runs. No GC timing or allocation
  layout involved — the escape and the drop are both explicit.
* **RISC-V QEMU:** not built — see `README.md`. Task spec §9 permits
  native-ASan-only artifacts where the QEMU leg is impractical.

## Mechanism

At the vulnerable commit `Lua::create_function` reads:

```rust
pub fn create_function<'lua, 'callback, A, R, F>(&'lua self, func: F) -> Result<Function<'lua>>
where
    A: FromLuaMulti<'callback>,
    R: ToLuaMulti<'callback>,
    F: 'static + Send + Fn(&'callback Lua, A) -> Result<R>,
```

`'callback` appears only in the bounds and is otherwise **unconstrained**, so the
*caller* picks it — including `'static`. The callback's arguments are handles into
the Lua state, but their lifetime no longer has to relate to the `&'lua self`
borrow that produced them. A callback can therefore move a `Table<'static>` into
storage that outlives the state.

The fix deletes `'callback` and ties everything to `'lua`:

```rust
pub fn create_function<'lua, A, R, F>(&'lua self, func: F) -> Result<Function<'lua>>
where
    A: FromLuaMulti<'lua>,
    R: ToLuaMulti<'lua>,
    F: 'static + Send + Fn(&'lua Lua, A) -> Result<R>,
```

## The trigger is upstream's own compile-fail test

`src/main.rs` is the file the fix commit added as
`tests/compile-fail/static_callback_args_tls.rs`, kept verbatim apart from
comments. Two things follow:

1. **That it compiles is half the bug.** Against a fixed rlua this program must
   fail to compile (`lua does not live long enough`). Compiling successfully at
   the pinned commit *is* the unsoundness.
2. The runtime crash is the other half: the escaped `Table` is read after
   `drop(lua)`, and ASan aborts in `Table::len`.

Upstream's own comment in that test — *"In debug, this will panic with a reference
leak before getting to the next part but it segfaults anyway"* — matches what we
see, which is why `build.sh` builds `--release`: a debug build trips rlua's
internal reference-leak assertion (`src/lua.rs:46`) first and buries the trace.

## Note on the ASan class

ASan reports **`stack-use-after-return`**, not `heap-use-after-free`. The escaped
handle refers into state whose frame has returned, so that is the class ASan
attributes it to. It is still a **temporal** violation — use after end of lifetime
— which is the class the benchmark cares about, unlike Rows 6 and 11 which turned
out spatial.

Heap-allocating the state (`Box::new(Lua::new())`) was tried and gives the same
`stack-use-after-return` in `Table::len`, so this is not an artifact of where the
`Lua` value happened to live.

## Toolchain note — requires a patch, unrelated to the defect

`rlua-modern-rustc.patch` removes a trailing semicolon from the `rlua_panic!`
macro body. The macro is used in expression position, which modern rustc rejects
outright (6 errors, all from that one line). Unlike Row 1's patch, this one is
nowhere near the code under test — it touches only panic formatting in an internal
assertion macro. See the patch header.

Requires **nightly** Rust for `-Zsanitizer=address`.

> **Supersedes an earlier SKIPPED filing, and corrects a bad pin.** The previous
> rationale reused Row 1's reasoning ("modern Rust bans `mem::uninitialized`");
> that does not even apply here, as this commit contains no `mem::uninitialized`
> in `src/`.
>
> More seriously, the old `build.sh` pinned **`4be78cb10177`, which is not an rlua
> commit at all** — it is a commit from the llvm-capstone repository
> ("ladder-perf runner: rebuild doms by default …") pasted in by mistake. The
> checkout was wrapped in `|| true`, so it failed silently and the build used
> whatever rlua HEAD happened to be (v0.20.1, a modern mlua-backed rewrite), while
> `src/main.rs` contained only a placeholder that printed a string. Nothing about
> the row was ever exercised.
