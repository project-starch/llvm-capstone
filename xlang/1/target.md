# target.md — rlua #19 (Row 1)

* **CVE/Issue:** rlua #19
* **Product:** rlua (Lua-in-Rust binding), crate version `0.8.2`
* **Vulnerability Type:** Heap Use-After-Free of Rust-owned memory, via Lua `__gc`
  finalizer resurrection across the FFI boundary
* **Status:** REPRODUCED (native ASan)
* **Vulnerable Commit:** `396a4b09169be429381d5db8e9bb1cd6bd5d5139`
* **Free Site:** `rlua/src/util.rs:279` (`rlua::util::destructor`), driven by Lua's
  `GCTM` / `runafewfinalizers`
* **Stale-Use Site:** `src/main.rs:35` (the `access` userdata method), reached from
  Lua via rlua's `callback_call_impl`
* **ASan Verdict:** `heap-use-after-free`, `READ of size 43`, offset 0 of the freed
  43-byte region
* **Determinism:** aborts on 10/10 consecutive runs; the free is forced at a fixed
  point by `collectgarbage("collect")`.
* **RISC-V QEMU:** not built — see `README.md`. Task spec §9 permits native-ASan-only
  artifacts where the QEMU leg is impractical.

## Toolchain note — requires a patch, which does not mask the defect

rlua at this commit calls `std::mem::uninitialized()`. Since rustc 1.48 that
aborts the process at runtime (non-unwinding panic: "attempted to leave type ...
uninitialized, which is invalid"), during `LuaTable::set` — before the trigger
ever reaches Lua's collector.

`rlua-modern-rustc.patch` replaces both call sites with `std::ptr::read`, applied
by `build.sh`. One of the two sites is the userdata destructor itself, so the
substitution was chosen to preserve the defect: `ptr::read` moves out and drops
exactly as `mem::replace(_, mem::uninitialized())` did, and likewise leaves the
slot unmarked, so a second `__gc` still double-drops. See the patch header for
the full argument.

This is also why the row needs **nightly** Rust: `-Zsanitizer=address` is
nightly-only.

> **Supersedes an earlier SKIPPED filing.** This row was previously recorded as
> infeasible on the grounds that modern rustc's `mem::uninitialized` ban made
> "runtime-based Use-After-Free/Double-Free verification infeasible". The ban is
> real and does abort the unpatched build, but it is a two-line toolchain
> incompatibility in rlua's internals rather than a property of the vulnerability,
> and removing it leaves the bug intact and reproducible.
