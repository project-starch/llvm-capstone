# tarantool #1955 — LuaJIT cdata ⟷ C `struct error` use-after-free

**One line.** `lbox_error` reads the fiber's last error out of the diagnostics
area and wraps it in a LuaJIT cdata (`error_ref`); a GC-invoked finalizer
(`lbox_fiber_cancel`) clears the fiber diagnostics — running `error_unref` →
`exception_destroy`, freeing that `struct error` — so the in-flight
`luaL_pusherror`/`error_ref` derefs freed memory. ASan heap-use-after-free.

## Identity

| | |
|---|---|
| Library | [Tarantool](https://github.com/tarantool/tarantool) core error/diag + bundled LuaJIT |
| Language pair | **C ⟷ LuaJIT**. The coupled resource is a first-class C DBMS object (`struct error`), not FFI plumbing — a strong CDP case. |
| Upstream | https://github.com/tarantool/tarantool/issues/1955 |
| Vulnerable build | ~1.10.2 era (the ASan report predates the fix milestone). |
| Fix | Hardened for the **1.10.3** milestone (diag/error refcount + testcancel ordering). |
| Native dep | Tarantool built **from source with AddressSanitizer** + bundled LuaJIT + submodules. |
| Detect | ASan heap-use-after-free (READ of size 4 in `error_ref`). |

## The two coupled objects (why unambiguous CDP)

1. **Lua-GC handle:** a LuaJIT cdata wrapping a `struct error *`, produced by
   `luaL_pusherror` / `lbox_error` (`src/lua/utils.c`) after `error_ref`.
2. **Separate native resource:** the C `struct error` (a `FiberIsCancelled`
   exception here), allocated by `BuildFiberIsCancelled` (`src/exception.cc:264`)
   inside `luaL_testcancel` (`src/lua/fiber.c:45`) and destroyed by
   `exception_destroy` (`src/exception.cc:45`) via `error_unref` when its
   refcount reaches 0.

Two distinct allocations. **Not** the string-borrow shape.

**Direction:** GC-frees. A GC pass runs `lbox_fiber_cancel` (`src/lua/fiber.c:459`),
which clears the fiber diagnostics (crossing 1: the cdata/finalizer side);
`diag_clear` → `error_unref` → `exception_destroy` frees the `struct error`
(crossing 2); the still-running `error_ref` in `luaL_pusherror` reads it.

## Reproduction status

**BLOCKED.**

**Exact reason.** Requires a full Tarantool **from-source build (bundled LuaJIT +
many submodules) with AddressSanitizer**; impractical in this sandbox. Two things
compound it, both verified here:

1. **Detection is ASan-only.** This bug manifests as an ASan heap-use-after-free
   *READ*, not a hard crash. The official prebuilt release Docker images are
   optimized non-ASan builds — a benign UAF read on freed-but-still-mapped memory
   does **not** fault there. (Directly demonstrated on the sibling case #7657:
   its genuinely-vulnerable release image printed the correct result and did not
   crash until the free was forced through a nulled vtable. #1955 has no such
   vtable hook — the freed `struct error` is read, not called — so there is no
   release-build path to a deterministic fault.)
2. **No minimal reproducer exists.** The issue ships an ASan trace from a *netbox
   fiber-cancel race*, not a minimal script. Triggering the free/use ordering
   deterministically from Lua would itself be a research task, and would still
   need ASan to observe.

The vulnerable code is ~1.10.2 (2018-era); an ASan from-source build of it —
bundled LuaJIT, many submodules, old dependency pins — on a modern
gcc-15/Debian-13 toolchain is a multi-hour, likely-patch-heavy effort. Per the
corpus's reality-check discipline, not sunk here. **Upstream-verified** via the
filed ASan trace (quoted in `evidence.txt`).

**What a dedicated env would need.** A pinned ~1.10.2 tarantool source tree built
`-DENABLE_ASAN=ON` (or `-fsanitize=address` across core + bundled LuaJIT) with the
era-correct submodule/dep versions, plus a driver that reproduces the netbox
fiber-cancel path (start a fiber doing a netbox call, cancel it, and read
`box.error.last()`/the error cdata while GC runs the cancel finalizer) under
`ASAN_OPTIONS=abort_on_error=1`. Control: the 1.10.3 fixed tree, which should
report no UAF.

## Vehicle note (LuaJIT-only)

The handle is a LuaJIT `cdata`, and LuaJIT does not target `capstone64`. For a
reference-Lua Capstone vehicle the `struct error` would be carried as a
**userdata** with a `__gc` metamethod — the same two-object coupling and the same
free-in-finalizer-then-deref bug, just PUC-Lua userdata in place of the cdata.

## PASS signature (for a future dedicated env)

On the vulnerable ASan tree: `AddressSanitizer: heap-use-after-free`, **READ** in
`error_ref` (`src/diag.c:100`) via `luaL_pusherror` (`src/lua/utils.c:913`) /
`lbox_error` (`utils.c:924`), freed by `exception_destroy` (`src/exception.cc:45`)
through `lbox_fiber_cancel` → `diag_clear` → `error_unref`, block allocated by
`BuildFiberIsCancelled` (`src/exception.cc:264`) via `luaL_testcancel`
(`src/lua/fiber.c:45`). Control on the 1.10.3 fix: no ASan report.
