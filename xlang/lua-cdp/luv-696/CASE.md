# luv #696 — scandir req userdata ⟷ `uv_fs_t` cross-thread use-after-free

**One line.** A Lua userdata wraps a libuv `uv_fs_t` scandir request; it can be
GC'd (its `__gc` runs `uv_fs_req_cleanup`) while a libuv threadpool worker is
still iterating the request — a cross-thread UAF.

## Identity

| | |
|---|---|
| Library | [`luv`](https://github.com/luvit/luv) (luvit) |
| Language pair | **C ⟷ Lua** (reference Lua 5.4; also LuaJIT/Neovim) |
| Upstream | https://github.com/luvit/luv/pull/696 |
| Vulnerable commit | **`0e4a895`** ("fix memory leak in fs.scandir sync mode" — made the req GC-able, introducing the UAF) |
| Fix commit | **`3e39f98`** ("Fix garbage collection of scandir reqs") |
| Native dep | libuv (verified 1.51.0) |

## The two coupled objects (why unambiguous CDP)

1. **Lua-GC handle:** the userdata wrapping the scandir `uv_fs_t` request.
2. **Separate native resource:** the `uv_fs_t` request the worker thread reads via
   `uv_fs_scandir_next`.

**Direction:** GC-frees. The req userdata is collected in the main thread
(`luv_fs_gc` → `uv_fs_req_cleanup`) while the threadpool worker still derefs the
`uv_fs_t` → SEGV.

## Reproduction status

**REPRODUCED (2026-08-03), with control.**

- Env: PUC Lua 5.4.7 (shared toolchain), libuv 1.51.0 (shared), gcc 15 ASan,
  cmake module build (`-DWITH_SHARED_LIBUV=ON -DBUILD_MODULE=ON`).
- Repro: the committed regression test `tests/test-fs.lua` "fs.scandir given to
  new_work" (added in `ec6ecf5`), reduced to `trigger.lua` with an explicit
  `req=nil; collectgarbage()` to free the req while the worker runs.
- Vuln `0e4a895`: **ASan SEGV** in `luv_fs_gc` (`src/fs.c:40`) →
  `uv_fs_req_cleanup`.
- Control, fixed `3e39f98`: clean, `DONE`.
- `./build.sh && ./run.sh` → PASS.

## PASS signature

Vuln: ASan SEGV/heap-use-after-free with a `luv_fs_gc` frame. Fixed: `DONE`, no
ASan. Both required.
