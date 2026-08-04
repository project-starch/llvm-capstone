# xlang Lua CDP corpus — host↔GC cross-domain-pointer UAFs

Ground-truth reproductions of **15 unambiguous cross-domain-pointer (CDP)
use-after-frees** at a native-host (C/C++) ↔ Lua boundary. Each case couples a
**Lua-GC handle** (userdata / cdata) to a **separately allocated native C/C++
resource**; one is freed while the other still references it → UAF, one boundary
crossing apart.

This is the same *kind* of artifact as `xlang/repro/` (stock-toolchain,
decoupled from our compiler/QEMU/board), so the reproduction half is
collaborator-friendly.

**Scope discipline (why these 15 and not others).** Every case is a *two-object*
coupling. The "native code caches a raw pointer *into* a Lua-GC string/table"
shape (e.g. Vim `luaV_print`, a cached `lua_tolstring`) is **excluded** — it is
a borrowed pointer into a single managed object, structurally identical to the
mruby VM-stack rows we do not count as cross-domain. If a candidate cannot name
*two distinct allocations*, it does not belong here.

## Per-case layout

Each `xlang/lua-cdp/<slug>/` holds:

| File | Purpose |
|---|---|
| `CASE.md` | Identity, pinned vulnerable version, the two coupled objects, direction, CDP-classification line, deps, **reproduction status**, PASS signature |
| `boundary.md` | Boundary annotation: object crossing, owner/borrower, free-site + stale-use-site (`file:line`), lifetime rule, capability note |
| `build.sh` | Clean fetch of the binding @ pinned commit + C deps, built with the sanitizer |
| `run.sh` | Runs the trigger under ASan/valgrind; **asserts the PASS signature, exits non-zero otherwise** |
| `trigger.lua` (+ `harness.c`/`main.cpp` if a host is needed) | The minimal reproducer |
| `evidence.txt` | Captured trace — upstream's quoted until we reproduce, then replaced with ours |

## The 15 cases

Status: `PENDING` = handoff written, not yet reproduced here; `REPRODUCED` =
`run.sh` passes on our machine; `BLOCKED` = needs infra we do not have.

| # | slug | Library | Lua handle ⟷ native resource | Dir | Detect | Status |
|---|---|---|---|---|---|---|
| 1 | `lua-openssl-141` | lua-openssl | userdata ⟷ `EVP_CIPHER_CTX` | native-frees | ASan | **REPRODUCED** (2026-08-03, w/ control) |
| 2 | `ldbus-20` | ldbus | iter userdata ⟷ `DBusMessage` | GC-frees | differential | **REPRODUCED** (2026-08-03, w/ control) |
| 3 | `xmlua-35` | xmlua | xpath cdata ⟷ `xmlDoc` nodes | GC-order | valgrind | **REPRODUCED** (2026-08-03, w/ control) |
| 4 | `cffi-lua-57` | cffi-lua | cdata ⟷ libffi `closure_data` | native-frees | ASan | **REPRODUCED** (2026-08-03, w/ control) |
| 5 | `lua-sdl2-75` | lua-SDL2 | userdata ⟷ `SDL_Window` | GC-frees | ASan/valgrind | **REPRODUCED** (2026-08-03, w/ control) |
| 6 | `wireshark-16807` | Wireshark | `Tvb` userdata ⟷ C `tvbuff` | native-frees | valgrind (apt tshark) | **REPRODUCED** (2026-08-03, w/ control) |
| 7 | `luv-696` | luv | userdata ⟷ `uv_fs_t` | GC-frees | ASan SEGV | **REPRODUCED** (2026-08-03, w/ control) |
| 8 | `sol2-1373` | sol2 | wrapper ⟷ interior ptr into C++ struct | GC-frees | ASan | **REPRODUCED** (2026-08-03, w/ control) |
| 9 | `luabridge-319` | LuaBridge | wrapper ⟷ C++ object | GC-frees | sentinel | **REPRODUCED** (2026-08-03, w/ control) |
| 10 | `wxlua-115` | wxLua | userdata ⟷ C++ `wxMenu` submenu | native-frees | ASan | **REPRODUCED** (2026-08-03, w/ control) |
| 11 | `lgi-122` | lgi | userdata ⟷ `cairo_region_t` | GC-frees | valgrind | **REPRODUCED** (2026-08-03, w/ control) |
| 12 | `lgi-65` | lgi | guard userdata ⟷ `GArray` in C struct | GC-frees | valgrind | **REPRODUCED** (2026-08-03, w/ control) |
| 13 | `luaossl-124` | luaossl | store userdata ⟷ `X509_STORE` (co-owned by `SSL_CTX`) | double-free | ASan | **REPRODUCED** (2026-08-04, w/ control) |
| 14 | `tarantool-7657` | Tarantool (LJ) | cdata ⟷ `merge_source` | GC-frees | SIGSEGV (docker) | **REPRODUCED** (2026-08-03, w/ control) |
| 15 | `sol2-1080` | sol2 | `sol.Foo*` userdata ⟷ native C++ `Foo` | native-frees | ASan | **REPRODUCED** (2026-08-04, w/ control) |

**All 15 REPRODUCED** — each with our own captured ASan/valgrind/gdb trace and a
passing vuln/control `run.sh` differential.

**Archived (not part of the 15 — upstream-verified, not buildable in this
sandbox).** Kept on disk as documented references (`CASE.md` + `boundary.md` +
`evidence.txt`, source-confirmed), swapped out of the canonical 15 in favour of
`luaossl-124` and `sol2-1080`:
- `tarantool-1955/` — ASan-only UAF (read of a freed `struct error`, no
  vtable-call fault), no minimal repro; needs a from-source Tarantool 1.10 +
  LuaJIT ASan build.
- `corona-858/` — no prebuilt Solar2D SDK; needs the full Corona engine
  (`librtt` + Box2D). Boundary source-confirmed against the real tree.

**Reproduction order** (cheapest deps / most deterministic first): 4 `cffi-lua`,
1 `lua-openssl`, 2 `ldbus`, 3 `xmlua`, 11/12 `lgi`, 8/9/15 `sol2`/`LuaBridge`,
13 `luaossl`, 5 `lua-SDL2`, 10 `wxLua`, 6 `wireshark`, 14 `tarantool` (docker).

**Vehicle note.** #3 (`xmlua`) and #14 (`tarantool-7657`) are LuaJIT `cdata` —
LuaJIT does not target `capstone64`, so for a Capstone build these reproduce in
reference-Lua **userdata** form (same coupling). The rest are
reference-Lua-compatible.
