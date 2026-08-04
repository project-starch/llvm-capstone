# wxLua #115 — Lua userdata ⟷ C++ `wxMenu` submenu double-delete

**One line.** `menu:AppendSubMenu(submenu, ...)` hands ownership of the C++
`wxMenu` submenu to the parent menu (wxWidgets deletes it in `~wxMenu`), but
wxLua keeps tracking the submenu userdata as gc-owned — so the userdata's `__gc`
`delete`s the already-freed C++ object a second time.

## Identity

| | |
|---|---|
| Library | [`wxLua`](https://github.com/pkulchenko/wxlua) (bindings for wxWidgets) |
| Language pair | **C++ ⟷ Lua** (PUC Lua 5.4). A GUI-toolkit binding — the coupled object is a real third-party C++ resource (a `wxMenu`), not FFI plumbing. |
| Upstream | https://github.com/pkulchenko/wxlua/issues/115 |
| Vulnerable commit | **`b5ffaccac0bbb2587952a932e5f80abc7c083a35`** (parent of the fix) |
| Fix commit | **`ded8e0a3e6b19bbb752c68282a9e37e9b88b7582`** — "Fixed double freeing of wxMenu items added as a sub-menu (closes #115)" (adds `%ungc` to `AppendSubMenu`) |
| Native dep | wxWidgets 3.2 (verified 3.2.9, wxGTK3) |

## The two coupled objects (why unambiguous CDP)

1. **Lua-GC handle:** the `submenu` userdata from `wx.wxMenu()`, registered in
   wxLua's gc-object list (so its `__gc` will `delete` the wrapped C++ object).
2. **Separate native resource:** a 256-byte C++ `wxMenu` (`new wxMenu(...)` in
   `wxLua_wxMenu_constructor2`, `wxcore_menutool.cpp:1243`).

Two distinct allocations. **Not** the borrowed-pointer-into-one-object shape.

**Direction:** native-frees. On `AppendSubMenu`, wxWidgets takes ownership and
`~wxMenu` (parent) frees the submenu (crossing 1). The Lua userdata's `__gc` then
`delete`s the same C++ object again (crossing 2, `wxcore_menutool.cpp:1387`).

## Reproduction status

**REPRODUCED (2026-08-03), with control.**

- Env: wxWidgets 3.2.9 (wxGTK3), PUC Lua 5.4.7 (toolchain, shared), gcc 15 ASan,
  headless under `xvfb-run`. Built via the wxLua CMake, bindings restricted to
  `core;base` (wxMenu lives in wxcore); the standalone `wxLua` interpreter.
- Vulnerable `b5ffacc`: ASan **heap-use-after-free** (double-delete), use site
  `wxLua_wxMenu_delete_function` (`wxcore_menutool.cpp:1387`) via
  `wxluaO_deletegcobject` (`wxllua.cpp:434`) ← `wxlua_wxLuaBindClass__gc`
  (`wxlbind.cpp:113`) ← `GCTM`; freed by `wxMenu::~wxMenu` (parent's
  `wxMenuItem` destructor); alloc'd at `wxcore_menutool.cpp:1243`.
- Control, fixed `ded8e0a`: **no ASan report** — reaches `NO-CRASH: reached end`.
- Full trace + control in `evidence.txt`.

## Detection note

README-planned method was **gdb**; we use **ASan**, which pinpoints the exact
double-delete (use + free + alloc frames) more cleanly than catching the abort
in gdb. "ASan or gdb" is the accepted latitude for this case.

## PASS signature

`run.sh` passes iff, on the **vulnerable** commit, ASan reports
`heap-use-after-free` with `wxLua_wxMenu_delete_function` at
`wxcore_menutool.cpp:1387` as the use site AND the wxLua GC path
(`wxluaO_deletegcobject` / `wxlua_wxLuaBindClass__gc` / `GCTM`) on the stack —
AND the **control** run on the fixed commit reaches `NO-CRASH` with no ASan
report. Either half missing = FAIL.
