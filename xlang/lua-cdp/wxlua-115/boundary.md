# Boundary annotation — wxLua #115

### The object that crosses the boundary

A raw `wxMenu*` to a heap C++ `wxMenu` (the submenu), wrapped by the Lua userdata
that `wx.wxMenu()` returns. The userdata is the Lua-visible handle; the wrapped
`wxMenu*` is what crosses. On `menu:AppendSubMenu(submenu, ...)` that same
pointer is also handed to the parent menu (stored inside a `wxMenuItem`).

### Owner vs. borrower

- **Before AppendSubMenu:** Lua (the GC) owns the submenu — wxLua tracks its
  userdata in the gc-object list and its `__gc` will `delete` the `wxMenu`.
- **After AppendSubMenu:** wxWidgets owns the submenu — the parent's
  `wxMenuItem` holds `m_subMenu` and `~wxMenu` (parent) `delete`s it.
- The bug: ownership transferred to the parent, but wxLua never un-tracked the
  submenu userdata, so **both** sides believe they own the one C++ object.

### Free site

`collectgarbage()` collecting the parent `menu` → wxLua `__gc`
(`wxlua_wxLuaBindClass__gc`, `wxlbind.cpp:113`) → `wxluaO_deletegcobject`
(`wxllua.cpp:434`) → `wxLua_wxMenu_delete_function` (`wxcore_menutool.cpp:1387`,
`delete o`) → `wxMenu::~wxMenu` → `wxMenuItemBase::~wxMenuItemBase` deletes the
child submenu (`delete m_subMenu`). The 256-byte C++ submenu is now freed.

### Stale-use site (one crossing later)

`collectgarbage()` collecting the `submenu` userdata → wxLua `__gc`
(`wxlbind.cpp:113`) → `wxluaO_deletegcobject` (`wxllua.cpp:434`) →
`wxLua_wxMenu_delete_function` (`wxcore_menutool.cpp:1387`) `delete`s the
**already-freed** `wxMenu` again → ASan heap-use-after-free (READ of size 8 —
the vtable load in `delete`). A plain double free of the same 256-byte block.

### The lifetime rule that is violated

When ownership of a native resource is transferred out of the managed handle,
the handle must stop treating itself as the owner — otherwise its finalizer frees
memory a second owner already freed. The fix (`ded8e0a`) marks the `AppendSubMenu`
parameter `%ungc`, emitting
`if (wxluaO_isgcobject(L, submenu)) wxluaO_undeletegcobject(L, submenu);`
(`wxcore_menutool.cpp:197`) so the submenu leaves wxLua's gc-object list at the
moment ownership crosses; its `__gc` then no longer deletes it.

### Capability note (revoke-on-free)

On a revoke-on-free allocator, the first delete (the parent's `~wxMenu` at the
free site) **revokes** the capability to the 256-byte `wxMenu`. The submenu
userdata's `__gc` then holds a revoked capability, so the second `delete` at
`wxcore_menutool.cpp:1387` faults at the contract point — exactly the delivered
fault the capability model promises, in place of the ASan-detected double-delete.
