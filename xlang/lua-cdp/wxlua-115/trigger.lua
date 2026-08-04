-- wxLua #115 — cross-domain double-delete of a wxMenu submenu.
--
-- Two distinct allocations:
--   1. the Lua userdata for `submenu` (wxLua GC handle, gc-tracked), and
--   2. a SEPARATE 256-byte C++ wxMenu object (new wxMenu, constructor binding
--      wxcore_menutool.cpp:1243).
--
-- menu:AppendSubMenu(submenu, ...) transfers OWNERSHIP of the C++ submenu to the
-- parent menu (wxWidgets stores it in a wxMenuItem and deletes it in ~wxMenu).
-- On the vulnerable tree wxLua does NOT un-track the submenu userdata, so it
-- still believes it owns the C++ object.
--
-- The two collectgarbage() calls make the double-delete deterministic and route
-- the SECOND delete through the submenu userdata's __gc (the documented stack):
--   GC #1: `menu` is unreachable -> wxLua __gc deletes the parent wxMenu ->
--          ~wxMenu -> the wxMenuItem destructor `delete`s the C++ submenu.
--   GC #2: `submenu` userdata is unreachable -> wxLua __gc calls
--          wxLua_wxMenu_delete_function -> `delete` on the ALREADY-FREED submenu
--          -> heap-use-after-free / double free (wxcore_menutool.cpp:1387).
--
-- On the fixed tree (ded8e0a) AppendSubMenu calls wxluaO_undeletegcobject on the
-- submenu, so GC #2 is a no-op and we reach the print below.
-- `wx` is the global binding table injected by the wxLua standalone interpreter.

local menu = wx.wxMenu()
local submenu = wx.wxMenu()
menu:AppendSubMenu(submenu, "Sub")   -- ownership of the C++ submenu -> parent

menu = nil
collectgarbage("collect")            -- GC #1: ~wxMenu frees the C++ submenu

submenu = nil
collectgarbage("collect")            -- GC #2: __gc double-deletes the submenu

print("NO-CRASH: reached end")       -- reached only on a fixed/guarded build
