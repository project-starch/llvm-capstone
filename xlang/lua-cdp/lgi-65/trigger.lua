-- lgi #65 VULN reproducer (verbatim from the issue).
-- `iface.methods = { method }` marshals a Lua table into the GArray that backs
-- DBusInterfaceInfo.methods; lgi's field-set guard (guard_gc -> g_array_unref)
-- frees that GArray while the C struct still points at it. Reading iface.methods
-- re-derefs the freed GArray in marshal_2lua_array.
local lgi = require "lgi"
local Gio = lgi.require "Gio"
local method = Gio.DBusMethodInfo()
local iface = Gio.DBusInterfaceInfo()
iface.methods = { method }
collectgarbage()
print(iface.methods)
print("DONE")
