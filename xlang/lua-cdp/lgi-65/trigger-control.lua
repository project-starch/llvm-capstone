-- lgi #65 CONTROL (safe access, same coupling, correct GC order).
-- Read iface.methods BEFORE any GC (while the backing GArray is still alive),
-- then detach the field so the C struct no longer points at the GArray before it
-- is collected. Clean even on the vuln tree.
local lgi = require "lgi"
local Gio = lgi.require "Gio"
local method = Gio.DBusMethodInfo()
local iface = Gio.DBusInterfaceInfo()
iface.methods = { method }
print(iface.methods)      -- read while the backing GArray is still alive (no GC yet)
iface.methods = nil       -- detach: struct field no longer references the GArray
collectgarbage()          -- guard frees the now-orphaned GArray; field is NULL
print("DONE")
