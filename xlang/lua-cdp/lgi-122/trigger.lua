-- lgi #122 VULN reproducer (verbatim from the issue).
-- A cairo.Region record is finalised first (its boxed cairo_region_t freed by
-- lgi's record __gc -> g_boxed_free -> cairo_region_destroy); a SECOND finaliser
-- then calls r:get_extents() on the freed region.
local cairo = require("lgi").cairo
do
  local r
  local function f() print(r:get_extents().x) end
  if _VERSION >= "Lua 5.2" then setmetatable({}, { __gc = f })
  else getmetatable(newproxy(true)).__gc = f end
  r = cairo.Region()
  f()
end
collectgarbage("collect")
print("DONE")
