-- lgi #122 CONTROL (safe access, same coupling, no finaliser resurrection).
-- Read the extents while the region is alive and keep `r` referenced across GC;
-- only drop it once nothing will touch it again. Clean even on the vuln tree.
local cairo = require("lgi").cairo
local r = cairo.Region()
print(r:get_extents().x)      -- read while the region is alive
collectgarbage("collect")     -- r still referenced -> boxed region not freed
print(r:get_extents().x)      -- still valid
r = nil
collectgarbage("collect")     -- region freed cleanly; nobody uses it afterwards
print("DONE")
