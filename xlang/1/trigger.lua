-- rlua #19 trigger: resurrect a userdata from its own __gc finalizer.
--
-- `userdata` is a Rust value handed to Lua as full userdata. Dropping the last
-- Lua reference makes it collectable; the metatable's __gc finalizer runs
-- rlua's `destructor<T>`, which drops the Rust value (freeing the heap buffer
-- its String field owns). But __gc receives the object being finalized, so the
-- finalizer can store it somewhere reachable again -- here into the global
-- `hatch`. Lua's collector honours the resurrection and keeps the userdata
-- block alive, while the Rust value inside it has already been dropped.
--
-- Calling a method on the resurrected handle then reads through the freed
-- buffer. Deterministic: collectgarbage("collect") forces the free at a fixed
-- point, so there is no timing or allocation-layout dependence.

local tbl = setmetatable({
    userdata = userdata
}, { __gc = function(self)
    -- Resurrect: publish the userdata to a global that outlives this finalizer.
    hatch = self.userdata
end })

print("collecting...")
tbl = nil
userdata = nil          -- drop every strong reference so the pair is collectable
collectgarbage("collect")

print("hatch = ", hatch)
hatch:access()          -- use-after-free: reads the String buffer freed by Drop
