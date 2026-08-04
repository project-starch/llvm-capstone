-- tarantool #7657 — cross-domain use-after-free of a `struct merge_source`.
--
-- A LuaJIT cdata (CTID_STRUCT_MERGE_SOURCE_REF) wraps a refcounted C
-- `struct merge_source`. `source:pairs()` returns the luafun triple
-- (gen=lbox_merge_source_gen, param=nil, state=<source cdata>); each gen call
-- pushes back a FRESH cdata wrapping the SAME merge_source* WITHOUT bumping the
-- refcount, yet that cdata's GC finalizer (lbox_merge_source_gc) DOES call
-- merge_source_unref. So a GC that runs mid-iteration collects an intermediate
-- cdata, drops the refcount to 0, destroys the merge_source, and the next
-- lbox_merge_source_gen derefs the freed struct's (now-nulled) vtab -> SIGSEGV.
--
-- The issue's verbatim script (`source:pairs():length()`) hits this only when
-- the GC happens to fire during iteration ("gc is called" at iter 5312 in the
-- report). On an optimized release build the freed slab usually stays intact,
-- so we make the coupling DETERMINISTIC by forcing a full collection from
-- inside fetch_chunk — the exact mid-iteration window the bug describes. Same
-- two objects, same free site, same use site; only the GC timing is pinned.
--
-- Two distinct allocations: the Lua cdata handle and the separately-allocated
-- C `struct merge_source` -> unambiguous CDP. LuaJIT-only (cdata); a
-- reference-Lua Capstone vehicle would carry the merge_source as userdata.

local merger = require('merger')
local iter = 1
local function fetch_chunk(context, state)
    if iter >= 7000 then return end
    collectgarbage('collect')      -- force GC while the iterator's cdata is live
    local data = {}
    for i = 1, 2 do
        data[#data + 1] = {iter, tostring(iter)}
        iter = iter + 1
    end
    return {}, data
end

local function mr_call()
    local source = merger.new_table_source(fetch_chunk, {}, {})
    return source:pairs():length()   -- UAF deref of the freed merge_source
end

print(mr_call())                     -- prints 7000 only on a fixed build
os.exit()
