-- cffi-lua #57 — cross-domain use-after-free of a libffi callback closure.
--
-- The Lua `callback` cdata wraps a SEPARATE heap closure_data (make_cdata_func,
-- ffi.cc:271). callback:free() destroys the closure_data (ffi.cc:127) but does
-- NOT null the cdata's pointer, so callback:set() then reads the freed block
-- (ffilib.cc:281). Two distinct allocations -> unambiguous CDP.
--
-- Verbatim from upstream issue #57.

cffi = require("cffi")
callback = cffi.cast("void (*)()", function() end)
callback:free()
callback:set(function() end)   -- UAF read on the vulnerable tree (ffilib.cc:281)
callback:free()                -- (double-free on the same block)
print("NO-CRASH")              -- reached only on a fixed/guarded build
