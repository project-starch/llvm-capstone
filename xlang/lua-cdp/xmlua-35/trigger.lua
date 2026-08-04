local ffi = require("ffi")
local libxml2 = require("xmlua.libxml2")     -- loads cdefs + the buggy xpath wrapper
local xml2 = ffi.load("xml2")
ffi.cdef[[ void* xmlReadMemory(const char*,int,const char*,const char*,int); ]]
local s = "<root><a>x</a><b>y</b><c/></root>"
local doc = ffi.gc(xml2.xmlReadMemory(s, #s, "n.xml", nil, 0), xml2.xmlFreeDoc)
local ctx = libxml2.xmlXPathNewContext(doc)
local obj = libxml2.xmlXPathEvalExpression("//a | //b", ctx)   -- ffi.gc(obj, xmlXPathFreeObject): NO doc tie (the bug)
xml2.xmlXPathFreeContext(ffi.gc(ctx, nil))
doc = nil; collectgarbage("collect"); collectgarbage("collect")   -- xmlFreeDoc frees the nodes
obj = nil; collectgarbage("collect"); collectgarbage("collect")   -- xmlXPathFreeObject -> xmlXPathFreeNodeSet reads freed node->type
print("DONE")
