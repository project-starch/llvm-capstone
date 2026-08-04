local ffi = require("ffi")
local libxml2 = require("xmlua.libxml2")
local xml2 = ffi.load("xml2")
ffi.cdef[[ void* xmlReadMemory(const char*,int,const char*,const char*,int); ]]
local s = "<root><a>x</a><b>y</b><c/></root>"
local doc = ffi.gc(xml2.xmlReadMemory(s, #s, "n.xml", nil, 0), xml2.xmlFreeDoc)
local ctx = libxml2.xmlXPathNewContext(doc)
local obj = libxml2.xmlXPathEvalExpression("//a | //b", ctx)
xml2.xmlXPathFreeContext(ffi.gc(ctx, nil))
obj = nil; collectgarbage("collect"); collectgarbage("collect")   -- free xpath object FIRST (nodes still valid)
doc = nil; collectgarbage("collect"); collectgarbage("collect")   -- then free the document
print("DONE")
