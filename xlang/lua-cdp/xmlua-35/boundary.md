# Boundary annotation — xmlua #35

### The object that crosses the boundary
An `xmlXPathObject` (a nodeSet holding raw `xmlNodePtr` into a document tree),
wrapped as a LuaJIT cdata with `ffi.gc(object, xmlXPathFreeObject)`.

### Owner vs. borrower
- **The `xmlDoc` (its own `ffi.gc` → `xmlFreeDoc`) owns the nodes.**
- **The xpath object borrows** raw node pointers, with **no reference** to the
  document (the bug: `xmlua/libxml2.lua:650-656`).

### Free site
`doc = nil; collectgarbage()` → `xmlFreeDoc` frees the whole node tree.

### Stale-use site (one crossing later)
`obj = nil; collectgarbage()` → `xmlXPathFreeObject` → `xmlXPathFreeNodeSet`
(`xpath.c`) reads each `node->type` (to free namespace nodes) → invalid read of
freed node memory.

### The lifetime rule that is violated
An xpath result that holds document node pointers must keep the document alive
for its own lifetime. The suggested fix zeroes `nodesetval.nodeNr` before freeing
so `xmlXPathFreeNodeSet` does not iterate the freed nodes.

### Capability note (revoke-on-free)
Revoke-on-free revokes the node-tree capabilities at `xmlFreeDoc`; the xpath
object's borrowed node pointers are revoked too, so `xmlXPathFreeNodeSet` faults
at the read instead of touching freed nodes.
