# mem5design -- mem5-design

Upstream fix `mem5-design`. Collected in round R1 (NVD and upstream history).

The reduction is argued in the case source's own header, reproduced here so the
claim travels with the case rather than living only in the build tree.

```
row25 / sqlite-mem5-design -- MEMSYS5 stores freelist metadata INSIDE freed
 * blocks (src/mem5.c: memsys5Link/Unlink write Mem5Link{next,prev} into the freed
 * block; CTRL_FREE is out-of-band in aCtrl[]). A dangling pointer into a block
 * freed back to the pool then reads/overwrites allocator metadata. Design-inherent,
 * never patched (paper: PoisonCap). Present in 3.22.0 (mem5.c:143 MEM5LINK).
 *
 * CONTROL arm: allocate two adjacent blocks, keep a dangling pointer to the first,
 * free it, force a neighbour free so memsys5 links/coalesces (writing in-band
 * metadata), then READ through the dangling pointer. The bytes read are now
 * allocator freelist indices, not the caller's data -- the aliasing the paper
 * names. On unprotected Capstone (whole pool is one allocation, invisible to ASan)
 * this READ SUCCEEDS and the domain RETURNS. Sublet would revoke the freed block.
```
