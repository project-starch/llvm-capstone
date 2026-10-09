# memcached allocator-repros: Sublet ONLY as the system allocator (`sublet-malloc`), 2026-10-09

**Question.** Column 2 of the per-bug table for memcached's slabs.c and cache.c: Sublet as the
system allocator, the two allocators stock. A stock slab page is one `malloc`, so a chunk pointer
carries the whole page's bound; a cache.c object is its own `malloc`; nothing is freed until a page
or object is given back.

**Build.** The allocators port's ledger gained a mode 2 (`src/shared/leases.c`): the page is never
cut -- every chunk is the page's alias at the chunk's address, checked at carve time against the
page's own bounds (code 509 otherwise) -- objects keep their own region, and the free-list
transitions revoke nothing; a page or object is revoked only at its discard. The guest loader
accepts mode 2 and `runners/capstone-domain/run-defects.py --modes sublet-malloc` judges it
against each case's `sublet-malloc` arm.

**Result, as pre-registered (f7adf03a1009): 1 of 9 caught, and that one by a bound.** Cases 0-4 (the
temporal ones, including the refcount races 1, 2 and 4) complete: a chunk goes back to its class's
free list and an object to cache.c's, and neither reaches `free()`. 5, 6 and 7 complete: the
crossings stay inside one page. 8 faults with cause 5 at the first byte past the 16384-byte rbuf
object, bounds exactly that object, in the defective scan itself -- the shape the `spatial` and
`sublet` arms recorded; the runner prints FAIL for it only because its oracle wants the probe's pc.

**Control, same images.** Mode 1 on cases 0, 2 and 4 faults with cause 24 at the labelled read
probe, so these images do revoke when the ledger asks; mode 2's completions are the configuration,
not a broken build.

**The page-bound check can fire.** Mode 2's carve-time check (a chunk's alias must carry exactly its
page's bounds, else code 509) was negative-tested: an image built with the expected end moved 16
bytes in refused case 5 with report status 509 and completed=0. With the real expectation every
case ran through it, so each chunk alias in this run carried its whole page's bound -- the stock
configuration, not per-chunk bounds without revocation.
