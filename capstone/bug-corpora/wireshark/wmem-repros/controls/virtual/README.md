# Controls for the virtual-Capstone arms of wmem-repros

Both arms run these in the same boot as the corpus, built with the same options:

| control | `virtual-malloc` (wmem on virtual mallocng, `WM_LIBC_SYSTEM`) | `virtual-nested-pools` (`WM_SUBLET`, mode 1) |
|---|---|---|
| `90_ctl_jumbo_freed_by_reset` (a link to `../sublet-malloc/`) | FAULT: the reset hands the jumbo to `g_free`, which mallocng retires | FAULT: the jumbo's region is revoked at that `g_free` |
| `91_ctl_chunk_freed_in_block` | COMPLETE: an individual BLOCK free never reaches the system allocator | FAULT: the chunk port revokes the chunk at its free |

90 proves column 2 can fault on wmem's own frees, so its completions are misses, not a dead heap.
91 is the per-chunk lifetime that column 2 cannot see, and column 3 must.
