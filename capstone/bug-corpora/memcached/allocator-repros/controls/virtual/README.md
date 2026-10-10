# Controls for the virtual-Capstone arms of memcached/allocator-repros

Built by `controls/virtual/shared/build-cases.sh capstone-application` with the same options as the
corpus, and run in the same boot as `buggy 90` (plus ` 1` on the nested arm).

| control | `virtual-malloc` (`MCP_STOCK_MALLOC`) | `virtual-nested-pools` (`MCP_SUBLET`, mode 1) |
|---|---|---|
| `90_ctl_chunk_freed_to_slab` | COMPLETE: `slabs_free` keeps the chunk inside its page, one virtual-mallocng object | FAULT at `read_probe`: the ledger revokes the chunk at `slabs_free` |

With the virtual heap's own controls (bounds-malloc, uaf-malloc), which fault on both arms, it tells
a column of completions apart from a dead instrument, and shows the nested arm revokes at a chunk free.
