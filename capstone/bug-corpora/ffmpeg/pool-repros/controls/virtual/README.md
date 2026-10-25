# Controls for the virtual-Capstone arms of ffmpeg/pool-repros

Built by `controls/virtual/shared/build-cases.sh capstone-application` with the same options as the
corpus, and run in the same boot as `buggy 90`.

| control | `virtual-malloc` (stock pools) | `virtual-nested-pools` (`FFPOOL_SUBLET`, patch 0003) |
|---|---|---|
| `90_ctl_buffer_returned_to_pool` | COMPLETE: the pool keeps the entry, one virtual-mallocng object | FAULT at `read_probe`: the release revokes the user's buffer |

With the virtual heap's own controls (bounds-malloc, uaf-malloc), which fault on both arms, it tells
a column of completions apart from a dead instrument, and shows the nested arm revokes at a return.
