# riscv-base plain_reuse.c counterparts

Diffs against perf/fair/musl-src-reuse's src/malloc/mallocng/plain_reuse.c (riscv-base's mapping cache, 512 / 64 MiB):
- plain-reuse-round.diff: __plain_map rounds the length to pages, so a chunk free() released with maplen*4096 can be found;
- plain-reuse-evict.diff: arm C's rule (malloc-mapping.c mr_make_room): a mapping that fits the byte bound by itself evicts
  from the front of the table until it fits;
- plain-reuse-round-evict.diff: both; the counterpart of arm C.
plain-map-wrap.c: the measurement logger (-Wl,--wrap=__plain_map), never part of an image.
