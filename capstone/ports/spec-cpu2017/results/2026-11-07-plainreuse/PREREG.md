# Plain with the mapping cache (the reviewer's baseline): pre-registration

Written and pushed before the board calls (7 Nov, 17:0x). PR #306's reviewer accepted in-domain mapping reuse but
asked that the baseline get the same optimization. Plain links musl 1.2.5 mallocng statically, the allocator the
virtual profile's heap is built on, so the counterpart is exact: the same cache below mallocng.

**The change.** perf/plainreuse/musl-1.2.5 is perf/T0b's musl source with `src/malloc/mallocng/plain_reuse.c` and
three call sites redirected:
- malloc.c's two mmap of group and large mappings → `__plain_map`;
- malloc.c's error munmap and free.c's munmap of a released mapping → `__plain_unmap`.
Policy as malloc-mapping.c's CAPSTONE_HEAP_MAPPING_REUSE: exact-length match, 512 entries, 64 MiB, memset to zero on
reuse, and at mmap failure the cache is unmapped and the mmap retried once. mremap (realloc) is unchanged, the
native path that suits Plain.

**Images** (label plain-rv64r, the same tree-plain and config, only the label and libc.a differ):
- gcc: cpugcc_r_base.plain-rv64r 4d9ddde9aee9b35f against the stage's 6385edb2ab2b20fe; `llvm-nm` differs by exactly
  the cache's symbols (__plain_map/unmap, three counters, four statics);
- perlbench: `--strip-all` of the tree build, 3bbd77760d40d34a; the 3 Nov stage binary bf1813c35f7fd845 is
  `--strip-all` of the tree's plain-rv64 build (reproduced byte for byte).
- Stages: copies of the Plain stages with only that binary replaced.
- QEMU: gcc train plain-rv64r rc 0, specdiff 1/1 PASS.

**Runs:** board 1, after the Plain perlbench run there (results/2026-11-07-fill) ends with DONE: perlbench test
plain-rv64r, then gcc train plain-rv64r (perf/b3/gt-run-ga2.sh 1, C.5a).

| # | Prediction |
|---|---|
| R0 | rc 0, specdiff PASS (perlbench 2/2 with 0 FAILED, gcc 1/1), control passes |
| R1 | Plain perlbench reuse vs the Plain run before it on board 1: change within −3 %…+1 % |
| R2 | Plain gcc reuse vs Plain gcc (3,515.5 s board 1; board 3's same-board reading from the fill run): change within −4 %…+1 %. Plain gcc's 266 s of system time and 89,080 minor faults bound the saving from above |

**What it decides.** If the cache makes Plain faster, the reported ratios of any configuration that uses
CAPSTONE_HEAP_MAPPING_REUSE must use these Plain readings. c3f and c3g do not use it; the unified stack does.
