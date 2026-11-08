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

**What it decides.** If the cache makes Plain faster, the reported ratios of every configuration that keeps released
mappings in the context must use these Plain readings. That is the unified stack (CAPSTONE_HEAP_MAPPING_REUSE) **and
c3f and c3g**: their SDK (perf/arena/sdkdir-ac12) is built with CAPSTONE_HEAP_ARENA_CACHE=ON (wt-arena heap.c), the
same optimization in the older heap: a released group mapping is kept, its lifetime child revoked, and a later
same-size request takes it back zeroed; 512 entries.

Correction (7 Nov, before any board call of these runs): the first version of this paragraph, pushed in c349058d7fd8,
said c3f and c3g do not use such a cache. They do (CAPSTONE_HEAP_ARENA_CACHE); the claim was written from the
switch's name without reading the SDK's build flags.

## Result, gcc (8 Nov 01:34): Plain with the mapping cache is 6.7 % FASTER; R2 FAILS (the band was too narrow); gcc c3f restated at 1.296×

Board 1, C.5a (the list moved to perf/b1q, results/2026-11-07-plainnc), rc 0, specdiff 1/1 PASS.

| Plain gcc train | time | minor faults | user / sys | maxrss |
|---|---|---|---|---|
| plain-rv64 (board 1, 1 Nov; board 3 fill 3,510.0 s) | 3,515.5 s | 89,080 (board 3: 88,982) | 2,864 / 266 s | 98,868 kB |
| plain-rv64r, the cache | **3,280.9 s** (0.933×) | 36,192 | 2,909 / 60.5 s | 146,688 kB |

- **R2 FAILS:** −6.7 %, outside the −4 %…+1 % band. The cache removes 59 % of Plain's minor faults and 77 % of its
  system time; it holds about 48 MB more.
- **The baseline that is fair to c3f and c3g is this one:** both keep released group mappings in the context
  (CAPSTONE_HEAP_ARENA_CACHE, the same bounds). Against it **gcc train c3f is 1.296× Plain** (4,253.3 / 3,280.9),
  not 1.212×. The reviewer's point holds, and it costs 8.4 points on gcc.
- perlbench plain-rv64r is running on board 1 now; its result restates perlbench the same way.

## Addendum 1 (8 Nov 02:3x, before their board calls): mcf and x264 with the cache, board 2

After gcc's −6.7 %, every Plain denominator is restated against the cache. mcf (mcf_r_base.plain-rv64r 4857348eb693)
and x264 (x264_r_base.plain-rv64r 2d8170fd12b5, imagevalidate 52b5c9539774) are built as gcc's; against their
plain-rv64 tree builds (the stage binaries, hash-equal) `llvm-nm` differs only in the cache's symbols. Stages: the
Plain mcf stage of results/2026-11-05-mcfx (perf/mx, 9,094.9 s on board 2) and the 3 Nov x264 Plain stage, binaries
replaced. Board 2 (perf/b3/gt-run-ga2.sh 2, C.5a), after perlbench c3g-thp there: mcf, then x264.

| # | Prediction |
|---|---|
| R3 | mcf plain-rv64r within −3 %…+1 % of 9,094.9 s (mcf allocates its arrays once; little churn) |
| R4 | x264 plain-rv64r within −8 %…+1 % of 50,320.6 s |

## Result, perlbench (8 Nov 08:0x): Plain with the cache is 6.0 % faster; R1 FAILS (the band was too narrow)

- Board 1, C.5a: 499.4 + 21,383.2 s = **21,882.6 s** (0.9401× Plain's 23,277.1 s, the same board and bitstream);
  rc 0, specdiff 2/2 PASS, test.out 317 ok / 0 FAILED.
- **R1 FAILS:** −6.0 %, outside −3 %…+1 %. As on gcc (−6.7 %), the cache is worth about 6-7 % to Plain.
- **Restated against it:** perlbench on the unified stack without the exit digest 30,120.3 s = **1.376×**; c3g
  33,667.7 s = 1.539×. gcc train c3f 1.296× (above). These are the fair readings for every configuration that keeps
  mappings in the context (ARENA_CACHE or MAPPING_REUSE).

## Addendum 2 (8 Nov 08:0x, before their board calls): mcf and x264 move to board 3

Board 3 freed first. Addendum 1's mcf and x264 runs move there (perf/b3q), unchanged otherwise; mcf first, x264
last, with gcc's split exitprof rerun between them (results/2026-11-08-exitprof Addendum 2). mcf's reference
9,094.9 s is board 2's; board 3 read Plain mcf 0.6 % slower (9,150.5 s on the THP kernel), so R3 is read against both.

## Addendum 3 (8 Nov 08:1x): correction of Addendum 2 - mcf and x264 stay on board 2, where they are running

Addendum 2 was wrong about the state: board 2's perlbench profiling run had already ended (07:03), and perf/b2q's
driver had started mcf there at 07:06. Stopping that driver at 08:0x did not stop its chain: mcf, then x264, continue
on board 2 under chain-board.sh. Nothing moves to board 3. The chain's end will not be followed by the driver's
flash back to bitstream 3; the next driver on board 2 flashes what it needs.
