# realloc publishes an inexact child when it keeps the object's offset (twin, kit-q34, 2026-10-29)

Images from the board SDKs `perf/sdk-T1` (heap_lifetimes false) and `perf/sdk-T3` (true), `capstone-cc -O2`.
`pair N0 N1 K`: K rounds of `p = malloc(N0); realloc(p, N1)`. A round whose base is not a multiple of the
granule of `repr_round(N1)` prints PRED-TRAP before the realloc. The prediction was written before the run.

| run | route | rounds before the fault | fault |
|---|---|---|---|
| pair-T1 32720 32768 | in place (realloc.c:25-31) | round 0, a%64=32, PRED-TRAP | cause=2, the `unimp` of issue_child's exactness check |
| pair-T3 32720 32768 | in place | round 0, a%64=32, PRED-TRAP | cause=29 at the inlined `cderive` |
| pair-T1 261888 262144 | mremap no-op (sizeclass 63) | 13 rounds a%512=0 in place, ok; round 13 a%512=256, PRED-TRAP | cause=2 |
| pair-T3 261888 262144 | mremap no-op | the same 13, then round 13 | cause=29 |
| pair-T1 32704 32768 (control) | cannot grow in place | 16 rounds a%64=32, every one moved | none, rc 0 |

`sweep KLO KHI R` (k = 12..20) faulted in both arms right after `CAND k=15 n0=32720 n1=32768 a%g=32 g=64`, the first
k=15 size a model of mallocng's size classes allows to grow in place. k=12 and k=13 print no candidates: their granules
(8, 16) cannot misalign a 16-byte-aligned base.

**Mechanism.** enframe steps an object's base down only to the granule of its ORIGINAL size. realloc's in-place
branch, and the mremap branch that keeps `base = p - start`, return the same offset for a larger size, and heap.c's
publish() derives `[p, p + repr_round(n))`, whose granule can exceed that alignment. T1's SHRINK check traps on the
rounded base; T3's CDERIVE refuses the inexact range.

**Not established:** which of the two routes the SPEC traps (gcc train 200.c, scilab.c; perlbench test japh/abigail,
re/pat_rt_report) take. Their fault sites are the same two instructions in realloc.
