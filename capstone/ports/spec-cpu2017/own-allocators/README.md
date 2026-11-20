# SPEC CPU2017: the benchmarks' own allocators as Sublet lifetimes

The kit's Full build protects what `malloc` hands out. 500.perlbench_r and 502.gcc_r also carve objects from blocks
they keep for themselves, and a freed object of that kind is reused without `malloc` seeing it. The scripts here edit
the benchmark sources so that each such object is a child lifetime of its block (CDERIVE), ended when the object is
freed (CREVOKE). Each edit is under a define and is asserted to match exactly once; without the define the build is
the Full build.

Measured 2026-11-18 on silicon (bitstream P1). The record, with pre-registrations, transcripts' hashes and limits, is
`capstone/docs/ref/nested-allocator-workloads.md`; the numbers below are copied from it.

## 500.perlbench_r (Perl 5.22.1): every scalar head

| file | sha256 | what it does |
|---|---|---|
| `500.perlbench_r/port-sv-heads.py` | `f55e5096…` | the 5.36.3 patch `ports/perl/sublet/patches/5.36.3/0001` ported to 5.22.1, under `PERL_CAPSTONE_SUBLET` |
| `500.perlbench_r/fix-lookup.py` | `f5ccf26e…` | replaces the patch's arena lookup, run on `sv.c` after the port |

- Build: run both scripts on a copy of the Full build's sources, then compile every object with the Full build's own
  line plus `-DPERL_CAPSTONE_SUBLET -I capstone/runtime/include`, and link with the Full build's link line.
- Control: compiled without the define, 69 of 69 objects are byte-identical to the Full build's.
- Beyond the 5.36.3 patch, 5.22.1 needs six sites changed that read or write a head after it is freed (`S_rv2gv`,
  `pp_bless`, `sv_resetpvn`, `S_glob_assign_glob`, `pad_free`, `pp_aassign`). The script's comments give each one.
  No CREVOKE is skipped and no scalar is exempt.
- **The lookup repair is required for a usable cost.** The 0001 patch finds a head's arena by walking the list of
  arenas, one load through each. On Perl 5.36 `suns.pl` that walk made the protected interpreter cost +208 %; with the
  arenas in an address-sorted array searched by bisection it costs +16.05 %. For 5.36.3 the same repair is
  `ports/perl/sublet/pending/5.36.3/0002-sv-arena-lookup-by-index.patch`, applied after 0001; no build in this tree
  applies it yet (`ports/perl/sublet/pending/README.md`).
- Result (protected image stripped `f5449ace…`): SPEC's test workload, `test.pl`, **+5.84 % in-context cycles,
  +5.41 % run time**, with `test.out` byte-equal to SPEC's reference. One run per arm, on a lifetime table of 2^20
  records: one test script needs more than 262,144.

## 502.gcc_r (gcc 4.5): alloc-pool elements

| file | sha256 | what it does |
|---|---|---|
| `502.gcc_r/port-alloc-pool.py` | `ac09f51c…` | `alloc-pool.c`: each pool element a child of its block, under `GCC_CAPSTONE_POOL` |
| `502.gcc_r/fix-stale.py` | `108b7a3b…` | `df-scan.c`: one use of a pool element after `pool_free`, fixed in both arms |

- Build: run both scripts on a copy of the Full build's sources, recompile the two changed files with the Full
  build's own lines, `alloc-pool.c` with `-DGCC_CAPSTONE_POOL -I capstone/runtime/include` added for the protected
  arm, and link with the Full build's link line.
- Control: without the define `alloc-pool.o` is byte-identical to the Full build's (the scripts restore upstream's
  line numbers with `#line`, since `gcc_assert` prints `__LINE__`). The two arms differ in that one object.
- The use after free is gcc's own: `df_insn_rescan` frees an insn record and installs references collected for it
  into the record allocated next. Upstream works because the pool's free list hands the same element back.
- Result (images `e9b0f80c…` stock with the fix, `54291a9f…` protected): train input 1, **+5.26 % in-context cycles,
  +5.09 % run time**, output hash equal to SPEC's reference.
- **Not protected:** the garbage-collected heap (`ggc-page.c`; SPEC's build never collects, so every object would
  stay live) and obstacks.
