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
- **An O(1) arena lookup** (`500.perlbench_r/fix-lookup-pages.py`, run after `fix-lookup.py`): arenas keep upstream's
  layout; a table indexed by the 4 KiB page of an address holds two verified arena pointers per slot (an arena's heads
  span at most two pages, and a page meets at most two arenas' heads), misses fall back to the binary search. On
  test.pl it takes the port's added instructions from +4.303 % to +2.894 % of perlbench's code, at +0.86 MB of peak
  memory (the table). `fix-lookup2.py` to `fix-lookup4.py` are the aligned-arena line it replaces (+2.859 %, but
  +8.56 MB: mallocng's aligned blocks). `make-heads-fixture.py` writes a fixture that embeds the interpreter from a build's objects: a freed
  head faults, live heads of the same and other arenas read, a second interpreter's heads read and the first one's
  fault; with a 1-slot table too, and a build without the verify-on-hit fails it.
- **SV bodies** (`500.perlbench_r/port-sv-bodies.py`, `3933328d…`, after the heads port and both lookup fixes, under
  `PERL_CAPSTONE_SUBLET_BODIES` as well): every arena body is a child lifetime of a 32 MiB segment, issued when it is
  popped and revoked when it is freed. Hash entries keep upstream's arenas. Without the define `sv.c` compiles to the
  heads build's `sv.o`. `make-bodies-fixture.py` (`1fc7613b…`) writes its fixture: freed and upgraded bodies fault,
  live bodies read, and a build with `PERL_CAPSTONE_SUBLET_BODIES_NOREVOKE` reads in modes 0-3. On test.pl it adds
  +0.733 % of perlbench's code to the heads port (+3.649 % against stock together), at the same peak memory. Cycles
  are not read yet.
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

## 502.gcc_r: the collaborator's one-operand port of all three allocators, refined

The port of gcc's pools, GC heap and obstacks with one-operand CREVOKE and CRENEW is the collaborator's
(`origin/native-kernel-new-qemu:capstone/ports/spec-cpu2017/sublet/502.gcc_r`). Two files here refine its obstack part
without changing what it protects (every object keeps a lifetime of its own, a free still ends every object it cuts
off):

| file | what it does |
|---|---|
| `502.gcc_r/obstack-variant.py DIR abc` | on a tree that the port's `apply.py` produced: (a) the issue fast path inline in `obstack_finish`; (b) a dropped chunk's references forgotten in O(1) by the stack depth kept in each chunk (with a pad, so that a chunk's `contents` stays 16-byte aligned); (c) the reference stack in segments of 1,024 that are never copied |
| `502.gcc_r/obstack-fixture.c` | six modes, built against an obstack: the controls must read, every freed object must fault (cause 25), the survivors must read; stock obstack reads the freed objects of modes 1, 4 and 5 |

Measured on QEMU (the environment of revision 71ce199d8fea, train input 1, exact counts of gcc's own code): the
obstack port costs +0.654 % as the collaborator wrote it and +0.348 % refined; all three allocators +0.784 % and
+0.478 %. Record: `capstone/docs/ref/nested-allocator-workloads.md`.

### gcc's own free lists, and a second fix in gcc

On top of the three allocators (and only there: these objects must already have a lifetime of their own), each object
that gcc puts on one of its private free lists is renewed (CRENEW), so the client's references die and the list keeps
the renewed one. A list that upstream moves in one step is walked to renew and relink each object.

| file | lists |
|---|---|
| `502.gcc_r/bitmap-port.py` | bitmap.c: elements (one, or a chain in bitmap_elt_clear_from) and bitmap heads |
| `502.gcc_r/lists-port.py` | lists.c: EXPR_LIST and INSN_LIST nodes |
| `502.gcc_r/tfl-port.py` | c-decl.c bindings and scopes, emit-rtl.c sequence stacks, gimple.c sequences, tree-phinodes.c phi nodes (below 4 KiB only) |
| `502.gcc_r/bitmap-fixture.c` | links against an arm's bitmap.o and obstack.o: freed elements must fault, the bitmap must still work |
| `502.gcc_r/fix-phi.py` | the fix below, for every arm |

**A second use after free in gcc 4.5, fixed in every arm.** `remove_phi_node` (tree-phinodes.c) calls
`release_phi_node (phi)`, which puts the node on gcc's free list, and then `release_ssa_name (gimple_phi_result (phi))`,
which reads the node through the same reference. Stock gcc survives because nothing reuses the node in between; with
the phi free list ported, train01 faults there (cause 25). `fix-phi.py` swaps the two calls. `release_ssa_name` does
not touch the phi node, so the output is unchanged; on train01 the fix moves gcc's own code by -543 instructions.

**Limit: SSA names are not ported.** tree-ssanames.c keeps released SSA names on a free list, and gcc reads released
names through old references by design: passes test `SSA_NAME_IN_FREE_LIST` on stale names. Renewing them would fault
on correct code.

Measured as above: the whole port, with these lists, costs +0.750 % of gcc's code on train01 against the arm with
both fixes; bitmap +0.236 %, lists.c +0.034 %, the five others +0.001 %.

### Objects of 4 KiB and more; growing obstack objects

`502.gcc_r/large-port.py` replaces the port's 4 KiB limit with `sublet_exact` (in sublet-port.h: whether CDERIVE holds
the bounds exactly, following QEMU 9e69171f1c's encoder), at the GC heap's allocation and release, at the phi list and
at the obstack's issue. On train01 the GC heap's 1,136 objects of 4 KiB or more are all exact and now have lifetimes of
their own (`large-count.py` is the counting build that showed it); the obstacks finish none. Cost +0.33 M instructions;
the whole port +0.752 % of gcc's code.

**Limit: a growing obstack object has no lifetime of its own until obstack_finish.** gcc reads a growing object's base
in pretty-print.c (three places) and reorg.c (one); none runs on train01.
