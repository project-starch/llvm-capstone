# Revocation-node cache: how often would a node cache hit?

Every load or store through a capability checks that the capability's revocation node is still
valid, and so does every capability loaded from memory. In the RTL this is a node read
(`capstone-ariane/core/anvil_build/capstone_rev_node.anvil` reads the node through `mem_ch`; no
node cache sits in front of it). This experiment measures the node-access stream of real workloads
and replays it against node caches of every common shape. It answers how many hits and misses a
cache of a given size and associativity would have. It does not answer what a miss costs.

## Instrument

- **capstone-qemu, branch `perf/revnode-trace`.**
  - `CAPSTONE_REVNODE_TRACE=<file>` appends one 8-byte record for every node access the emulator
    makes. Format: `target/riscv/cap_rev_tree.h`. A run of identical records is written once,
    followed by a repeat count.
  - Every access is attributed to a site:
    - lifetime check of a load/store (`ldst`), lifetime check of a loaded capability (`ldc`);
    - `mrev`, `split`, `revoke`, `delin`: the reads and writes the RTL makes, in its order;
    - allocation and free-list traffic;
    - the emulated supervisor's own checks and collector (`supervisor`, `gc`).
  - Without the variable the emulator is unchanged except for one not-taken branch per access.
- **`cachesim.c`**: reads the trace once, front to back, from a file or a pipe.
  - Simulates LRU caches: fully associative at 8 to 65536 entries, and direct-mapped, 4- and 8-way
    at 16 to 4096 entries. An entry holds 1 node, or `--nodes-per-line N` consecutive nodes.
  - All fully associative sizes come from one LRU list with a marker at each size boundary,
    because LRU caches nest. A repeat of the previous line is a hit everywhere and is counted
    without touching any cache.
  - This keeps three simulators ahead of QEMU. A per-size simulator held QEMU to 30 % of a core.
    `results/20261005-qemu` was simulated by the earlier per-size version (b77546d0), which also
    had 2-way. On the SQLite trace the two versions agree on all 29 configurations they share.
  - Write-allocate. A tree reset empties the caches.
  - Reports hits and misses per site, compulsory misses (first touch of a line), and revoke walk
    lengths.
- **`selftest.py`**: checks `cachesim` before it is used. `run.sh` runs it first and stops with
  exit 75 if it fails.
  - It compares `cachesim` against an independent Python LRU model on random traces with runs,
    resets and excluded sites, in both trace formats.
  - It also checks hand-derived cases: cyclic sweeps, a direct-mapped conflict, a reset, the
    compulsory count, and revoke walks of 0, 1 and 3 nodes.
  - It also checks traces past 65536 distinct lines, so every boundary crossing and the
    eviction path are exercised.
  - Negative controls, run once:
    - an MRU victim instead of LRU fails 62 checks;
    - ignoring repeat records fails 9;
    - not moving a line across a size boundary fails 56;
    - a truncated trace and an empty one both exit non-zero.
- **`run.sh <workload>`**: boots the VM with the trace going to a FIFO and runs three simulators on
  it through `tee`: 1 node/line, 4 nodes/line, and without the supervisor/gc sites. Nothing is
  stored. A mruby trace runs to hundreds of GB, and a plain trace passed 30 GB in two minutes.
- **`summarize.py`**: prints the tables below from the JSON reports.
- **`prepare-bench.py`**: scales mruby's own `benchmark/` programs to emulator-sized inputs. The
  code is unchanged; only the input sizes shrink. It records each program's native output, and the
  STUDY-ORACLE lines of SQLite `speedtest1 --size 1`.
- **`run-mix.py`** (`run.sh mix-<par|seq>-<label>`): runs the programs in `$MIX` in one VM.
  - `par`: all at once, each its own Linux process and domain.
  - `seq`: the same list one after another in the same boot. This is the control that differs
    only in the interleaving.
  - Every output is compared with native byte for byte. A changed or empty output reads FAIL
    (checked once by hand).
- **`compare-mix.py`**: lifetime-check miss rate of a `par` report against its `seq` report.
- **Alias accounting** (capstone-qemu 0bcc3dc7):
  - The trace also records every capability copy entering or leaving a 16-byte memory granule,
    from a shadow of the tag map, and the 32 GPRs and the PCC just before each revoke.
  - The trace now records drop's node write as well, which the RTL makes and QEMU does not. The
    earlier results lack those writes.
- **`aliasstat.c`**: reads the same stream.
  - It keeps its own copy of the emulator's revocation tree (the list of nodes with depths,
    replayed from mrev/split/revoke).
  - It reports: copies per node; copies at each revoke (of the revoked capability, and the
    copies the revoke makes stale); children, depth and tree size.
  - It exits 3 on its own errors: a copy count going negative, or a revoke whose recorded walk
    differs from its tree copy.
  - `selftest_alias.py` checks it against a hand-built trace. Negative controls: no depth
    increment at mrev fails 6 checks, counting repeated copies once fails 4, and not counting
    children fails 1.
- **`summarize_alias.py`**: the per-program table below. Percentiles above 8 are bucket bounds.
- **`clovsim.c`**: Capstone's lifetime checks against Clover's capability index, on one trace.
  - The trace (capstone-qemu 2f16a090) also carries each copy's 16-byte granule, and an END
    record. Every reader rejects a stream without END: a reader that died once cut `tee`, and
    the others then wrote complete-looking reports.
  - Two caches of 64-byte lines, one per design, fully associative LRU, 16 to 65536 lines:
    - Capstone: every traced node access (16-byte node records).
    - Clover: the controller steps of `ideas/clover/detailed/sections/tracking-metadata.tex`:
      `reserve_alias_entry`, `register`, `unregister`, a 2-level radix walk plus per-frame
      sidecars, and revoke clearing every alias of every invalidated node. Clover keeps its own
      index and revokes eagerly; the emulator is lazy.
  - The Capstone side reproduces `cachesim --nodes-per-line 4` exactly (all 7 sizes, so_lists).
  - `selftest_clover.py` counts every metadata access of a hand-built trace. Negative controls:
    no same-node shortcut fails 3 checks, and no directory walk on unregister fails 2.
- **`summarize_clover.py`**: the comparison below, per 1000 lifetime checks.
- **`bucketsim.c`**: the index proposed below ("Ein Index mit einem Zugriff pro Aenderung"),
  in variants (inline capacity, slot cache size, node record size), on the same stream. It keeps
  its own view of which slot holds which node under Clover's eager revocation, since the
  emulator's is lazy; its error count must be 0.
  - `selftest_bucket.py` counts every access of three hand-built traces, including the CAM
    cases at revoke. Negative controls: no same-node shortcut fails 3 checks, a tag clear of a
    slot that holds another node fails 1, and a slot cache that absorbs nothing fails 3.
- **`summarize_bucket.py`**: the comparison of all designs below.
- **`ptrbench.py`**: builds the pointer benchmarks of the llvm-test-suite (Olden, Ptrdist,
  MallocBench: 17 C programs) twice, with the host compiler as the reference and with the
  application SDK's `capstone-cc` on the **sublet heap** (one revocation node per allocation), and
  writes a manifest `programs.json` for `run-mix.py --programs`. The sources are unchanged except
  one adaptation (voronoi, below); `run.sh ptr-<par|seq>-<label>` runs them.

## Results (`results/20261005-qemu/`)

Platform:
- Emulator: dev's capstone-qemu pin 674cdab03c with the two trace commits (`PROVENANCE.txt`).
- Kernel, firmware, rootfs and module: the SQLite/mruby lanes' kit, whose emulator is an ancestor
  of that pin.
- `CAPSTONE_REV_NODES=16777216`.

Every workload's own check passed:
- SQLite: `work`, `again` and `speedtest1` are byte-identical to native.
- mruby: `smoke.rb` reached `SMOKE_DONE` in all three heap arms.

A second boot of `fx-sublet` from `run.sh` reproduced the allocations, revokes and walks exactly,
and the check counts to within 0.05 %.

| workload | node accesses | lifetime checks | distinct nodes | revokes | nodes invalidated | longest walk |
|---|---:|---:|---:|---:|---:|---:|
| boot-only (control) | 139,552 | 139,235 | 41 | 0 | 0 | 0 |
| SQLite 3.22 -O2, level0 heap | 402,007,868 | 401,950,956 | 72 | 24 | 36 | 4 |
| mruby smoke, level0 | 5,141,341,669 | 5,141,047,151 | 59 | 4 | 7 | 4 |
| mruby smoke, sublet | 249,791,223 | 248,303,563 | 123,243 | 82,187 | 123,188 | 4 |
| mruby smoke, sublet-gc | 316,584,161 | 312,701,059 | 323,985 | 198,827 | 323,930 | 84,110 |

Lifetime-check miss rate, as a share of `ldst` + `ldc` checks. Fully associative LRU; the
"(4/line)" rows hold four consecutive nodes per entry.

| workload | 16 | 64 | 256 | 1024 | 4096 | 16384 | 65536 |
|---|---:|---:|---:|---:|---:|---:|---:|
| SQLite, level0 | 0.002% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| mruby, level0 | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| mruby, sublet | 0.180% | 0.067% | 0.056% | 0.050% | 0.046% | 0.041% | 0.023% |
| mruby, sublet (4/line) | 0.101% | 0.028% | 0.024% | 0.021% | 0.018% | 0.011% | 0.000% |
| mruby, sublet-gc | 0.834% | 0.541% | 0.533% | 0.515% | 0.479% | 0.429% | 0.229% |
| mruby, sublet-gc (4/line) | 0.402% | 0.223% | 0.213% | 0.199% | 0.173% | 0.114% | 0.011% |

What the numbers say:

- **Heaps whose pointers share one node need almost no cache.**
  - level0 derives every pointer from the arena capability, so 5.1 billion checks touch 59 nodes.
  - 16 entries leave no measurable miss.
- **Per-object nodes put a floor under the miss rate.**
  - sublet-gc: 1.69 M check misses at 64 entries and still 1.50 M at 4096. The curve is flat from
    64 to 4096 entries and falls clearly only at 32 K and 64 K entries.
  - So the misses are long-reuse accesses, a working set larger than any plausible node cache. They
    are not conflicts.
  - No lifetime check is a compulsory miss in any run: every node is first touched by its own
    allocation.
- **Associativity matters only below about 64 entries.**
  - sublet at 64 entries, all accesses: direct-mapped 1.70 %, 2-way 0.80 %, 4-way 0.45 %, 8-way
    0.19 %, fully associative 0.18 %.
  - From 1024 entries on, every shape is within 0.03 points of fully associative.
- **Packing nodes into lines helps the revoking heaps.**
  - Four nodes per line cuts the sublet and sublet-gc check miss rates by a factor of 1.8 to 2.8
    between 16 and 4096 entries.
  - The reason is that consecutive allocations get consecutive node ids.
- **Revoke walks are short, with one exception.**
  - Mean 1.50 (sublet) and 1.63 (sublet-gc) nodes.
  - One sublet-gc walk invalidated 84,110 nodes. Which revoke that was is not attributed (see open
    questions).
- **The supervisor and collector sites change little.** Without them, sublet-gc at 64 entries
  misses 2.25 M of 315.9 M accesses instead of 2.57 M of 316.6 M.

## Several processes at once (`results/20261005-qemu-mix/`)

Workloads:
- Programs: mruby's benchmark programs, run in the sublet-gc heap arm:
  - `ao_render`, a ray tracer at 16x16;
  - `so_lists`, array shuffling;
  - `fib(30)`;
  - `so_mandelbrot` at 150;
  - `lc_fizzbuzz` over 1..30;
  - `mandel_term`.
  SQLite 3.22 `speedtest1 --size 1` (level0 heap) runs alongside them.
- Mixes: N = 2, 4 and 5 programs, taking the first N of: ao_render, speedtest1, so_lists, fib,
  so_mandelbrot.
- Each mix ran twice, in two boots: all at once (`par`), and one after another (`seq`).
- All 30 program runs matched their native output, and `mix-seq-all7` (all seven, one after
  another) did too.

**Five processes is the most the platform runs at once.**
- Each sublet-gc process takes a 160 MiB heap (the image's descriptor) plus its domain.
- With the default 1024 MiB CMA, a mix of three mruby processes and SQLite started only one of the
  mruby processes. The other two failed with "cannot allocate application heap".
- The kernel reserves CMA below 4 GiB physical. `--cma-mib 4096` boots with "cma: Failed to
  reserve 4096 MiB", and every domain then fails, so `run.sh` now refuses such a boot.
- At 1920 MiB, 4 mruby processes and SQLite run together. Of 8 (7 mruby), 3 are refused.

Lifetime-check miss rate, fully associative LRU, `seq` → `par`, with the misses interleaving
adds:

| entries | N=2 | N=4 | N=5 |
|---:|---|---|---|
| 32 | 1.086 → 1.086 % | 0.657 → 0.662 % (+273 k) | 0.570 → 0.575 % (+362 k) |
| 64 | 0.745 → 0.745 % | 0.451 → 0.455 % (+262 k) | 0.390 → 0.397 % (+448 k) |
| 256 | 0.275 → 0.275 % | 0.166 → 0.171 % (+249 k) | 0.145 → 0.151 % (+408 k) |
| 1024 | 0.153 → 0.153 % | 0.092 → 0.093 % (+25 k) | 0.080 → 0.081 % (+74 k) |
| 4096 | 0.101 → 0.101 % | 0.061 → 0.062 % (+12 k) | 0.053 → 0.054 % (+86 k) |

Lifetime checks per run: 3.34 G (N=2), 5.52 G (N=4), 6.39-6.40 G (N=5).

- **Interleaving costs little.** Fully associative, it adds 1 to 4 % to the miss rate at 32-256
  entries and 0 to 2 % at 1024-4096. That is +0.4 M misses on 6.4 G checks at N=5.
  - The N=2 mix adds nothing measurable: SQLite's level0 heap uses a handful of nodes.
  - Reading of it: a quantum is long against a cache refill. A switch can cost at most one
    cache-full of misses, and between switches a process makes far more checks than that. The
    number of switches is not in the trace.
- **Small set-associative caches feel it more.** A 4-way 16-entry cache goes from 1.74 % to 2.20 %
  (+26 %, +29 M misses) at N=5, and 64 entries 4-way +12 %. From 256 entries 4-way the
  difference is within ±3 %.
- **Node ids are reused across processes.** A destroyed domain's nodes return to the pool:
  - `seq` touches 2,989,001 distinct ids at every N, the high-water of its largest program;
  - `par` touches 2.08-2.87 M.
  A reused id is rewritten by its allocation, which the simulator counts like any write.

The real benchmarks also raise the single-process floor that `smoke.rb` showed. All seven
programs one after another miss 0.67 % of lifetime checks at 64 entries and 0.33 % at 4096 (11.4 G
node accesses, 10.9 M distinct ids).

## Aliases and tree shapes (`results/20261005-qemu-alias/`)

Each program ran alone, one boot each, in the sublet-gc heap arm; SQLite `speedtest1` ran on
level0. All outputs matched native. Every report has 0 negative counts and 0 walk mismatches. In
every run the collector freed 0 nodes with a copy still recorded, which it must not since it
clears stale tags first. That is the check that the hooks see every place a tag is cleared.

| | ao_render | lc_fizzbuzz | so_lists | fib | so_mandelbrot | mandel_term | speedtest1 (level0) |
|---|---:|---:|---:|---:|---:|---:|---:|
| nodes allocated | 2,989,001 | 10,880,348 | 6,554 | 4,264 | 21,584 | 4,452 | 59 |
| most copies of one node at once: p50 | 4 | 3 | 1 | 1 | 7 | 1 | 2 |
| p99 | <=16 | <=16 | <=32 | <=32 | <=16 | <=32 | <=16384 |
| max | 6,995 | 34,829 | 6,923 | 6,913 | 6,925 | 6,936 | 8,378 |
| copies made stale by one revoke: mean | 3.56 | 3.73 | 5.84 | 7.51 | 4.87 | 7.55 | (4 revokes) |
| p99 | 6 | <=16 | <=16 | <=16 | 6 | <=16 | |
| max | 7,095 | 98,874 | 6,752 | 6,748 | 6,742 | 6,773 | |
| nodes one revoke invalidates: mean | 1.18 | 1.18 | 2.10 | 2.61 | 1.39 | 2.68 | 1.75 |
| live node depth: mean / max | 9.4 / 20 | 10.0 / 20 | 10.8 / 20 | 11.0 / 20 | 6.8 / 20 | 10.1 / 20 | 0.1 / 1 |
| children of a live node: p90 / max | 2 / 1,024 | 2 / 1,024 | 2 / 1,024 | 2 / 1,024 | 1 / 1,024 | 2 / 1,024 | 0 / 4 |
| largest tree (nodes) | 26,579 | 232,101 | 4,630 | 4,094 | 9,769 | 4,250 | 49 |

- **Few copies per node under sublet, many under level0.**
  - Under sublet-gc a node typically has 1 to 7 copies at its peak, and 99 % have at most 16 or 32.
  - A handful of nodes in every mruby run have about 7,000 copies, and one in lc_fizzbuzz has
    34,829. Which nodes those are is not attributed.
  - level0 is the opposite: 59 nodes, one of them with 8,378 copies, since every pointer derives
    from the arena.
- **A revoked capability exists once.** At its revoke it has one copy in memory and one in a
  register, never more than 3 and 2. The copies a revoke makes stale are those of the nodes it
  invalidates: 3.6 to 7.6 on average, rarely more than 16. One lc_fizzbuzz revoke made 98,874
  stale.
- **The trees are deep and narrow, with one wide level.**
  - Depth reaches exactly 20 in every mruby run, and live nodes sit at depth 7 to 11 on average.
  - A node has 1 or 2 direct children at the 90th percentile, and exactly 1,024 at most in every
    mruby run.
  - The mean of about 1.0 children says nothing: every node but a tree's root is someone's child.
  - Not verified: 1,024 matches mruby's 1,024-object heap page, which sublet-gc splits one slot
    at a time. Depth 20 would fit the buddy heap's levels.
- **What a reference count would have to sustain** (memory copies only; register copies, every
  `movc` and every capability loaded into a register, come on top):
  - 25 to 32 count changes per 100 lifetime checks.
  - 87 to 99 % of them are a capability stored over another capability, which touches two nodes:
    one count down, one up.
  - A count would need more than 15 bits: the largest is 34,829.

## Capstone's node checks against Clover's index (`results/20261005-qemu-clover/`)

Same programs, each alone, plus the five-program mix. All outputs match native.

Metadata misses per 1000 lifetime checks (64-byte lines, fully associative LRU), Capstone →
Clover:

| | 64 lines (4 KiB) | 1024 lines (64 KiB) | 16384 lines (1 MiB) |
|---|---|---|---|
| ao_render | 6.5 → 60.8 | 1.41 → 5.09 | 0.49 → 0.71 |
| lc_fizzbuzz | 7.0 → 118.6 | 4.96 → 25.2 | 3.35 → 18.0 |
| so_lists | 0.009 → 0.60 | 0.004 → 0.030 | 0.002 → 0.010 |
| fib | 0.005 → 1.56 | 0.001 → 0.019 | 0.001 → 0.006 |
| so_mandelbrot | 0.04 → 3.01 | 0.018 → 0.118 | 0.006 → 0.030 |
| mandel_term | 0.02 → 3.42 | 0.006 → 0.077 | 0.003 → 0.025 |
| speedtest1 (level0) | 0.000 → 12.3 | 0.000 → 0.107 | 0.000 → 0.023 |
| five at once | 3.3 → 31.5 | 0.75 → 2.77 | 0.27 → 0.43 |

- **Clover misses more at every size, in every workload.**
  - The mruby programs: 1.4 to 312 times as many misses at the sizes above. For the two with the
    most misses, ao_render and lc_fizzbuzz, it is 1.4 to 17 times.
  - SQLite on level0: Capstone has almost none, so the ratio is not meaningful.
  - Clover makes fewer accesses: 98 to 1,044 against Capstone's 1,000 to 1,031 per 1000 checks.
    They miss far more often, though: node records, other aliases' records and per-frame
    sidecars scattered over the heap, against Capstone's repeated checks of a few hot nodes.
- **The misses sit elsewhere.**
  - Capstone's are on capability loads and stores.
  - Clover's are on capability stores (register and unregister: 38 to 75 % of its misses at 64
    lines, about two thirds in most programs), on data stores over a capability (most of the
    rest), and on revokes (under 3 %).
  - Clover's protocol completes each update before the operation does, with no buffering.
- **Capability stores are 126 to 162 per 1000 checks.**
  - 58 to 95 % of them store over a copy of the same node, which leaves Clover's index unchanged.
    That is high (95 %) for the floating-point programs and lowest (58-64 %) for the
    allocation-heavy ao_render and lc_fizzbuzz.
  - No program stored a capability after its node had been revoked.
- **Revoke stops are mostly short, sometimes very long.**
  - Mean 15 to 30 metadata accesses per revoke in the mruby programs. SQLite's 4 revokes
    average 6,320, since level0 revokes whole arenas.
  - The longest: 39,401 (ao_render) and 487,816 (lc_fizzbuzz).
- **Clover's metadata is small.**
  - Peak alias records: 8,782 to 461,482, i.e. 0.1 to 7.0 MiB at 16 B.
  - Sidecars for 207 to 4,551 frames, 0.2 to 4.4 MiB.
- **This is the paper's baseline layout, not an optimised one.** Node records of 32 B, a
  separate alias pool, sidecars per frame. Co-locating a node's first alias with the node, or
  the sidecar with the tags, changes these numbers, and is the next thing to model.

## Ein Index mit einem Zugriff pro Aenderung

The question: a structure that finds every copy of a revoked capability, costs at most one
metadata access per store that changes it, and is hardware. This section derives one from the
problem and the measurements, places it against the literature, and measures it on the same
traces.

### The problem

State: the set S of (slot g, node n) with "g holds a tagged capability of n". A slot holds at
most one. Operations, with their measured frequency per 1000 capability loads/stores (the seven
programs above):

| operation | effect on S | per 1000 checks |
|---|---|---:|
| capability store, same node already in g | nothing | 94-139 |
| capability store into an untagged g | insert (g, n) | 1.6-21.6 |
| capability store over another node's copy | delete (g, m), insert (g, n) | 4.6-45.9 |
| data store over a capability | delete (g, m) | 1.6-21.6 |
| revoke of a run of nodes N | delete all (g, n), n in N; clear their tags | 0.001-2.1 |

So 58 to 95 % of capability stores change nothing, and of those that do, 70 % replace one
node's copy by another's. A revoke touches 1.2-2.7 nodes with 3.6-7.6 copies. Registers are
inspected at revoke (32 + PCC, fixed), as Clover does.

### Where the floor is

Any index that enumerates copies by node in time proportional to the copies must, for each
insert, write to a location determined by n. On a line-granular memory that is one line
access unless several changes share a line, which different nodes do not. Deferring the
write into a sequential journal (1/8 line per change) only postpones that access to the
moment the journal is sorted by node, and leaves stale entries in the journal until the node
is revoked. So the floor for an exact, eagerly maintained index is **one line read-modify-write
per index change**, and the only ways below it are to make fewer changes: detect the stores
that change nothing, and coalesce rewrites of the same slot before they reach memory.

Deletion is where Clover's baseline spends most: it needs to find the record for slot g, through
a radix walk and a per-frame sidecar. The observation that removes all of that: **the old
granule is in the line the store writes.** The line is in the L1 for the write (write-allocate,
and the tag bit lives in it), so reading the overwritten capability's node m costs a read of
the data array the store already addresses, and no memory access. The slot's own content is the reverse directory. Deletion is then one RMW of
node[m], the same cost as insertion.

### What this is in the literature

The structure is a GC write barrier's remembered set, keyed by allocation instead of by region:
- Lieberman and Hewitt's entry tables (1983) and Ungar's remembered sets (1984) record incoming
  references per region on each store; card marking (Sobalvarro 1988, Wilson and Moher 1989)
  records one bit per card at a store and scans dirty cards later; the sequential store buffer
  (Hosking, Moss and Stefanovic 1992) appends the slot address to a buffer, the cheapest
  barrier they measured. The slot cache below is a store buffer that coalesces by slot.
- G1's remembered sets (Detlefs et al. 2004) keep per region a sparse table, a fine bitmap
  or a coarse flag, depending on how many cards point in: the same skew handling as the
  inline bucket with an overflow table here.
- Coalescing reference counting (Levanoni and Petrank 2001) logs a slot once per epoch and
  reads old and new value at collection, so a slot rewritten many times costs one update.
  The slot cache is that in bounded hardware form.
- In temporal safety: DangNull (2015) keeps per-object lists of pointer locations with eager
  removal on overwrite (Clover's baseline, in software); DangSan (2017) appends locations to
  per-object logs, tolerates duplicates and validates at free (the journal alternative below);
  CHERIvoke (2019) and Cornucopia (2020, Reloaded 2024) sweep capability-holding memory with
  quarantine, and Cornucopia Reloaded's capability-dirty page bits are card marking maintained
  by the MMU at no store cost; Chromium's BackupRefPtr counts references per allocation on
  pointer assignment, in software, in production. Watchdog (2012) checks a per-allocation lock
  at every dereference, as Capstone does.
- Write-optimised indexes (LSM trees, B-epsilon trees) are the journal-and-compact answer with
  tombstones; two-choice and cuckoo hashing give constant-bounded lookups for the overflow.

### The design

**Node record = one 64-byte line.** Capstone's node state (prev, next, depth, valid, linear;
16 bytes) and K = 12 slot references of 32 bits each: the granule (physical address / 16) of
every copy of this node. No separate alias pool, no reverse directory.

**Store path.** The store unit reads the overwritten granule's node m from the line it writes
and emits (g, m, n) to the controller:
- m = n: nothing.
- otherwise, if g is in the slot cache: update the cached entry. Nothing reaches memory.
- otherwise: RMW node[m] (find g among 12 entries in parallel, remove), RMW node[n] (append).
A data store over a capability emits (g, m, none).

**Slot cache.** W fully associative entries (g, c, f): the slot's current node c and the node f
the memory index still lists it under. The W most recently capability-written slots live here
and nowhere else. On eviction an entry is written out: delete (f, g), insert (c, g), if they
differ. The data says why this matters: 70 % of index changes overwrite another node's copy,
and (below) nearly all of them hit a slot written moments before, the VM stack's top.

**Overflow.** A node whose 12 inline entries are full gets a table of 64-byte lines with 16
entries each, two-choice hashed by granule: insert into the first choice with room, else the
second, else double the table (rehash, a controller slow path). Lookups and deletions touch
one or two lines. Only nodes with more than 12 live copies have one: 1-3 % of nodes.

**Revoke.** Walk the run as Capstone does. Per invalidated node: RMW node[c]; clear the tag of
every listed slot that the slot cache does not hold under another node (a CAM check); clear
the tags of cached slots whose c is in the run; read and free the overflow lines; free the
record. Registers: the fixed scan. The node id is reusable at once: nothing references it.

**Per operation:**

| operation | metadata accesses |
|---|---:|
| capability store, same node | 0 |
| capability store, slot in the cache | 0 (deferred to its eviction: 1 or 2) |
| capability store into an untagged slot | 1 |
| capability store over another node's copy | 2 (one delete, one insert) |
| data store over a capability | 1 |
| revoke, per invalidated node | 1 + overflow lines; tag clears apart |

**Hardware.** The core side: the store unit reads the old granule's node field (30 bits) from
the line it writes, and pushes (g, m, n) to the controller's queue; stores never wait for
metadata. The controller: a W-entry CAM with LRU; a 12-wide comparator over the node line; a
two-choice hash over 16-entry lines; RMWs issued asynchronously from the queue; a revoke
drains the queue, consults the CAM, and walks the run. Table growth and node-id exhaustion are
slow paths that stall the core, like Clover's REVOKING state.

**Invariants.** (1) g holds a tagged capability of n iff g is cached with c = n, or g is not
cached and listed under n. (2) For a cached g, f is exactly what memory lists. (3) A revoke
removes every (g, n) of the run from both. (4) A node id is handed out again only when no
record and no cache entry names it. The Lean model of `ideas/clover` states (1) for its
logical index; this is a refinement of that index, not a change to it.

### Alternatives, and why the data rejects them

- **Journal with lazy validation** (DangSan-style, the sequential store buffer): 1/8 line per
  change, no reads on the store path. But entries are removed only at their node's revoke, and
  a long-lived node with copy churn collects them without bound: SQLite's level0 arena would
  gather 4.4 M entries (its 12.1 + 10.4 index-changing stores per 1000 checks over 196 M
  checks) for 59 nodes that are never revoked. Validating them at revoke is a data read per
  entry while the core stalls.
- **Lazy deletion inside the inline bucket:** stale entries fill the 12 slots, and reclaiming
  them costs 12 random data reads, against the eager delete's one node-line RMW.
- **Per-node Bloom filter of pages:** the same node-line RMW per store, but a revoke sweeps
  every candidate page (64 lines each) instead of touching 3.6-7.6 slots.
- **Sweeping with quarantine** (CHERIvoke, Cornucopia): no store cost, but node ids and
  storage are reusable only after a sweep, which Clover's requirements exclude.
- **Reference counting:** 25-32 count changes per 1000 checks, two nodes each, 16-bit
  counts, and it does not say where the copies are.

### Measured (`results/20261005-qemu-bucket/`)

Same programs, each alone. Every variant's error count is 0. The seven programs' outputs match native.

Misses per 1000 lifetime checks at 64 lines (4 KiB of metadata cache): Capstone, Clover's baseline,
and the proposed index without a slot cache (K12/W0), with 64 and 256 cached slots, and with 32-byte
node records holding 4 inline entries (K4/W64/32B).

| | Capstone | Clover baseline | K12/W0 | K12/W64 | K12/W256 | K4/W64/32B |
|---|---:|---:|---:|---:|---:|---:|
| ao_render | 6.534 | 60.8 | 18.1 | 6.347 | 4.960 | 5.353 |
| lc_fizzbuzz | 6.965 | 118.6 | 33.5 | 19.5 | 20.8 | 16.4 |
| so_lists | 0.009 | 0.609 | 0.078 | 0.057 | 0.045 | 0.052 |
| fib | 0.005 | 1.633 | 0.043 | 0.035 | 0.030 | 0.031 |
| so_mandelbrot | 0.040 | 2.988 | 0.963 | 0.514 | 0.201 | 0.538 |
| mandel_term | 0.020 | 3.423 | 2.044 | 0.231 | 0.118 | 0.236 |
| speedtest1 (level0) | 0.000 | 12.2 | 12.7 | 4.068 | 0.726 | 4.069 |

The same at 1024 lines (64 KiB) and 16384 lines (1 MiB), Capstone → K12/W64 → K4/W64/32B:

| | 1024 lines | 16384 lines |
|---|---|---|
| ao_render | 1.410 → 2.975 → 2.206 | 0.488 → 1.054 → 0.684 |
| lc_fizzbuzz | 4.956 → 12.122 → 9.593 | 3.351 → 9.038 → 6.330 |
| so_lists | 0.004 → 0.023 → 0.017 | 0.002 → 0.010 → 0.007 |
| fib | 0.001 → 0.014 → 0.011 | 0.001 → 0.005 → 0.004 |
| so_mandelbrot | 0.018 → 0.100 → 0.071 | 0.006 → 0.041 → 0.035 |
| mandel_term | 0.006 → 0.059 → 0.044 | 0.003 → 0.021 → 0.017 |
| speedtest1 (level0) | 0.000 → 0.024 → 0.024 | 0.000 → 0.011 → 0.011 |

Metadata accesses per 1000 checks, and what the index does:

| | Capstone | Clover baseline | K12/W0 | K12/W64 | changes/1000 | absorbed by 64 slots | to overflow (W64) | overflow nodes (W64) | revoke: mean / max accesses |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ao_render | 1012 | 908 | 177.7 | 20.49 | 57.6 | 83.2 % | 3.28 % | 1,003 | 3.2 / 10,197 |
| lc_fizzbuzz | 1032 | 1044 | 196.8 | 45.79 | 67.4 | 75.6 % | 5.23 % | 68,342 | 3.2 / 106,218 |
| so_lists | 1000 | 636 | 136.8 | 0.17 | 39.7 | 99.8 % | 0.07 % | 108 | 4.5 / 2,166 |
| fib | 1000 | 439 | 96.3 | 1.07 | 27.5 | 97.5 % | 0.81 % | 81 | 5.4 / 1,972 |
| so_mandelbrot | 1000 | 116 | 25.1 | 1.89 | 7.3 | 87.8 % | 4.57 % | 2,125 | 3.8 / 5,772 |
| mandel_term | 1000 | 100 | 20.7 | 2.09 | 6.3 | 77.4 % | 6.68 % | 79 | 5.5 / 2,125 |
| speedtest1 (level0) | 1000 | 359 | 87.1 | 13.98 | 22.5 | 71.2 % | 15.54 % | 2 | 275.8 / 1,094 |

Five processes at once (ao_render, speedtest1, so_lists, fib, so_mandelbrot), misses per 1000
checks, concurrent → one after another: Capstone 3.33 → 3.68 at 64 lines, the index with 64
slots 3.50 → 3.33, with 32-byte records 3.09 → 2.84. Interleaving changes the index's figures
by under 5 %; its slot cache absorbs 86.9 % concurrent against 87.3 % sequential. The index is
not sensitive to the process mix at this scale.

What the numbers say:

- **The slot cache is the design.** 64 slots absorb 71 to 99.8 % of the index changes: those are
  rewrites of a slot written within the last 64 capability stores. Without it (K12/W0) the index
  touches memory 87-197 times per 1000 checks and misses 2-3 times as often as Capstone on the
  two allocation-heavy programs; with it, 0.2-46 accesses per 1000 checks. Which slots those are
  is not attributed (the trace has no pc); the VM stack's top is the hypothesis.
- **Accesses: 22 to 5,900 times fewer than Capstone, 23 to 3,700 times fewer than Clover's
  baseline.** Capstone's thousand per 1000 checks are the checks themselves; they hit.
- **Misses at 64 lines: Capstone's level on the heavy programs, 6 to 47 times below Clover's
  baseline.** ao_render 6.35 against Capstone's 6.53; lc_fizzbuzz 19.5 against 7.0, the one
  program where the index misses more than Capstone at every size. The five others are below
  0.6 per 1000 for every design.
- **Larger caches favour Capstone 2-3x on the heavy programs.** At 1024 and 16384 lines
  Capstone's misses are a third to a half of the index's. Capstone's model packs four 16-byte
  nodes per line and checks hot nodes; the index touches node[m] of the capability being
  overwritten, which is colder. 32-byte records with 4 inline entries (two per line) take back
  a third of that gap, and cost nothing in accesses: the nodes that overflow 4 entries (9-10 %
  of them) are rarely touched once the slot cache is there.
- **What remains is spread over four causes.** For ao_render with 64 slots, per 1000 checks:
  flushes of evicted slots 11.0 accesses, direct deletes 4.1, node creation 2.9, revokes 2.6.
  Node creation and revokes exist in Capstone too. A new node record is a full-line write, which
  a controller can issue without reading the line; the model counts it as a miss.
- **Overflow is rare.** 0.07 to 6.7 % of changes reach a table in the mruby programs; 15.5 % in
  SQLite, whose level0 heap has three nodes with thousands of copies. Tables grow 14 to 24,569
  times per run; the grow traffic is 0.01-0.07 accesses per 1000 checks.
- **Revokes cost 3.2 to 5.5 accesses** in the mruby programs (1 per node plus the overflow lines),
  and 276 in SQLite, whose four revokes retire arenas. These figures, and the Clover baseline's
  revoke figures above, leave out the write that relinks the node after the run (Capstone's
  `cachesim` counts it): at most one access per revoke, 0.001-2.1 per 1000 checks. The models
  count it since d2b8619e's successor; the tables are from the run before. The largest, lc_fizzbuzz's 98,874-copy
  revoke, takes 106,218 accesses and would stall for them.
- **The CAM check at revoke is load-bearing**: in ao_render 443,101 memory entries named slots
  that had since been rewritten in the cache. Without the check those would be wrong tag clears.
- **Memory:** 4 bytes per live copy (48,708 to 461,447 at peak), inside 64-byte node records or
  tables; no alias pool, no sidecars.


## Pointer benchmarks on the sublet heap (`results/20261006-qemu-ptrbench/`)

The mruby runs left one question open: is the slot cache's absorption an interpreter artefact?
The programs the memory-safety literature measures pointer behaviour on answer it (SoftBound,
CETS, Watchdog and DangSan all used Olden and Ptrdist): 17 C programs from the llvm-test-suite,
built with `-O2` on the sublet heap, each checked byte for byte against the host build's output
on the same input.

**What it took to run them.** Every item below is a fact about the platform, not the design, and
each cost a run:
- **The compiler crashes on `char *x;` under `-fcommon`** (C-77, `tests/compiler-repros/C77-…`):
  seven programs stopped on it. All build with `-fno-common`; bh, which defines one variable in
  several files, links with `-Wl,--allow-multiple-definition`.
- **The sublet heap hands out 256-byte atoms** (`CAPSTONE_SUBLET_ATOM_LOG 8`) from a pool the
  launcher caps at 256 MiB of grant: at most ~520,000 live blocks. treeadd 20 (a million
  48-byte nodes) and bisort 700000 get NULL from malloc and fault on it (cause 24, value 0);
  they run at 18 levels and 250,000 elements. A ladder (12, 15, 17, 18 pass; 19 faults) fixed
  the limit before the cause was read in the source.
- **The current SDK writes the 56-byte descriptor** of the threads generation, which the
  09-30 launcher refuses; these runs use the threads lane's platform (firmware `fw-t5`, module
  `module-t5`, launcher `threads-merge/guest2`, same kernel and QEMU source). The mruby and
  SQLite runs above are on the 09-30 platform; the two cannot run each other's images.
- **voronoi** keeps an edge's rotation in the low bits of its pointer and computes siblings by
  integer arithmetic cast back to a pointer: no capability, cause 24 at first use. The
  adaptation (in `ptrbench.py`, applied to a copy) computes the same address and moves the
  pointer there by pointer arithmetic, as inline functions because one call site has side
  effects. It also needs `-DMEMALIGN_IS_NOT_AVAILABLE`: the sublet heap has no `memalign`.
- **ft and espresso** draw random numbers from the C library, and glibc's and musl's generators
  differ; both builds get the same LCG under the library's names. bisort has its own.
- **p2c** did not finish natively with the suite's arguments; gs, gawk, make and perl are their
  own ports' size. Not built.
- **Soft float.** power (0.35 s natively) ran 19.5 minutes; ks 12. The platform has no FPU.

All 17 programs ran to completion with output identical to the host build's; every variant's
error count is 0. Each ran alone, one boot each.

| | checks (M) | nodes | cap-stores /1000 | same node | index changes /1000 | copies/node p50 / max | revokes |
|---|---:|---:|---:|---:|---:|---|---:|
| treeadd (Olden) | 250 | 786,559 | 156 | 85 % | 23.5 | 1 / 326 | 10 |
| bh (Olden) | 1,120 | 8,876 | 229 | 89 % | 24.4 | 1 / 329 | 10 |
| bisort (Olden) | 339 | 393,344 | 134 | 77 % | 31.3 | 1 / 380 | 11 |
| em3d (Olden) | 301 | 6,290 | 126 | 94 % | 7.8 | 1 / 64,257 | 10 |
| health (Olden) | 391 | 896,467 | 194 | 78 % | 43.6 | 1 / 324 | 10 |
| mst (Olden) | 271 | 4,904 | 138 | 91 % | 12.3 | 1 / 1,311 | 10 |
| perimeter (Olden) | 323 | 1,048,697 | 157 | 82 % | 28.0 | 1 / 324 | 10 |
| power (Olden) | 14,010 | 54,880 | 268 | 90 % | 26.0 | 1 / 473 | 10 |
| tsp (Olden) | 1,662 | 393,343 | 248 | 81 % | 48.2 | 1 / 363 | 10 |
| voronoi (Olden) | 459 | 196,666 | 204 | 89 % | 22.4 | 1 / 386 | 66 |
| anagram (Ptrdist) | 3,905 | 18,006 | 115 | 75 % | 28.4 | 1 / 394 | 16,843 |
| bc (Ptrdist) | 5,212 | 12,586,343 | 128 | 60 % | 51.3 | 1 / 315 | 8,835,517 |
| ft (Ptrdist) | 819 | 624,615 | 99 | 88 % | 11.6 | 1 / 373 | 13,841 |
| ks (Ptrdist) | 11,894 | 6,163 | 176 | 73 % | 47.9 | 1 / 332 | 18 |
| yacr2 (Ptrdist) | 1,439 | 5,633 | 88 | 97 % | 2.3 | 1 / 348 | 384 |
| cfrac (MallocBench) | 7,319 | 15,763,280 | 204 | 56 % | 89.9 | 1 / 508 | 13,326,643 |
| espresso (MallocBench) | 2,088 | 3,663,726 | 130 | 62 % | 48.9 | 1 / 18,383 | 2,670,676 |

Misses per 1000 lifetime checks at 64 lines (4 KiB of metadata cache):

| | Capstone | Clover baseline | K12/W0 | K12/W64 | K12/W256 | K4/W64/32B |
|---|---:|---:|---:|---:|---:|---:|
| treeadd | 3.255 | 14.1 | 10.8 | 9.911 | 11.9 | 5.218 |
| bh | 1.529 | 9.330 | 2.916 | 0.324 | 0.092 | 0.376 |
| bisort | 12.7 | 42.8 | 14.0 | 7.574 | 7.411 | 5.895 |
| em3d | 2.300 | 3.759 | 2.264 | 1.801 | 1.780 | 1.777 |
| health | 5.802 | 22.4 | 11.4 | 8.073 | 9.699 | 6.122 |
| mst | 2.548 | 4.643 | 2.237 | 1.566 | 1.593 | 1.590 |
| perimeter | 5.055 | 18.4 | 11.8 | 9.478 | 12.3 | 5.967 |
| power | 0.063 | 0.887 | 0.504 | 0.188 | 0.024 | 0.180 |
| tsp | 3.356 | 12.7 | 6.216 | 1.085 | 1.219 | 0.792 |
| voronoi | 1.022 | 21.4 | 3.604 | 2.416 | 2.508 | 3.357 |
| anagram | 43.5 | 7.031 | 1.547 | 1.002 | 0.418 | 1.004 |
| bc | 1.525 | 8.998 | 5.124 | 3.309 | 2.715 | 1.696 |
| ft | 203.5 | 7.478 | 3.886 | 4.026 | 4.090 | 2.495 |
| ks | 3.584 | 21.3 | 3.796 | 0.004 | 0.004 | 0.003 |
| yacr2 | 0.039 | 0.312 | 0.087 | 0.021 | 0.018 | 0.015 |
| cfrac | 1.628 | 14.8 | 10.5 | 3.757 | 3.051 | 2.038 |
| espresso | 2.909 | 36.7 | 14.8 | 8.617 | 7.186 | 7.054 |

At 1024 and 16384 lines, Capstone → K12/W64 → K4/W64/32B:

| | 1024 lines | 16384 lines |
|---|---|---|
| treeadd | 3.150 → 8.413 → 4.769 | 3.019 → 8.400 → 4.761 |
| bh | 0.078 → 0.164 → 0.175 | 0.002 → 0.023 → 0.017 |
| bisort | 7.524 → 5.261 → 3.908 | 4.105 → 4.236 → 2.882 |
| em3d | 0.016 → 1.067 → 1.054 | 0.005 → 0.368 → 0.341 |
| health | 5.425 → 7.220 → 5.345 | 5.368 → 7.187 → 5.324 |
| mst | 0.937 → 1.082 → 1.065 | 0.005 → 1.053 → 1.033 |
| perimeter | 4.183 → 8.679 → 5.757 | 3.932 → 8.660 → 5.751 |
| power | 0.061 → 0.017 → 0.013 | 0.001 → 0.010 → 0.005 |
| tsp | 0.736 → 0.827 → 0.583 | 0.448 → 0.770 → 0.522 |
| voronoi | 0.510 → 1.388 → 1.676 | 0.284 → 1.218 → 1.273 |
| anagram | 0.002 → 0.005 → 0.004 | 0.001 → 0.005 → 0.004 |
| bc | 1.225 → 2.464 → 1.238 | 1.225 → 2.439 → 1.230 |
| ft | 163.618 → 3.186 → 2.317 | 8.511 → 2.626 → 1.677 |
| ks | 0.061 → 0.001 → 0.001 | 0.000 → 0.001 → 0.000 |
| yacr2 | 0.002 → 0.011 → 0.007 | 0.001 → 0.005 → 0.003 |
| cfrac | 1.444 → 2.171 → 1.087 | 1.080 → 2.158 → 1.080 |
| espresso | 1.038 → 2.902 → 1.867 | 0.877 → 1.782 → 1.146 |

Accesses per 1000 checks, and what the index does (K12/W64):

| | Capstone | Clover baseline | K12/W64 | absorbed | to overflow | overflow nodes | largest table (lines) | revoke accesses: mean / max |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| treeadd | 1029 | 377 | 22.05 | 76.6 % | 3.06 % | 15 | 32 | 78,659 / 786,491 |
| bh | 1000 | 391 | 5.09 | 89.5 % | 4.40 % | 1,845 | 32 | 1,638 / 16,274 |
| bisort | 1011 | 494 | 16.73 | 78.6 % | 2.01 % | 3 | 32 | 35,759 / 393,251 |
| em3d | 1000 | 120 | 2.34 | 87.5 % | 10.91 % | 517 | 8,192 | 3,903 / 38,934 |
| health | 1021 | 693 | 17.81 | 85.5 % | 1.58 % | 9,992 | 32 | 91,645 / 916,353 |
| mst | 1000 | 179 | 11.31 | 64.1 % | 34.19 % | 1,594 | 128 | 9,907 / 98,886 |
| perimeter | 1030 | 443 | 21.08 | 79.3 % | 0.92 % | 4 | 32 | 104,870 / 1,048,607 |
| power | 1000 | 415 | 0.68 | 99.1 % | 0.60 % | 3 | 64 | 5,494 / 54,788 |
| tsp | 1002 | 770 | 3.51 | 96.6 % | 0.50 % | 10 | 32 | 39,336 / 393,265 |
| voronoi | 1004 | 355 | 10.70 | 77.5 % | 3.60 % | 2,094 | 32 | 3,046 / 200,690 |
| anagram | 1000 | 447 | 2.38 | 95.6 % | 0.18 % | 4 | 64 | 4 / 1,147 |
| bc | 1030 | 804 | 17.19 | 95.0 % | 0.65 % | 4 | 32 | 4 / 2,160 |
| ft | 1007 | 182 | 6.35 | 84.5 % | 4.17 % | 1,535 | 64 | 52 / 652,436 |
| ks | 1000 | 766 | 0.01 | 100.0 % | 0.00 % | 4 | 32 | 344 / 6,065 |
| yacr2 | 1000 | 37 | 0.62 | 91.3 % | 5.66 % | 103 | 32 | 18 / 5,382 |
| cfrac | 1028 | 1414 | 18.08 | 95.9 % | 0.50 % | 3 | 64 | 4 / 96 |
| espresso | 1022 | 766 | 26.14 | 83.3 % | 4.75 % | 1,056 | 2,048 | 4 / 2,052 |

Where the index's misses at 64 lines come from (K12/W64, per 1000 checks):

| | flush | delete | node creation | revoke | table growth |
|---|---:|---:|---:|---:|---:|
| treeadd | 1.134 | 2.345 | 3.282 | 3.150 | 0.000 |
| bh | 0.251 | 0.043 | 0.008 | 0.015 | 0.007 |
| bisort | 4.226 | 0.980 | 1.209 | 1.159 | 0.000 |
| em3d | 1.307 | 0.033 | 0.025 | 0.130 | 0.306 |
| health | 1.552 | 1.792 | 2.386 | 2.344 | 0.000 |
| mst | 0.465 | 0.035 | 0.029 | 0.365 | 0.672 |
| perimeter | 0.432 | 2.475 | 3.322 | 3.247 | 0.000 |
| power | 0.145 | 0.035 | 0.004 | 0.004 | 0.000 |
| tsp | 0.408 | 0.193 | 0.248 | 0.237 | 0.000 |
| voronoi | 1.125 | 0.381 | 0.472 | 0.438 | 0.000 |
| anagram | 0.979 | 0.017 | 0.005 | 0.001 | 0.000 |
| bc | 0.464 | 0.146 | 2.528 | 0.171 | 0.000 |
| ft | 1.705 | 0.529 | 0.823 | 0.820 | 0.150 |
| ks | 0.001 | 0.002 | 0.001 | 0.001 | 0.000 |
| yacr2 | 0.008 | 0.004 | 0.004 | 0.004 | 0.000 |
| cfrac | 0.523 | 0.188 | 2.449 | 0.598 | 0.000 |
| espresso | 3.032 | 1.970 | 2.144 | 1.450 | 0.022 |

What the numbers say:

- **The slot cache is not an interpreter artefact.** 64 slots absorb 64 % (mst) to 100 % (ks) of
  the index changes in C, typically 77-96 %. The capability stores of these programs are mostly
  the compiler's spills of pointer registers into the same stack slots (every function entry
  stores its callee-saved registers), and 56-97 % of them store the node that is already there.
- **Where many objects are read broadly, Capstone's check misses and the index does not.** ft's
  Fibonacci heap (625,000 nodes): Capstone 203 misses per 1000 checks at 64 lines, 164 at 1024,
  8.5 at 16384; the index 4.0, 3.2, 2.6. anagram: 43.5 against 1.0. bisort: 12.7 against 7.6.
  Capstone pays per access over the live objects; the index pays per change.
- **Where millions of objects are created, the index pays for its records.** treeadd, health,
  perimeter (0.8-1.0 M nodes, no frees): Capstone 3.3-5.8 at 64 lines, flat across cache
  sizes; the index with 64-byte records 8.1-9.9, with 32-byte records 5.2-6.1. The breakdown
  says why: node creation 2.4-3.3 and the teardown revoke 2.3-3.2 per 1000 checks, both writes
  to records touched once, plus deletes into a million cold records. A controller writing a
  full new line without reading it removes the creation misses; Capstone's model counts its
  own mrev/split writes the same way.
- **Hub nodes are the index's weak spot.** em3d has one node with 64,257 copies (a 8,192-line
  table), espresso one with 18,383, mst 1,594 nodes with up to 1,311 copies that grew their
  tables 7,337 times. At 1024-16384 lines the index misses 1.0-1.1 per 1000 on em3d and mst
  where Capstone misses 0.005-0.016: the flushes land in cold table lines and the doublings
  rehash them (mst: 0.67 of its 1.57 misses at 64 lines are growth). A region bitmap for hubs,
  or growing by more than two, is the next design iteration to model.
- **Frees are cheap revokes.** bc (8.8 M revokes), cfrac (13.3 M), espresso (2.7 M) and anagram
  free their objects; a revoke costs 4 accesses (root, the node, the node after it, root
  again) and 0.2-1.5 misses per 1000 checks. The Olden programs never free, so their ten
  revokes are the runtime tearing down the heap: 79,000-105,000 accesses each, once.
- **Accesses:** 0.01-26 per 1000 checks against Capstone's 1000-1030 and the Clover baseline's
  37-1,414. cfrac's baseline exceeds Capstone: 90 index changes per 1000 checks at 7-8
  accesses each, which is the case the slot cache and the inline records exist for.
- **Copies per node:** median 1 everywhere; most nodes never exceed 12 (overflow takes
  0.2-5 % of changes), except the hubs.

Taken with the mruby and SQLite runs: on the fourteen non-hub programs the index with 32-byte
records is within 0.5-2x of Capstone's misses at every cache size and 20-5,000x below it in
accesses, with no check on the load path. The two things that would decide a hardware choice
and are not measured here remain the cost of a miss on a store against a miss on a load, and
the hub nodes.


## What this does not measure

- **No cost.** QEMU has no timing model. The numbers are counts of a functional access stream. They
  are not normalised per instruction: the trace carries no instruction count.
- **Only where the emulator checks.** QEMU reads a node at every capability load/store and every
  capability load. It does not read one at `cjalr` or at instruction fetch. If the RTL checks
  there too, those reads are missing. This has not been compared against the RTL.
- **A dedicated node cache, modelled alone.** In the RTL a node read goes through the memory
  channel and therefore through the data cache. Its interaction with data traffic is not modelled.
- **One trace per workload.** Run-to-run variation, measured between two boots of each of SQLite
  and sublet, is under 0.05 % on the counts.
- **One hart.** Several processes interleave on one hart at the supervisor's 5 ms quantum. They do
  not run in parallel, and QEMU keeps a tree per hart, so a multi-hart run needs that checked
  against the RTL first.

## Open questions

- Why the sublet-gc floor is flat from 64 to 4096 entries. A cyclic pass over a large live set,
  such as a GC mark phase visiting every object, would look like this, but that is a hypothesis.
  The trace has no pc, so the misses cannot yet be attributed to code. Recording the guest pc of
  the miss-heavy sites would settle it.
- The 84,110-node walk: which revoke it was.
- level0's smoke run takes 541 s against 20 to 28 s for the sublet arms, and makes 16 to 21 times
  as many capability accesses. The cause is not measured here.
