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
  - Simulates LRU caches: direct-mapped, 2-, 4- and 8-way, and fully associative, each at 8 to
    65536 entries. An entry holds 1 node, or `--nodes-per-line N` consecutive nodes.
  - Write-allocate. A tree reset empties the caches.
  - Reports hits and misses per site, compulsory misses (first touch of a line), and revoke walk
    lengths.
- **`selftest.py`**: checks `cachesim` before it is used. `run.sh` runs it first and stops with
  exit 75 if it fails.
  - It compares `cachesim` against an independent Python LRU model on random traces with runs,
    resets and excluded sites, in both trace formats.
  - It also checks hand-derived cases: cyclic sweeps, a direct-mapped conflict, a reset, the
    compulsory count, and revoke walks of 0, 1 and 3 nodes.
  - Negative controls, run once: an MRU victim instead of LRU fails 62 checks, and ignoring
    repeat records fails 9. A truncated trace and an empty one both exit non-zero.
- **`run.sh <workload>`**: boots the VM with the trace going to a FIFO and runs three simulators on
  it through `tee`: 1 node/line, 4 nodes/line, and without the supervisor/gc sites. Nothing is
  stored. A mruby trace runs to hundreds of GB, and a plain trace passed 30 GB in two minutes.
- **`summarize.py`**: prints the tables below from the JSON reports.

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

## Open questions

- Why the sublet-gc floor is flat from 64 to 4096 entries. A cyclic pass over a large live set,
  such as a GC mark phase visiting every object, would look like this, but that is a hypothesis.
  The trace has no pc, so the misses cannot yet be attributed to code. Recording the guest pc of
  the miss-heavy sites would settle it.
- The 84,110-node walk: which revoke it was.
- level0's smoke run takes 541 s against 20 to 28 s for the sublet arms, and makes 16 to 21 times
  as many capability accesses. The cause is not measured here.
