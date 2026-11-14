# perl test.pl's kernel excess: the per-exec census, the node-table pool, and two retractions (2026-11-14)

This note covers the kernel work on perl's full test.pl (318 native execs per run), from the compiler lane. All
runs are native QEMU twins with outputs compared byte for byte, and each arm's own validation is quoted. Figures are
twin instructions, not silicon cycles. The twin kernel's DEBUG_VM and MEMCG config also inflates per-page cost
relative to the board (see the mcf attribution of 13 Nov).

## Instruments

- **Census kernel** perf/census (MEASUREMENT ONLY; build-host local): images-a0ed's tree plus two switches.
  - `capstone_vm.diag_census` prints one retire line per arena: task, context, count, kind flags, and whether it is a
    teardown. There are no per-page lines.
  - `exit_census` is a core parameter: one `@@EXIT` line per exiting process, with RSS high-water and faults.
  - census-2 adds a per-context node-table line.
  - Analysis: census2.py, which groups by (tgid, context). The context's kernel address alone is reused across execs.
- **kcmp.py** (perf/kpin) now splits PCs at the kernel's load address 0x80200000:
  - below it is firmware;
  - above it is kernel text at its physical alias, symbolised as VA.
  - `KCMP_FINAL=1` compares whole runs, for stages without mark windows.

## Correction: the "OpenSBI +0.56 G"

The first test.pl kernel table reported OpenSBI at +0.56 G, but the bucket was kernel memset, which on the twin runs
at its physical alias. The evidence: memset's 256-byte loop counts 20.8 M at PA 0x808ff0f4 and 171 at the VA copy.
OpenSBI itself is +0.3 M. In such a profile the count at an entry PC is not the number of calls.

## The per-exec node table

- Every native exec's context_create cma_allocs a flat lifetime table of node_flat_records × 16 B and memsets all of
  it. Destroy cma_releases it.
- In test.pl's windows: 318 cma_alloc, all on the success path.
- At the board kits' 262144 records (4 MiB), a perl exec's table peaks at:
  - mean 7,791 records, p50 5,492, p99 54,928, max 132,493;
  - 318 execs, census-2.
- Most of this week's twins used 524288 records, twice the board's table: 209 runs did, 17 used 262144, and 14 used
  the default.

## The node-table pool (native-hpm-counters 7719bccb8ca4; lane branch node-table-pool)

- `node_table_pool=N`, default off.
  - A released table is zeroed over [0, next) and reused.
  - The argument that this restores a fresh table, and why every record matters, is in the module comment. It rests on
    capstone-ariane rtl-ids55 afd35e449, read by an RTL oracle and re-checked on the load-bearing lines:
    - no record cache keyed by table PA;
    - the revocation cache is flushed on an srevroot edge;
    - virtual reads are unbounded;
    - vp_load_hdr reads table[2..3].
- Directed test: tests/native-task/node-table-pool, with mutant controls that fire.
- perl test-sw, pool off against pool=4 on one kernel, both 2/2 with identical outputs:
  - kernel −773 M (−2.97 %);
  - pgfree −325 k;
  - user unchanged.
- Board A/B: bkit81, on F.

## Retraction 1: the "fault-ahead windows"

- I attributed 460 k of 530 k pinned pages to adaptive fault-ahead windows. They were arena_pin's rest-of-arena pins.
  pin_ahead counts both in the per-arena `windows` field.
- arena_pin=1 came from my twin perl stage (`perf/cclever/camp/{p,vm}-*q-500.perlbench_r-test-sw`). That stage carries
  the gcc train01 tuning set: step_budget_ns, resume_quantum, retire_exit_sweep and arena_pin.
- A run with fault_ahead_max=0 was void, because arena_pin makes it moot.

## Retraction 2: "the board's perl stage sets arena_pin too"

- It does not. Only gcc's stages write module parameters.
- The claim came from two `grep -n` runs over two files without `-H`: the twin's line 18 was read as the board's.
  The board file has 53 lines, and the output printed a "line 69".

## Board-faithful numbers

The stage st-perl-bf is the board's run.sh plus the `@@VM` lines; it uses the board stage's binary a5794577.

- **User instructions** do not depend on the tuning. Tuned against board-faithful, same kernel: +250,508 (0.0003 %),
  outputs identical. So the week's full-perl user counts stand. Their kernel figures are the tuned configuration's.
- **Kernel:**
  - board-faithful 22.25 G against riscv-base 17.86 G: +4.39 G, at 524288 records;
  - the tuned stage read +4.73 G;
  - main parts: page allocator +1.06 G, capstone_vm +0.95 G, memset +507 M (about 354 M of it the 8 MiB table),
    `__free_one_page` +236 M.
- **Pages per exec, Capstone pinned (census-bf, 262144):** 1,258 in all.

  | region | pages |
  |---|---|
  | heap | 1,095 |
  | image | 152 |
  | stack | 6.3 |
  | meta | 4 |
  | exchange | 1 |
  | startup | 1 |

  - RSS high-water: 1,788 pages against riscv-base's 1,014.
  - pgalloc: 455,069 against 234,675.
  - The runtime does not pre-touch the 1 MiB stack.
- **The heap** is about 200 mallocng-group arenas per exec, each 83-100 % used. These are touched pages, not
  over-pinned ones.
- **A pinned-page pool** (lever a) could save at most the allocator and freeing share, under 1 % of the row's
  instructions. It is deferred until the board A/B reads.

## Other measurements sent this session

- **gp relaxation, full test.pl:**
  - GPR2 against GPC: −2.78 % user;
  - riscv-base RVGP against RVF3 (the same image as RVB): −2.38 %;
  - x264: GPR2 equals GPC.
- **For the stage-2 ISA sketch** (perf/cclever fpcount.py and regclass.py):
  - executed c.fld/c.fsd/c.fldsp/c.fsdsp: 10.05 % of nab's user instructions, 5.67 % of lbm's, 0.33 % of imagick's;
  - the x8-x15 forms capture 4.16 % (gcc) and 8.67 % (perl makerand) of the 7.31 % and 9.71 % that shapes 2, 5 and 6
    cover.
