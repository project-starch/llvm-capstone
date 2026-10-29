# The flat ids55 module: twin grid and first board boot, pre-registered

Written 2026-10-30, before any run of this module. Branch `module-ids55`.

## What is under test

`runtime/virtual/module/capstone_vm.c` built FLAT with 55-bit lifetime ids, for option B's RTL (capstone-ariane `rtl-ids55`
567e6ee02, `capstone-docs/ids55-design.md` there): `-DCAPSTONE_VM_FLAT_TABLE=1 -DCAPSTONE_VM_IDS=55
-DCAPSTONE_VM_INDEX_BITS=25 -DCAPSTONE_VM_BENCH_COUNTERS=1`, with capstone-qemu `a9832b7180`'s `cap_rev_table_abi.h`
(the wide record format, `CAP_REV_WIDE_*`).

- The table is one flat allocation indexed by the 25-bit slot. Its size is `node_flat_records`: default 2^20 (16 MiB),
  maximum 2^25. 2^25 records would be 512 MiB per context, more than the 256 MiB CMA pool.
- Above the buddy allocator's largest block the table comes from CMA. That path is option A's commit (`module-ids20`
  866c56893fdc), carried here: `dma_alloc_coherent` through the misc device, with `table_pa` replacing
  `virt_to_phys(c->table)`.
- New in the 25-bit build only: one KERN_EMERG line for the first table a load allocates (records, bytes, CMA or buddy,
  pa, linear or not), and one for each allocation failure, up to 4.
- Width 25 compiles only as the flat 55 build. The 16, 18 and 20 flat builds, the paged ids55 build and the default build
  compile to the same code as option A's commit. That comparison pads A's source with blank lines so `__LINE__` matches.
  The comparison does fire: the unpadded comparison differed on all five builds.
- `#error` fires at all three negative controls: width 25 without IDS=55, flat IDS=55 at width 16, and paged IDS=55 at
  width 25.
- The flat build reads neither scapctl (0x5c0) nor urevavail (0xcc0). objdump counts 0 such accesses in it and 2 in the
  paged build.

Builds (`capstone_vm.ko` sha256, first 16):

| kernel | flags | hash |
|---|---|---|
| QEMU guest 6.1.0 (`build-qemu` linux-custom) | as above | 3fe888d9c595b80c |
| board 6.4.14 (`build-fpga`), `-march=rv64g -mabi=lp64d` | as above | recorded at the board boot |

The board build has no 16-bit instructions: 0 of 5,267.

## 1. Twin grid (QEMU), run after this file is pushed

Kits under `/mnt/hdd/<user>/capstone/f55/`. All three share the same QEMU binary, images and launcher: kit-w55's QEMU
binary 1d08439b20186210, built from virtual-capstone a9832b7180 plus the private Q-34 pin. Each runs with
`x-capstone-ids=55,x-capstone-addr=sv39`, 1 GiB, `cma=256M` and `-icount shift=0`. The images are cf8a024596065c29 over
the set. The launcher is 62753dd693cc4a25, built from 69fb6cae3db4.

| arm | kit | module | test CSRs | module args |
|---|---|---|---|---|
| W | kit-w55c (a private copy of kit-w55) | kit-w55's paged ids55, 1bf7d2dee537bfcd | on | none |
| F | kit-f55 | flat ids55, 3fe888d9c595b80c | on | none |
| F0 | kit-f55-nocsr | the same | off, as silicon | none |
| F0s | kit-f55-nocsr | the same | off | `node_flat_records=65536` |

W against F changes one variable, the module. F against F0 changes one variable, the test-CSR switch.

Tier 1 runs `runtime/virtual/run.py` from 69fb6cae3db4 with `--timeout 7200`. Its images are built from the same tree
with the qualified virtual SDK and cc-prmerged: contract 00428f273c653d9f, sqlite3 5494c1c306b1f1c3, mruby
152ae8dc725d59c7. The live smoke is option A's `live-T3` (1eb994b5fb979862), with N = 1000, 400000 and 1100000 objects
alive at once.

Predictions:

1. **Tier 1, F against W.** Every cell is equal except `sparse_recycling`, where F reads false by the check's own
   threshold. The check requires `nodes > 2 * node_capacity`. sparse-churn makes about 600,000 allocations
   (600,041 on the batch kit), well short of twice 1,048,576. The prediction for F's cell is false, with
   `collections=0 reclaimed=0` and nodes between 400,000 and 800,000. This is not a defect. W's own failing set is not
   predicted, because no Tier 1 record of kit-w55 exists.
2. **Tier 1, F0 against F.** Every cell is equal. Running without the test CSRs changes nothing, because the flat
   build reads neither CSR.
3. **Tier 1, F0s.** `sparse_recycling` is true: a 65,536-record wide flat table recycles through the free list. That
   means collections 0, reclaimed 0, nodes > 131,072 and node_bytes = 1,048,576. Every other cell is as F0. The table is
   1 MiB, so it comes from the buddy allocator, not CMA. The 25-bit next/prev links of the free list are therefore
   exercised on a table the run fills many times over.
4. **Live smoke, parameters.**
   - F and F0 read `ids=55`, `index_bits=25` and `node_flat_records=1048576`.
   - W reads `ids=55`, and `index_bits` and `node_flat_records` are absent; the paged build has neither parameter.
5. **live 1000.** LIVE-OK on all three arms.
6. **live 400000.** LIVE-OK on all three arms. On F and F0, `node_capacity=1048576` and nodes is between 400,000 and
   430,000. Option A's 20-bit kit read 414,217.
7. **live 1100000.**
   - F and F0 read LIVE-NULL at an index between 990,000 and 1,047,000, rc 3, nodes between 1,040,000 and 1,048,576,
     and no domain fault. That is the table's capacity reached through the ordinary ENOMEM path, scaled from option A's
     18-bit control: LIVE-NULL at 251,479 with nodes 260,455 of 262,144.
   - W reads LIVE-OK with bad=0. Its paged table grows towards 2^25.
8. **The allocation line.** On F and F0 exactly one line appears:
   `capstone_vm: flat table 1048576 records, 16777216 bytes, from CMA, pa X, linear mapping`. X lies inside the guest's
   CMA reserve, so X >= 0xb0000000 and X + 16 MiB <= 0xc0000000. That is where option A's boot reserved it; the reading
   uses the region the boot's own `cma: Reserved` line names. No `ALLOCATION FAILED` line appears. W prints no such line.

A reading outside these is reported as such, and the module goes to no board until it is explained.

## 2. The module's first board boot (Tier 1), pre-registered now, run later

A new module revision gets its own Tier 1 boot before any SPEC run uses it. This boot needs a B bitstream built from
`rtl-ids55` (or its successor). It waits for that bitstream and for a reflash, which needs the lead's word. It spends no
board time now.

Conditions:

- **What runs.** The bitstream's hash is recorded and the board is loaded with it. The board build of this module is
  hashed at bake time, and that hash and its build flags go into an addendum to this file before the boot. The
  launcher and the Tier 1 images are the twin grid's, by hash.
- **Procedure.** The board-run skill's procedure applies, including the preflight and the stubbed dry run of any
  unattended script.

Predictions, each read against F0's twin reading (test CSRs off, the silicon's configuration):

1. **Load and allocation.**
   - The insmod banner is present.
   - The allocation line reads `from CMA` and `linear mapping`. Its pa lies inside the board's CMA region. That region is
     read from the boot's `cma: Reserved` line if the console carries it. Otherwise it is the device tree's 256 MiB
     reusable pool at 0xac000000, as the runtime lane reports it; a reading that depends on that report says so.
   - No `ALLOCATION FAILED` line appears.
2. **Tier 1.** The failing set equals F0's cell by cell. Any extra failure is a board-versus-twin difference to explain
   before any SPEC boot.
3. **live 1000 and 400000.** As F0. **live 1100000:** LIVE-NULL in F0's window, with no domain fault. A board result
   inside this window, beside the twin's, is the evidence that the RTL's 25-bit flat walk and free list agree with the
   twin's.

The boot ORDER follows the skill's rule: everything expected to return runs first. live 1100000 runs last, because it
is the step that drives the table to exhaustion.

## Addendum 1 (2026-10-30, before the run): perlbench train T3 on F0, at real churn

The runtime lane asked for one more arm. The flat table at 2^20 has headroom only if freed records recycle: perlbench
train's scrabbl command creates 14,753,501 records, while its paged table grew only to 435,200. So the arm that SPEC train
depends on is recycling under a real workload. F0s exercises it only with a synthetic fill.

Run after the grid (`.grid-done`), on kit-f55-nocsr (F0): the stage `stages-trainT1/500.perlbench_r-train-capstone-lifetimes`
(image b05dcf06469c2024), `run-stage.py run --backend qemu --step-s 28800 --twin-inband-max 1048576`, as
`spec/census-w55/chain.sh` ran it on kit-w55. The reference is that run (W, paged 55): 5 of 5 rc 0, specdiff 6/6. Its
per-command `nodes` / `node_capacity` read 7,893,537/123,904; 1,047,762/13,312; 14,753,501/435,200 (scrabbl);
2,230,218/431,104; 998,956/177,152.

Predictions:

1. Commands and output. All 5 commands return rc 0, and specdiff passes 6 of 6.
2. Allocation counts. Each command's `nodes` equals W's within 0.1 %. Both count the header's allocations, pops
   included, from the same image and inputs.
3. Table shape. On F0 every command reads `node_capacity=1048576`, `collections=0` and `reclaimed=0`. Its node_bytes is
   16,777,216.
4. No exhaustion. No command meets ENOMEM or a domain fault from the table. W's largest grown table, 435,200, is below
   1,048,576 less the 1,024-record floor and the 256-record reserve.
5. Instructions. Each command's instret is within 0.1 % of W's.

Reading 2 together with reading 4 is the evidence that freed records recycle through the wide flat free list: scrabbl
creates 14x the table's records and still completes.
