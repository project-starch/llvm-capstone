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

## Addendum 2 (2026-10-30, before the bake): the board build, the bitstream, and the grid read so far

Written before the board kit is baked and before the module's first board boot (section 2).

**What runs.**

- Board module: `capstone_vm.ko` sha256 `6422805f49533ea5494b943421e3a4cb8901405a3e8fade8f5f36c9edf841438`. Built from
  `module-ids55` 3523f8833531; its `capstone_vm.c` is byte-identical to that commit's. Built against `build-fpga`
  linux-6.4.14 with `-DCAPSTONE_VM_FLAT_TABLE=1 -DCAPSTONE_VM_IDS=55 -DCAPSTONE_VM_INDEX_BITS=25
  -DCAPSTONE_VM_BENCH_COUNTERS=1`, and `-march=rv64g -mabi=lp64d` after the kernel's own `-march`. objdump counts 0 16-bit
  instructions of 5,267. Its `cap_rev_table_abi.h`, `vm-abi.h` and `wire.h` are the QEMU build's.
- Bitstream: `caplifive_vp2c_325966ee4.bit`, sha256 `f0893218dcdb53f4ad0b9fe452571f9e5c3bc5fa7ffade0025657c662c4da8cc`,
  11,443,722 bytes. The hash was re-read here from the board-side copy. It is built from capstone-ariane 325966ee4, the
  successor of `rtl-ids55` 567e6ee02 named above. The runtime lane reports it sealed and graded PASS by the rtl lane, and
  reports that the virtual-slot guard (a slot bounded at the latched header capacity) is not in this build. Both are that
  lane's reports, not checked here.
- The launcher and the Tier 1 images are the twin grid's, by hash, as section 2 says: launcher 62753dd693cc4a25. The
  perf evaluation's board kits carry a different launcher, 978c09d61f774315; this boot does not use it.
- ISSUES.md on origin/dev has no row for this bitstream at the time of writing.

**The twin grid, read so far (W, F, F0; F0s and the perlbench arm have not finished).**

- Tier 1: 34 of 45 on all three arms, with the same 11 failures. No cell differs between W and F, or between W and F0.
- Prediction 1: F's `sparse_recycling` reads false with nodes 600,041, collections 0, reclaimed 0 and node_bytes
  16,777,216, as predicted. **Not predicted:** W's `sparse_recycling` cell is false too, so W and F are equal in that cell
  as well. The cause is the check's metadata bound (run.py at 69fb6cae3db4, line 344): a hardware run needs
  `node_bytes <= 16 * capacity`, and W's paged table reads node_bytes 28,672 against capacity 1,024 (16,384). This W cell
  does not involve the flat module.
- Prediction 2 (F0 = F): holds in every cell.
- Predictions 4-7 (parameters, live 1000, 400000 and 1100000) hold on every arm. At 1100000, F and F0 read LIVE-NULL at
  1,011,158, rc 3, nodes 1,046,942, and no fault line. W reads LIVE-OK, bad=0.
- **Prediction 8 MISSED on the mapping word.** F and F0 each print the predicted line once, with `from CMA` and pa
  0xb0100000, which lies inside 0xb0000000-0xc0000000. But it reads `NON-LINEAR mapping` where `linear mapping` was
  predicted.
  - The cause is a kernel difference between the twin and the board, read from the source of both kernels.
  - The QEMU guest is 6.1. Its `dma_default_coherent` is a bare `bool` (kernel/dma/mapping.c:20) with no RISC-V writer,
    and its .config has no `CONFIG_ARCH_DMA_DEFAULT_COHERENT`. So the misc device, which has no DT node, is non-coherent,
    and `dma_direct_alloc` remaps the CMA pages through a vmap alias (kernel/dma/direct.c:248, `CONFIG_DMA_DIRECT_REMAP=y`).
  - The board kernel, 6.4.14, has `dma_default_coherent = IS_ENABLED(CONFIG_ARCH_DMA_DEFAULT_COHERENT)`
    (mapping.c:23), and its .config sets that option, so the same call returns `page_address(page)` (direct.c:296).
  - The prediction applied the board's configuration to the twin. Section 2's board prediction, `linear mapping`, stands
    and is the one the 6.4.14 source gives. The module uses the alias only for header writes, and the table's root is
    `table_pa` from the DMA handle in both cases.
- Both kernels zero the buffer in `dma_direct_alloc` (`memset(ret, 0, size)`: direct.c:302 on 6.1, :301 on 6.4.14). The
  buddy path asks for `__GFP_ZERO` (capstone_vm.c:358). Every open allocates a fresh table, and none is reused.

Section 2's predictions are unchanged. F0's reading, which they are read against, is the one above.

## Addendum 3 (2026-10-30, before the run): the twin reference for a board Tier 1 with resolve_dirty=1

**Why.** Section 2 took F0's Tier 1 cells as the board reference, cell for cell. That missed S-24 (ISSUES.md), a
silicon-only fault that the twin cannot show. On silicon, `vm`'s store after `mprotect(RW)` loops on cause 15 unless the
module loads with `resolve_dirty=1`, whose default is off (capstone_vm.c:1577). The runtime lane's boot 2 on bitstream
325966ee4 (spec-silicon c9ee688c8629, Addendum 14) loaded the module with no parameters.
- It read `vm` 139, with S-24's bound line at the launcher's fault address.
- It read 5 cells off F0's: linux_demand_faults, linux_pages_released, nested_denial_site, vm_denial_sites and
  spin_entered. The first four are that lane's attribution to the `vm` step, not checked cell by cell here.
- The miss is this file's: section 2 did not check S-24's row against the boot.

**The arm.** F0r: kit-f55-nocsr (F0's kit, module 3fe888d9c595b80c), Tier 1 only, `run.py --module-args
"resolve_dirty=1"`, with everything else as the grid's F0.

**Prediction.** F0r equals F0 in every one of the 45 cells: 34 of 45, with the same 11 failures.
- On QEMU the guest kernel handles a D-clear store fault itself, so `resolve_dirty`'s branch either never runs or runs
  without effect. The branch ignores fixup_user_fault's return value (capstone_vm.c:1768-1772) and changes no result.
- vp2's QEMU control with resolve_dirty=1 was byte-identical to its baseline (vp2-kit-2026-10-17 README, the r13
  pre-registration).
- A cell that differs is a reading about the branch on the twin, and is reported before any board boot uses F0r as its
  reference.
