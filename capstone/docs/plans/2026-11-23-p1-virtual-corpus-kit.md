# A P1 board kit for the virtual-profile bug corpora (SQLite and PHP engine-repros): scope and the open questions

Status: scoping, 2026-11-23, runtime lane. Nothing is built. It was asked for by the nested lane (programme item 5,
the SQLite and PHP bug corpora on the board).

## Why there is no kit today

- Every non-native t3-smoke kit targets an older bitstream (8522d72fe, c721c1a2a, dcc6daf6e) or riscv-base. The
  validated board backend (bug-corpora/tools/boardvm.py, dev d4c66e750f42) ran on vp2 bf3db3175.
- vp2's kit rule (docs/plans/2026-10-17-vp2-corpus-on-silicon.md: a flat RECYCLE table of at most 65,536 records,
  PAGED never set, 16-bit ids) is vp2's. P1 is ids55 with the paged table-tree, so that kit cannot be reused by a
  bitstream swap.
- The prediction kit the nested lane uses, kit-devq, carries dev's module, which has no CAPSTONE_VM_TABLE_TREE. The
  table-tree lineage (llvm-capstone table-tree 722416c35e77, the boards' module) is 122 commits off dev.

## The candidate path (to be checked, not assumed)

1. The board side reuses the native kits' kernel: linux-board-k2n-native-hpm2tree, whose built-in capstone_vm is the
   table-tree module. Its native smoke already runs .dom domains on P1 (probe.dom, io-direct.dom and others). So
   virtual domains run under this kernel on P1 today.
2. The prediction side is the twin of that same module, perf/tree/images-tree (qemu55, ids55, the same capstone_vm.c
   blob as the board tree), not kit-devq. Tier 3's rule then holds as written: the prediction is QEMU's verdict on the
   same image bytes under the same module.
3. The corpus images must be built against the VM ABI that module serves. The runtime revision sets the VM ABI, and
   this has bitten before. The images come from an SDK whose vm-abi.h and wire.h equal the hpm2tree module's, checked
   by content before any build.
4. The launcher: boardvm expects adapter/capstone-vexec and adapter/module/capstone_vm.ko. With a built-in module the
   kit carries no .ko, so the recipe either skips insmod or the backend learns a built-in-module kit. To be decided
   from boardvm's code, not by editing it in place.

## Open questions, in order

- Q1. Does the hpm2tree module accept a virtual-profile launch (capstone-vexec) on the native kernel, or only native
  tasks and the smoke's capstone-exec? Read capstone_vm.c's entry paths and run one corpus case on the twin
  images-tree.
- Q2. Which SDK matches hpm2tree's VM ABI? Diff vm-abi.h and wire.h between table-tree 722416c35e77 and the SDK
  candidates: dev's virtual SDK, and the sdkm13b lineage.
- Q3. Table geometry on P1 for corpus images: does the virtual profile under the table-tree module allocate from the
  paged tree, as native tasks do? The 262,144-record default and the vp2 65,536 cap differ. Pre-register whichever
  holds.
- Q4. boardvm and a built-in module: what the recipe needs (item 4).

## Order of work

Q1 and Q2 are read at source and checked on the twin first. Only then is a kit cut (t3-smoke/bkit1xx-p1-virtual), and
its first boot is a Tier 1 with the corpus's own positive control (contract.dom's stale access denied with 25).
Nothing here touches dev's module.

## Q2, read at source (2026-11-23)

- vm-abi.h is identical on table-tree 722416c35e77 and on dev: version 7, two load segments, mallocng inside
  Capstone. So corpus images built by dev's virtual SDK carry the ABI the boards' module serves.
- wire.h differs. table-tree's cv_map drops the `linear` field ("copyable-only grant ABI: smaller payload gives CV_ADD a
  new ioctl number; old linear-grant launchers are rejected, not silently converted"). So dev's capstone-vexec is
  refused by the table-tree module.
- ~~The kit's launcher must therefore be built from table-tree's sources.~~ Corrected 2026-11-23: table-tree retired the
  launcher build (b9f5f1976d36, an ancestor of 722416c35e77: build-adapter.sh now exits 1 and says so). exec.c's
  ioctl launcher source is still in the tree, unbuilt. The route the native kits use is the kernel's own `.dom` exec
  (capstone_vm.native_task=1 capstone_vm.dom_binfmt=1, tests/native-task/README.md), which is how the board smoke runs
  probe.dom and the other domains. The images stay dev-built.
- Q1, restated: does a dev-built corpus image run when exec'd as a `.dom` under the table-tree kernel on the twin
  images-tree? Only if it does not is a launcher built from exec.c, and that would then be a deviation from the
  tree's own recipe.

## Q1, read on the twin (2026-11-23)

Twin: images-tree (the table-tree kernel, built-in capstone_vm), qemu55 with x-capstone-ids=55 and x-capstone-addr=sv39,
the native SPEC twins' module parameters (native_task, dom_binfmt, c5a_model, the direct-arm native_* set). One boot ran
five `.dom` files exec'd directly from a read-only ext4 stage.

| Image | Built by | Reading |
|---|---|---|
| io-direct.dom, io-exchange.dom (the board smoke's) | native SDK | entered, ran, returned rc 3: their file checks fail with errno 30 (EROFS, the stage is mounted read-only), as they would |
| control_uaf_mem5.dom (a2c6489d70b5386c), control_bounds_mem5.dom (9234c522603fda9f) | dev virtual SDK (wt-perl0002 deb4ebf0b06b) | `capstone_task: fault ... cause=26 pc=0x3ff7c02870`, SIGSEGV, before any output |
| 25_415540ddaa_fts5_getvarint_overread.dom (5cf0ec9a1350b1bd) | same | the same fault at pc 0x3ff7802870 |

- So the answer to Q1 is no. A dev-built virtual-profile corpus image does not run when exec'd as a `.dom` under the
  table-tree kernel. All three die at a pc ending in 0x2870 before printing anything, while two native images on the same boot run their own code and return.
- The 26 is QEMU's cause numbering, not the silicon table, and it has not been decoded.
- The two VM ABIs are equal (vm-abi.h 561f933ae0f4e86c on both). So the difference is the launch path: these images
  expect capstone-vexec's set-up, not the kernel's `.dom` exec.
- Two ways on, neither taken yet:
  - (a) build table-tree's exec.c launcher, whose recipe the tree retired;
  - (b) build the corpus with the native SDK, so each case is a native task on both the twin and the board, under the
    same module.
  (b) keeps the tree's own route, but it changes the corpus's images from the ones its QEMU verdicts were read on. So it
  needs a twin re-read of the corpus before any board kit.

## The launcher route, decided (2026-11-23)

The nested lane keeps the image bytes: the corpus references (the 10-29 bundle, run18) and the arms' allocators are all
virtual-profile. So the kit runs the images' own adapter.
- (a), table-tree's exec.c, is infeasible. table-tree's module has no ioctl interface: "ioctl" occurs only in comments,
  and the module registers only a binfmt. exec.c's CV_ADD and CV_IMAGE have nothing to talk to.
- A second capstone_vm.ko cannot load beside the built-in one.
- **(c), taken:** dev's own module and launcher (1a051da6e82b, the adapter the images ran under) built FLAT for ids55
  with `-DCAPSTONE_VM_FLAT_TABLE=1 -DCAPSTONE_VM_IDS=55 -DCAPSTONE_VM_INDEX_BITS=25` and the ids-55
  cap_rev_table_abi.h, on a board kernel without the built-in module. Prior art is dev
  capstone/tests/rtl-smoke/ids55-flat-module/ (PREREG 2026-10-30, RESULTS-twin.md). P1 runs flat tables on silicon
  (bkit88/89, 2026-11-16). This is P1's flat table, not its paged tree.
- Withdrawn on the way, never acted on:
  - "bkit40 carries such an adapter": it is a native kit with the adapter built in;
  - "dev's module has no ids55 knobs": the width is a source define, not a Makefile variable, and only the Makefile was
    grepped.
- Gate before any kit: on the twin, the module's index_bits and ids parameters read 25 and 55, and control_uaf_mem5,
  control_bounds_mem5 and case 25 read 25, 28 and 28 through this adapter. Then the nested lane re-reads both corpora on
  that twin, and the board pre-registration comes from that reading.

## The twin gate on kit-p1v (2026-11-23)

Kit `perf/p1v/kit-p1v`:
- capstone-qemu 0542c5556c behind a wrapper that adds x-capstone-ids=55 and x-capstone-addr=sv39 and strips exact
  bounds and test CSRs;
- kit-devq's Image, firmware and rootfs;
- dev 1a051da6e82b's launcher (2488e5d2f88169e4, the same bytes as kit-devq's) and its module built flat for ids55
  (0d5a7d627787af56, perf/p1v/buildq.sh).

The nested lane ran the gate behind the VM-lock turnstile, 11:12:04-11:12:27:
- applied -cpu `rv64,sstc=false,h=false,sv48=false,sv57=false,x-capstone-u-mode=true,svadu=false,x-capstone-ids=55,x-capstone-addr=sv39`;
- module parameters ids=55, index_bits=25, node_flat_records=1048576;
- the pools arm's controls read 25 (uaf) and 28 (bounds), and case 25 is CAUGHT with 28 in sqlite3Fts5GetVarint:
  25/28/28, as predicted.

The board's predictions come from the nested lane's full re-read of both corpora on this kit, not from the 0542c5556c
exact-bounds readings.

## The board kit: what is settled and what is open (2026-11-23)

Settled:
- The kit's module is dev 1a051da6e82b's capstone_vm.c with the twin's defines (FLAT_TABLE=1, IDS=55, INDEX_BITS=25,
  BENCH_COUNTERS), built for the board kernel (6.4.14 build-fpga, -march=rv64g, a 0 count of 2-byte instructions
  checked). Its .ko bytes differ from the twin's 0d5a7d627787af56. The prediction rests on the same source and the same
  defines. The README records the .ko sha256, the build line, and a vermagic and defines diff against perf/p1v/buildq.sh
  f55dev.
- The launcher is byte-identical, 2488e5d2f88169e4.
- Prior art for exactly this module on silicon: dev tests/rtl-smoke/ids55-flat-module, RESULTS-twin.md "Board",
  bitstream 325966ee4. The table came from CMA as a linear mapping, inside 0xac000000..0xbbffffff. Tier 1 equalled the
  twin only with resolve_dirty=1 resolve_pte=1 (S-24's fix defaults off). Kit bkit14-b4r-b2.
  So this kit loads the module with resolve_dirty=1 resolve_pte=1, and the twin re-read must use the same parameters.
- The twin appends cma=256M when the guest cmdline has none (the board DT reserves 256 MiB). Without it, PHP's flat
  table failed to allocate (nested lane, 2026-11-23).
- The nested lane's first-boot gates are the kit's acceptance:
  1. the cause mapping is fixed from the RTL before the boot;
  2. ids, index_bits and node_flat_records are read from sysfs;
  3. dmesg shows "from CMA" with no failure;
  4. supervisor_slots equals P1's hardware count;
  5. the positive control runs first, with at most one wedge risk last.

Open, before the kit is cut:
- Q5. The monitor. bkit14 ran this module under the vp2c monitor line (csbi-batch, wrapper wt-wrap-s26). P1's kits run
  native01/csbi-ids55-paged. Is there a P1 monitor that serves the virtual profile's module, and does its CSCHECK
  emulation walk a FLAT table (bkit88/89 ran flat tables on P1 with firmware 4f51541)? To be read at source.
  Candidate: bkit89's firmware, monitor native01/csbi-ids55 at 4f51541 with bkit89's FW_DEFINES (-DCAPSTONE_VP2_CSMINT
  among them) and wrapper wt-wrap-hpmpaged. It ran a flat ids55 table on P1 (B-FLAT gcc, 2026-11-16), and dev's module
  issues CSRUNV, CSMINT and CSRETIRE itself (capstone_vm.c:950, 1227, 1419).
  - The csbi-* worktrees under /tmp/capstone were removed in the 2026-11-23 cleanup. Their commits are in the
    capstone-sbi repository, and their HEADs and diffs are in attic/tmp-capstone-20261123/<name>/.
- Q6, answered by the rtl lane from d83d3a633's RTL:
  - P1 has ONE saved-context slot (csr_regfile.sv:392; runv_used/paused/terminal are single flops). So supervisor_slots=1
    on the board and x-capstone-supervisor-slots=1 on the twin. The PHP twin read at 32 slots must be redone at 1.
  - The causes are unrenumbered mcause values: stale 25, out of bounds 28, as on QEMU's twin. In the capmode-M load check
    bounds precede lifetime (load_store_unit.sv:1413-1465), so a stale capability used out of bounds reads 28.

## Board kit build, state at 2026-11-23 14:xx

- Board module built: perf/p1v/b55dev (perf/p1v/buildb.sh), capstone_vm.ko sha256
  74b4cceb69d159c14c234d88eb411b17ff3789de9153e86dbdd8b6e04f1b9b2f, vermagic `6.4.14 SMP riscv` (the twin's is `6.1.0 SMP
  mod_unload riscv`), 8,071 instructions with 0 compressed. Source, defines and build line are the twin's (f55dev); only
  KERNEL_BUILD and CROSS_COMPILE point at build-fpga's linux-6.4.14, whose .config equals the kits' board-stock.config.
- The kit carries the twin's launcher, 2488e5d2f88169e4, not b55dev's rebuild (a8a4dbe637ac758a). Both are static, so
  the launcher stays byte-identical to the twin's.
- **Blocker:** the vp2 virtual-profile recipe (bkit14's board-run.sh) replaces /capstone.ko with the process-ABI control
  module, /tmp/capstone/r55-table/fpga-pkg-127/module/capstone.ko. That file was deleted in the 2026-11-23 cleanup: it is
  a binary, and the salvage kept scripts only. build-fpga's target/capstone.ko is the stock module (its sections equal
  bkit91's capstone-stock.ko), not the process-ABI one. The native kits' adapter/control/capstone.ko has equal
  sections but was built against the native kernels.
  Options:
  - rebuild the process-ABI module from its source against build-fpga;
  - or run this kit without the b0 process-ABI controls, with the corpus's own memsys5 controls as the first steps of
    the batch rung. That would be a recorded deviation from the Tier 1 recipe.
