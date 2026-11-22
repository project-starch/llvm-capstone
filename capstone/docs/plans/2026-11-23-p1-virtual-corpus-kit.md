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
