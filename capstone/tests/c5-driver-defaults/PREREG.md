# C5 on the virtual gates: pre-registration (2026-11-25)

C5 makes the redundant-bounds pair (`-capstone-drop-redundant-shrink`, `-capstone-drop-redundant-shrink-declarations`)
every virtual SDK's application default (CAPSTONE_APPLICATION_REDUNDANT_BOUNDS, default ON) and puts the rest of the
Full arm's set behind CAPSTONE_APPLICATION_MEASURED_CODEGEN (default OFF). This file is written and pushed before the
gates run.

## Arms

Both arms use runtime/virtual/gates/run-arm.sh `--mode off`, with all nine gates, the same compiler (clang 29027126f39b,
binary sha256 ecd6d8c49efb...) and the same platform. Each arm builds its own SDK, contract images and applications.
- **control:** dev 634fd5531e99 unchanged (no redundant-bounds words in sdk.json).
- **treatment:** the same dev plus C5 (lane commit e1732b3faff1, compiler/c5-on-dev-634fd). Its sdk.json codegen ends
  `-mllvm -capstone-drop-redundant-shrink -mllvm -capstone-drop-redundant-shrink-declarations`.

## Platform (kit-c5, built for this run)

- QEMU at dev's capstone-qemu pin ea57497908.
  - runtime/virtual/README.md names 7dbfdd534eea. That pin is stale: dev has pinned ea57497908 since 57ce67cf6258
    (CREVOKE takes the object alone, CRENEW added).
- The adapter (capstone-vexec, capstone-job, capstone_vm.ko) and the CV_TEST_HOOKS module are built by dev's
  build-adapter.sh, against caplifive-buildroot's build-qemu kernel tree.
- Image, fw_jump.elf and rootfs.ext2 are that build's.
  - Known deviation: build-qemu is caplifive-buildroot d04bd83b13cd, 18 commits behind dev's pin 8fd1ea1249a9. Those
    commits are modcapstone and opensbi-monitor changes for the physical route.

| file | sha256 |
|---|---|
| qemu-system-riscv64 | d45acb26a27bca43 |
| adapter/capstone-vexec | 2488e5d2f88169e4 |
| adapter/capstone-job | f49e70ffac4ad90b |
| adapter/module/capstone_vm.ko | 31317f6f7dbd24a8 |
| hooks/capstone_vm.ko | 12eff2a98c9b5be6 |
| images/Image | e58613598c897103 |
| images/fw_jump.elf | da04ba67276fc9a5 |
| images/rootfs.ext2 | 9903242c37a72425 |

Prior art: an ABI-7 kit built on 2026-11-14 with QEMU 0542c5556c (the commit before ea57497908) and these same
Image/rootfs failed dev's own pthread contract. The contract faulted early, after two prints (cause 24, "Cap mem
access requires capability"), and 19 to 29 checks failed. The suspected cause is the CREVOKE/CRENEW change between the
two QEMU commits.

## Image identities (both builds finished before this file)

| image | control | treatment | |
|---|---|---|---|
| [mruby] mruby/src/mruby/build/capstone/bin/mruby | 5fa91a7427f45d02 | ef42a4124cddfadf | differs |
| [perl] perl/src/perl-5.36.3/perl | f0df70e25419a55d | d3078d4ce2018d0f | differs |
| [sqlite] sqlite/sqlite3.dom | 9814f9baee6c6a75 | 4a745facd6facbd2 | differs |
| img/contract.dom | e5273bb0b5719eb5 | 8f6c00d53d46fa7a | differs |
| img/fp.dom | 86740b58e16132ac | 396f03dfd41e5880 | differs |
| img/hosted.dom | cc0283a1b81a6597 | a9c2d08116d6f43b | differs |
| img/malloc.dom | be5ab42bd3619909 | be5ab42bd3619909 | same |
| img/native-hosted | 4630cba5b2e6a770 | 4630cba5b2e6a770 | same |
| img/native-malloc | 86a5f05a9ea1c71c | 86a5f05a9ea1c71c | same |
| img/node.dom | ef6bea0796dbfd7e | 998de21f537e3692 | differs |
| img/noreturn.dom | bb533cb9824c9bea | bb533cb9824c9bea | same |
| img/pin-ahead.dom | 50704b91d2c8b776 | 0866450f6c6d1555 | differs |
| img/pthread.dom | 31caeb2cbe901e3a | 5fb9a39276e6e87b | differs |
| libc archive libc-capstone-virtual.a | f540f2c3c641c017 | f540f2c3c641c017 | same |
| memcached libc-capstone.a | f540f2c3c641c017 | f540f2c3c641c017 | same |
| memcached/domain/memcached.dom | f5281e5ff93031f6 | 964243087fa279f5 | differs |
| memcached/marker/memcached.dom | 56856c82de8f8680 | 21aee3eed6935fcb | differs |
| memcached/pthread.dom | 74b2816b109ac7a2 | 59b98978c7d83f2b | differs |
| memcached/safety/memcached-safety-virtual.dom | ecbce1c71512756a | e3ce220e700f5b48 | differs |

- The libc archives are identical. malloc.dom (built at -O0, where the pass does not run), noreturn.dom and the two
  gcc-built native references are identical too.
- Every other image differs.
- Attribution check: the control SDK plus the two flags reproduces the treatment's contract.dom (8f6c00d53d46fa7a)
  and hosted.dom (a9c2d08116d6f43b) byte for byte. Without the flags it gives the control's contract.dom
  (e5273bb0b5719eb5). So the differences are the flags, not the tree.

## Predictions

1. **Kit check (control first, alone):** the nine gates all run, and SUMMARY.txt shows `rc=0` for each, with no
   MODE_CHECK_FAILED.
   - **Falsifier of the kit:** the 2026-11-14 shape, a cause-24 domain fault early in the contract programs, before
     their checks run.
   - If that happens, or the control fails any gate, the treatment gates are NOT run. The kit is reported as not
     serving, and C5 does not land on it.
2. **Comparison (treatment right after control):** `compare.py arm-ctl arm-trt` exits 0, IDENTICAL: every gate's FAIL
   set is equal, and no row is NOT RUN.
   - Any DIFFERENT gate blocks the landing until it is explained.
   - INCOMPLETE is not a pass.

## Amendment 1, before the re-run (2026-11-25)

The first control run (dev 634fd5531e99) read rc=1 on runtime, nodes, fp, pthread and memcached, so the stop rule
fired and the treatment did not run. The cause was two harness defects, not the kit:
- the module's one default supervisor slot on a 32-slot CPU, which refused every thread create;
- node_count_reported, which no contract ever satisfied.
Both are fixed by runtime/run-arm-supervisor-slots c51a2bb381ed, approved by the semantics owner. That fix is applied
to both arms' trees:
- **control:** dev d139d076f84e (no runtime/ports change since 634fd5531e99) plus c51a2bb381ed;
- **treatment:** this branch, which carries the same commit as a15b5eea738f.
The images are the ones built and hashed above (runner-only change). The first run's gate outputs are kept beside the
new ones, under gates-run1-noslots.

Predictions:
1. **Positive control first** (control arm, GATES=pthread, CPU x-capstone-supervisor-slots=1): `SLOTS pthread` reads
   [1], the guest logs "thread create refused, 1 supervisor slot in use", and pthread reads rc=1, as on the first run.
2. **Control** (all nine gates, CPU default x-capstone-supervisor-slots=32): rc=0 on all nine, every `SLOTS` line [32],
   no MODE_CHECK_FAILED. The stop bar is unchanged: anything else stops the run before the treatment.
3. **Treatment then compare.py:** IDENTICAL.
