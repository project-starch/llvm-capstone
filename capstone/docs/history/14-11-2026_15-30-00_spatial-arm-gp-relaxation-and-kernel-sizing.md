# The table's spatial arm, gp relaxation on both arms, and a kernel-lever sizing (2026-11-14)

This records:
- the work for the lead's three-arm table (riscv-base, temporal, spatial);
- the gp-relaxation lever, built for both arms;
- the sizing of a kernel lever (the pin scan);
- one retraction.

All figures are native-twin user instructions, with outputs compared byte for byte, unless marked otherwise. Each
arm's own validation is quoted.

## FRCDKMS: the five remaining rows

- Images: arm FRCDKMS (llvm-c9 29027126f39b, the perf/memset SDK), all eight rows. Hashes are in
  perf/cclever/arms/FRCDKMS/RESULTS.
- Twins against FR: outputs identical on imagick, nab, lbm, xz and x264.
- x264's log.dec differs only in ldecod's wall-clock field. Its "specdiff 0/3" is the twin stage's: imagevalidate never
  runs there, on either arm.

## The spatial arm (FRCDKMSsp)

Definition: CAPSTONE_HEAP_LIFETIMES=OFF, SCRUB_FREE=OFF. That is the ports manifest's capstone-spatial and the 3 Nov
board arm (spec/sdks-dev2/sdk-spatial).
- SDK perf/memset-spatial is perf/memset's script with that one option changed. By archive member it differs in
  exactly libc's four mallocng objects and the runtime's malloc-mapping.c.obj; builtins are identical.
- On this lineage the heap is musl's mallocng inside libc, and build-sdk.sh passes the heap switches to libc. So
  "differs in heap.c.obj only" (the ports README, an older lineage) cannot hold literally.
- All 774 benchmark objects of the eight rows are byte-identical to FRCDKMS's.

What lifetimes OFF changes, measured on gcc (entries identical):
- `__libc_malloc_impl` +2.85 instructions per call: CDERIVE becomes cursor, add, CSHRINK.
- `__libc_free` −2.0 per call: no CREVOKE.
- Mapping, cache, eviction and in-place realloc are unchanged to the instruction.

Twins against FRCDKMS:
- Outputs identical on seven rows: gcc +1.2 M, the rest within ±0.02 %.
- Perl full, on the twin kernel with the board's capstone_vm (below): both arms specdiff 2/2, outputs identical
  across 12 files, user +20.6 M (+0.02 %).

## Retraction: perl test-sw's kernel share

I reported "perl test-sw kernel 3.45 G, 86 % of user" as a Capstone cost. It is withdrawn.
- Every Capstone native perl test-sw twin validated "specdiff 1/2": test.pl's sub-tests failed throughout.
  - The stage maker predated nat-fair-sdk e4631fa7ac3d.
  - The native twin kernel (capstone_vm 2d380284ee86) has no dom_grant_root.
- riscv-base ran the same stage fully (2/2, 75.7 G user in test.pl).
- The fix is a twin kernel with the board's capstone_vm a0ed3250249b, booted with dom_grant_root=/tmp/nt/sr,
  dom_null=1 and dom_local_tmp=1: perf/natio/images-a0ed.
  - Its COMMAND_LINE_SIZE is raised to 1024, as the board tree's riscv setup.h has it.
  - The twin runner's limit is now TWIN_CMDLINE_MAX.
- On it, FRCDKMS perl test-sw reads 2/2, "All tests successful".
- Lesson: a cross-arm "IDENTICAL" says nothing when both arms fail validation. perf/cclever/cmpc.py now warns when an
  arm's own validation is incomplete.

## Kernel-lever sizing: the pin scan

A pin-scan lever was proposed: pin_remaining and next_resident in capstone_vm.
- Kernel PCs were mapped to the twin kernel's System.map (perf/kpin/ksyms.py).
- The scan comes to 0.01 % of user on gcc, mcf and xz, and 0.36 % on perl test-sw. That last run was the broken
  test.pl, so it is an upper bound at most.
- Not built.
- The Capstone-only kernel cost that stands out is capstone_vm's retire: 5-5.5 % of kernel instructions on gcc, mcf
  and xz, mostly clear_page_tags. Every retired pinned page costs one store per granule.

## gp relaxation, both arms

Design:
- **lld (compiler-gp-relax 784137547b99, 81193d42edb0).** PC-relative gp relaxation under --relax-gp, all or nothing
  per auipc, as GNU ld performs it. On Capstone it also needs the scc's R_Capstone_GP_SCC marker
  (`-capstone-mark-gp-relaxable`, relocations only).
- **Small data.** `-capstone-small-data-limit=16` (pointer-sized) on Capstone; `-msmall-data-limit=8` on riscv-base.
- **Runtime (runtime-gp-cursor e120e53b053f).**
  - gp's cursor is __global_pointer$ in every context.
  - The transport lock's identity moved out of gp's cursor (option B: single-context constant, then service
    identities).
  - The link script gains a small-data block.

Why both arms: lld relaxed only the absolute lui pair, and only under --relax-gp. The board-3 riscv-base images had no
gp-relative access at all. A GCC and binutils riscv-base would have it by default.

The GlobalMerge policy was pre-registered (docs/plans/2026-11-14-gp-relax-globalmerge-rule.md) and decided by
riscv-base alone. The outcome is LLVM's default on both arms.

| row | riscv-base RVGP vs RVF3 | Capstone GPR2 vs GPC (control GPC vs FRCDKMS: +681 / +260 / +645) |
|---|---|---|
| perl makerand | −239.6 M (−8.7 %) | −391.7 M (−12.7 %) |
| gcc train01 | −46.0 M (−0.42 %) | +82.0 M (+0.67 %) |
| mcf test | −0.47 M | +0.28 M |
| imagick, nab, lbm, xz | | −4 k, +9.1 M (+0.13 %), +13, −80 k |

Gates:
- Native-route directed test (tests/native-task/gp-cursor): 5/5 on both arms. The handler and context checks do not
  apply on that route: the kernel refuses caught handlers (EOPNOTSUPP) and further contexts (EAGAIN). Its mutant fails.
- hosted contract equal to FRCDKMS's.
- lit 221 of 223. Both failures are shared with llvm-c9.

Adoption is decided on silicon (perl makerand and gcc pairs on both arms). The dev landing gate is deferred until then.
The plan document records what it requires.
