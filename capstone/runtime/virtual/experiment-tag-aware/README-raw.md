# Raw granule transport experiment

Apply linux-6.1-raw-captag.patch after linux-6.1-captag.patch. The incremental
patch is the local Linux experiment/raw-tag-mm branch change. Use QEMU
experiment/raw-tag-moves with x-capstone-u-mode=true,
x-capstone-test-csrs=true and x-capstone-raw-moves=true. The last property is
default off. Keep THP disabled.

Boot controls:

- `captag=raw captag_free=tagged` (default): replace copy_page with 256 raw
  load/store pairs; query freed pages and scalar-clear only tagged pages.
- `captag=raw captag_free=scalar`: scalar-clear every freed granule.
- `captag=csr`: the test-only page-CSR copy/clear control.
- `captag=off`: ordinary copies/frees, the negative control for tag loss.

The raw path is translated, non-consuming and bypasses scalar-load projection.
It preserves exact-bounds model metadata. Interrupts stay disabled while t0
holds opaque cargo, because Linux's ordinary integer trap frame loses it.
The test-only stagany query still refuses tagged pages for swap. This is not
a swap implementation or a proposal for a whole-page instruction.

Use the existing torture program and runner, with lane-owned paths, VM state,
platform copies and raw-move CPU property. Its launcher fork hook induces COW
in the parent; it does not implement a protected child or clone its lifetime
namespace. The runtime cannot establish child-free/parent independence.

Build `reuse.c` with the same application SDK as the torture program. `reuse`
checks that a newly acquired mapping contains untagged slots. `reuse stale`
then dereferences a retained capability after munmap and must fault. It prints
whether the virtual address was reused; that alone is not physical-frame reuse.

`raw-page-probe.c` is a small kernel test module for a private VM with no active
protected domains. Build with raw-page-probe-build.sh against the matching
kernel. Load under capprof: it emits two windows, each with 1024 copies of the
same two untagged pages, first copy_page and then copy_highpage. Interrupts are
masked during each measured phase. The windows include loop/call overhead;
use their difference for the incremental raw-loop cost. Instruction attempts
do not measure latency, bandwidth or hardware tag-cache cost.

After measurement the module mints a storage tag with a temporary private
node table, restores srevroot, tests tagged copy_highpage, frees the tagged
source page and reallocates. It requires both the same physical frame and
an untagged result, printing the actual observations. The module is test-only;
it changes the active table briefly under local interrupt exclusion and is
not a production interface. Its PASS is separate from the application stale
capability oracle.

The first four copy-related Linux counters retain the prototype's one-hart
scope. This QEMU virtual profile rejects multiple harts; SMP, protected fork,
and full DMA/page-ownership protocols remain unqualified.
