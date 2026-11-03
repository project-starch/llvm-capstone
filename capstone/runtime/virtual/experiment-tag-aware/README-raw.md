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

## Summary-based clearing on the current runtime

For the second experiment, apply `linux-6.1-summary-captag.patch` after the two
patches above. It corresponds to Linux `experiment/raw-tag-mm-2`; use QEMU
`experiment/raw-tag-moves-2`. Raw mode now uses CPTAG physical-page summaries
instead of stagany in both free and swap refusal. The unrolled scalar clear
stores once per granule, eight granules per iteration, then requires a successful
verified-empty acknowledgement. A tagged page cannot pass the acknowledgement.
The old CSR boot control remains test-only. `captag=off` remains a negative
control. `captag_free=scalar` uses the same unrolled loop on every page.

The hardware flag is authoritative. PG_arch_1 already belongs to RISC-V cache
state and cannot serve as a tag flag. Every actual tag publication sets the
physical flag without trapping; ordinary writes clear tags but leave the flag
conservative. Raw copy publication sets destination flags, including COW and
migration. A copy whose source only has a conservative flag and no actual tags
copies no tags, so the destination needs no flag. The flag is cleared only after
verified emptiness. See the QEMU raw-move README for CPTAG encoding and privilege.
This is an ISA proposal, with no hardware latency or coherence qualification.

Build the module with `CV_UNPINNED=1` on current dev. Current lazy grants and
batched pinned retirement are retained; unpinned CV_TOUCH resolves through Linux
without taking payload pins. Collection stays refused. Supervisor frames still
need stable physical addresses and remain pinned. The inherited floor-only
variant is unsafe on a stock kernel; the measured unpinned arm requires the
tag-aware kernel. THP activation has been removed from the launcher/module.

The page probe now also measures allocator free of 256 clean pages and 256
tagged pages, with allocation/seeding outside the two new marker windows. The
old stagany test CSR remains its independent actual-tag oracle, alongside CPTAG,
so a defective summary cannot hide a surviving tag. Production raw free/swap
paths do not depend on a test CSR. Run this test in a private guest without
active protected domains.

Set `LANE_ROOT` and `PLATFORM` for torture-run.sh; `VM_HOST` optionally overrides
`LANE_ROOT/vmhost`. Its tested fragmentation helper must be placed in
`LANE_ROOT/torture/hog`. The PFN sample precedes compaction so the first movement
is observable. The runner supports HOG (MiB of fragmentation), TWOPROC and
MODPARAMS. Build helpers require private output paths; build-kernel.sh refuses
an existing output directory. Keep all source, VM state, outputs and lock files
inside the lane. A real pinned baseline uses its stock kernel on the same QEMU;
using the summary kernel for the pinned arm can redundantly clear its pages.
`TEST_CSRS=false` can probe independence from test CSRs. The current virtual
module still reads urevavail/scapctl during context creation, so the full
torture stack requires test CSRs even though the raw Linux tag paths do not.
The runner retains that compatibility default. QEMU's standalone summary
probes run with test CSRs disabled. The independent raw-page probe requires
them for its actual-tag oracle. Removing the module's old discovery probes
requires a separate architectural discovery interface and remains unresolved.
