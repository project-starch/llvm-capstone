# Optional unpinned domains

This branch is a proposal and qualification candidate. It requires the matching
`isa/raw-tag-moves-2` QEMU branch and the Linux patch here. It does not change an
ISA ruling or provide an RTL implementation. `unpinned=0` is the module default.
Set `unpinned=1` at module load; the parameter cannot change while loaded.
The module refuses that option on a stock kernel or a mismatched ID-width build.

On the `experiment/native-faults` branch, `unpinned_fast_fault=1` additionally
reuses the already-validated VMA for private anonymous faults. This parameter
defaults off and has no effect on the pinned path. It calls `handle_mm_fault`
with the same write-fault policy and without retry/killable flags, under the
existing mmap read lock and grant lifetime lock. File-backed and special VMAs
retain `fixup_user_fault`; image-text handling is unchanged. Grant bounds,
VMA permissions, repeat bounds, stop checks, tag-preserving mm and retirement
rules remain in force. See [the fault-cost experiment](fault-cost/README.md).

The QEMU option is `x-capstone-raw-moves=true`, with `x-capstone-u-mode=true`.
The runtime platform uses `x-capstone-test-csrs=false`. The model supplies a
`capstone,unpinned-v1` device-tree node reporting its ID width, exact-bounds
mode, page size and granule size. The patched kernel validates that declaration
and exports `capstone_unpinned_config`; the module resolves it only when opting
in. No capacity or bounds mode is inferred from a failed instruction probe.
Default pinned discovery retains its existing behavior.

## Kernel and build

`linux-6.1-unpinned.patch` is a complete patch against upstream Linux 6.1
(commit `830b3c68c1fb`). Platform-dev4's Image reports Linux 6.1.0. Its embedded
config has SHA-256
`9c7247993f725f3f4313605a8ec1785177d6fe9906692b3c8f181231f0aa59e1`.
Extract that exact config with `scripts/extract-ikconfig`; do not use a generic
defconfig. KSM and THP are disabled. Swap is configured, but tagged swap is
refused. Keep builds and writable state in a private directory on the build host.

```sh
# In an otherwise unmodified Linux 6.1 source tree:
git apply /path/to/linux-6.1-unpinned.patch
scripts/extract-ikconfig "$PLATFORM_DEV4/images/Image" > .config
make ARCH=riscv CROSS_COMPILE="$CROSS_COMPILE" olddefconfig
make ARCH=riscv CROSS_COMPILE="$CROSS_COMPILE" -j4 Image modules_prepare modules

# In this runtime checkout; KERNEL_BUILD is the prepared patched tree.
source capstone/tests/capstone-test-env.sh
CAPSTONE_BUILD_JOBS=4 bash capstone/runtime/virtual/build-adapter.sh "$ADAPTER"
bash capstone/runtime/virtual/unpinned/platform.sh \
  "$PLATFORM_DEV4" "$RAW_QEMU" "$KERNEL_BUILD" "$ADAPTER" "$NEW_PLATFORM"
```

The adapter build also needs `capstone/capstone-qemu/target/riscv/cap_rev_table_abi.h`
from its matching QEMU. The recipe makes a real-file copy of platform-dev4,
replaces QEMU, Image, System.map and the adapter, and regenerates all file hashes.
The JSON settings it records are instructions for the runner, not automatic
activation. For a SPEC arms file, set `platform.cpu_props` to the recorded CPU
properties and the Capstone arm's `module_params` to `unpinned=1`.

The patched kernel enables tag-aware mm only after discovering the opt-in node.
`copy_highpage` and `copy_user_highpage` use a raw granule loop, including bytes,
instead of a scalar copy. Interrupts stay masked while the opaque temporary is
live. `arch_free_page` queries each physical page's sticky flag, clears flagged
pages with scalar stores, and verifies acknowledgement before allocator reuse.
`arch_prepare_to_swap` refuses a folio if any constituent page may contain tags.
`captag=off` disables these hooks for negative tests; an unpinned module then
refuses to load. No test page CSR or software PTE tag flag is used.

## Ownership and lifetime recycling

Payload faults use Linux's fault handler without persistent GUP pins. Supervisor
frames remain pinned because CSRUNV identifies them physically. Lifetime
retirement still precedes unmapping; it does not recycle an identity by itself.

Collection temporarily holds references to tagged resident pages from the entire process mm,
including retired arenas still mapped. It holds mmap_lock through the hardware
collection and identity reuse, preventing fork/unmap from moving copies beyond
the inspected set during that operation. Saved context frames are included.
Every temporary reference is released on success and failure. Shared tagged backing,
unsupported large-page mappings, migration entries and swapped PTEs refuse the
sweep; an incomplete sweep never permits recycling.

Before publishing a fresh namespace in an mm, the same scan removes old tags
left by a prior close/open. A raw load followed by a scalar store rewrites the
original low word, retaining bytes while clearing the storage tag under rule 1.
This avoids scalar-load projection. The global context lock excludes a second
active namespace in the same mm.

These paths still require qualification; the accompanying lane report is the
source of measured results and unresolved failures. No result is inferred from
successful compilation.

## Tests and limits

`torture.c`, `pfns.c`, `reuse.c` and `torture-run.sh` retain movement, COW,
two-process, stale-reference and reuse stimuli. `build-torture.sh SDK OUT`
builds them and the optional `hog` fragmentation helper with CROSS_COMPILE set.
Set OUT to LANE_ROOT/torture for the runner. It returns failure for a nonzero
normal result or a stale run that fails to report the expected revoked-node
fault (cause 25). The launcher's sleeping-child
COW stimulus exists only when built with `CAPSTONE_VM_TORTURE`; production
adapter builds contain no such hook. Select it with `CAPSTONE_VM_TORTURE=1`
when running build-adapter.sh. This is not protected fork support.

`mm-probe.c` is a standalone kernel module (`obj-m += mm-probe.o`) for an
otherwise idle private guest. It checks refusal to acknowledge a live tag,
copy_highpage tag retention, and same-frame free/reallocation with no tag.
With `captag=off`, load it with `expect_copy_tags=0`: scalar copying must lose
the tag, and the production module must refuse `unpinned=1`. The fixture
scrubs all of its pages before freeing them, including in this negative mode.
It uses only the proposed raw instructions and architectural mint/root setup.

The model refuses more than one hart in the virtual profile. Protected fork
requires a separate namespace/context-cloning contract. Tagged pages shared
with an ordinary sleeping child can make a collection refuse until exclusive
ownership returns. General DMA ownership and register preservation across traps
remain OS responsibilities; the model rejects tagged raw publication into a
writable DMA mapping. Tagged swap has no save/restore format. KSM and huge pages
are outside this configuration. Instruction-attempt counts do not measure tag
cache latency, coherence, bandwidth or interrupt latency.

## Experimental qualification status

The module parameter defaults to `unpinned=0`; the matching QEMU property
`x-capstone-raw-moves` also defaults off. Leave both off for landing. The
Linux patch is an explicit opt-in platform component, not applied by this tree.

The completed earlier three-arm qualification found default-OFF identical to
dev across all nine runtime gates. ON differed in runtime, pin-ahead, pthread
and step-hooks: 87 additional named failed checks. Pin-specific predicates
need mode-specific replacements with positive execution witnesses. Four
point-1 fault/stop stimuli did not inject, and five extra-fault attributions
remain unresolved. No gate predicate has been relaxed.

The earlier instruction-attempt census measured gcc boundary overhead
48.3 -> 37.2 percent and lbm 3.69 -> 0.70 percent. These are EXPERIMENTAL
ON results, not default-OFF improvements, cycle counts or elapsed-time claims.
They do not establish complete ON qualification. One-hart private-anonymous
lifetime recycling and tested process exits passed; protected fork, SMP,
shared tagged backing, tagged swap and a whole-system leak bound remain outside
that evidence.

## Current-dev integration

The default remains OFF. The branch includes dev `6ae238ca3022` without
rewriting its previously published history. The main exchange (register 11)
is lazy and uses `prepare_write` / `CV_TOUCH`; only the metadata grant
(register 10) stays eager. In unpinned mode admission faults a page in without
pinning it. The launcher's successful-admission cache is valid for the fixed
main-exchange mapping: later backing changes are covered by the experimental
kernel copy/free hooks, rather than by continued residency. Retirable worker
transports continue to touch each attachment.

If an unpinned collection refuses a non-present or shared tagged page, node
allocation now tries fresh-record growth while keeping unswept IDs retired.
Collection and open still fail closed on unsupported mappings; this does not
qualify swap/migration pressure or make ON production-ready. All ON results
remain **EXPERIMENTAL** and the qualification gaps above still apply.
