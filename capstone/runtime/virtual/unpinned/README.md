# Optional unpinned domains

This branch is a proposal and qualification candidate. It requires the matching
`isa/raw-tag-moves` QEMU branch and the Linux patch here. It does not change an
ISA ruling or provide an RTL implementation. `unpinned=0` is the module default.
Set `unpinned=1` at module load; the parameter cannot change while loaded.
The module refuses that option on a stock kernel or a mismatched ID-width build.

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
two-process, stale-reference and reuse stimuli. The launcher's sleeping-child
COW stimulus exists only when built with `CAPSTONE_VM_TORTURE`; production
adapter builds contain no such hook. This is not protected fork support.

The model refuses more than one hart in the virtual profile. Protected fork
requires a separate namespace/context-cloning contract. Tagged pages shared
with an ordinary sleeping child can make a collection refuse until exclusive
ownership returns. General DMA ownership and register preservation across traps
remain OS responsibilities; the model rejects tagged raw publication into a
writable DMA mapping. Tagged swap has no save/restore format. KSM and huge pages
are outside this configuration. Instruction-attempt counts do not measure tag
cache latency, coherence, bandwidth or interrupt latency.
