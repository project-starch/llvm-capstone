# Optional unpinned domains

This branch is a proposal and qualification candidate. It requires the matching
`isa/raw-tag-moves-2` QEMU branch and the Linux patch here. It does not change an
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
It refuses activation with KSM or transparent huge pages configured; byte-only
page merging is outside this contract.
`copy_highpage` and `copy_user_highpage` query the source page's sticky flag. A
clear flag means the page holds no tag, so an ordinary copy is exact and its
stores clear any tag in the destination. A set flag selects a raw granule loop,
including bytes, instead of a scalar copy; only flagged pages pay for it.
Interrupts stay masked while the opaque temporary is live, and the pass is
repeated after a firmware timer interrupt (see "What is and is not claimed about
firmware traps"). `arch_free_page` queries each physical page's sticky flag. For
a flagged page it asks whether a tag is live (an empty answer resets the flag),
clears live pages, and verifies a second empty acknowledgement before allocator
reuse.
`arch_prepare_to_swap` refuses a folio if any constituent page may contain tags.
`captag=off` disables these hooks for negative tests; an unpinned module then
refuses to load. No test page CSR or software PTE tag flag is used.

### Optional bulk page operations

`captag_bulk=on` independently opts into bulk page zeroing and an eight-granule
raw copy loop. It defaults off, including on an otherwise valid unpinned
platform. Activation requires one possible CPU, the device-tree Zicboz ISA
extension and a declared 64-byte zero block. The firmware must permit S-mode
`cbo.zero`; this patch does not configure firmware permissions. Inspect
`/sys/module/captag/parameters/bulk_enabled` rather than inferring activation
from a successful boot. Unsupported declarations leave the original path active.

Both `clear_page` and `clear_user_page` use the zero primitive. On the bulk
path, each ordinary `cbo.zero` clears the actual storage tags by the model's
non-capability writer rule. The physical summary remains sticky. The free
hook still queries every page, skips unflagged pages, zeroes flagged pages,
and verifies empty acknowledgement before the allocator can reuse them for
either userspace or the kernel. No separate per-granule sweep follows bulk
zeroing. Raw copying preserves bytes and live or stale tags; IRQ masking and
clearing the opaque temporary before restoring IRQs remain required.

The executable [physical-page lifecycle double](lifecycle/README.md) specifies
the versioned copy/clear/query contract and rejects unsupported operations.
It is an interface fixture, not a replacement for kernel qualification.

`bulk-probe.c` is a trusted private-guest module (`obj-m += bulk-probe.o`).
Build it against the exact patched kernel. It checks every granule of two
pages containing alternating live/stale tags through both highpage copy APIs,
then checks mixed live/stale/scalar/zero cargo with unique cursors against a
previously fully tagged destination and a clear-flag source over a fully tagged
destination. Representation checks sample both stored
words after invalidating a byte outside the sampled word. It checks zero
contents and sticky-versus-actual tag state, and requires same-frame
order-1 kernel reuse without `__GFP_ZERO`. Run each arm through the host gate
queue:

```sh
python3 capstone/runtime/virtual/unpinned/run-bulk-probe.py \
  --qemu "$PLATFORM/qemu-system-riscv64" --images "$PLATFORM/images" \
  --module "$PROBE/bulk-probe.ko" --work "$RESULT" --bulk on
```

Repeat with `--bulk off`. For negative controls, separately build a kernel
substituting scalar `copy_page` for the raw copy and use `--expect lost-copy`;
then build a kernel omitting only the bulk free-hook clear and use
`--expect stale-free`. The latter must hit the existing nonempty-acknowledgement
BUG after the fixture's successful zero tests. Neither an early boot failure
nor a generic crash counts. Keep these broken kernels out of production.
The probe's independent scalar cleanup makes the lost-copy case safe to free.
Also remove both the clear and its acknowledgement assertion and use
`--expect leaked-tag`: the reuse observation itself must find the surviving
tag. The runner disables allocation/free zeroing so these controls cannot be
masked by generic allocator initialization.


This experiment needs actual COW/compaction/fork/reuse torture, unchanged
runtime gates, matched quick/full census and independent review before landing.
The sleeping-child torture remains a COW stimulus, not protected fork support.
Model instruction attempts do not establish hardware latency or bandwidth.

### What is and is not claimed about firmware traps

Masking S-level interrupts does not stop a firmware-handled interrupt, and a
firmware trap saves and restores integer registers only, so the opaque
temporary between a raw load and its store does not survive one. On the
instruction-exact clock the census platforms use (`-icount shift=0`) and the
firmware timer (no Sstc), `irq-window-probe.c` places a firmware timer deadline
1 to 24 ticks after an arming call and runs the kernel's own `copy_highpage`
over a page tagged on every granule. A kernel with one raw pass and no redo lost exactly one granule's tag in 372 of
3000 iterations (bulk off; 1990 had the deadline inside the copy call) and 281
(bulk on; 865 inside); the final kernel lost none in 3000 and 3000 and redid 1859
and 753 passes; 300 iterations with deadlines outside the copy lost none. The raw
transport inherited from the raw-move model has the same exposure. Without
`-icount` interrupts are taken at translation-block boundaries and no tag was
lost (3000 iterations; 2 redos). The probe pads after the arming call: arming
re-aligns the instruction count to the timer grid, and with the padding before
it only three of the loop's six instruction boundaries were ever hit (traced with
`-d int`), which made one no-redo layout look safe by accident. With the padding
after it, both of the layouts without a guard (no redo, no mask) lose tags.

The kernel's own interrupt mask is covered separately: with `unmasked=1` the
probe enables S interrupts around the copy, as a real caller does. The final
kernel lost none (3000 bulk-off iterations with 1864 redos, 3000 bulk-on with
759); a kernel without the `local_irq_save`/`restore` around the query and copy
lost the tag in 408 of 3000.

The kernel therefore repeats a pass when a firmware timer interrupt was served
during it. OpenSBI 1.3 serves that interrupt by clearing MTIE and setting STIP,
so STIP clear before the pass and set after it identifies one; the source is
unchanged (one hart, S interrupts masked across the flag query and the copy), so
the redo is exact. STIP already set before the pass implies MTIE is clear, so
none can arrive, and the second pass always ends the loop; a second detected
interrupt raises `BUG` before a third pass can start. `copy_retries` counts redos. The runner reports its change across the fixture.

Claimed: the tested firmware and model, with Sstc off. Not claimed: any other
firmware or silicon; any other asynchronous machine-level event (software or
external interrupts, a firmware-owned timer under Sstc, which sets no STIP);
an interrupt-deferral guarantee in the ISA (none is added or assumed). A
platform that cannot give this contract must keep the raw transport off. A
guest booted with `sstc=true` in this model did not finish booting within four
minutes and was stopped; no Sstc statement is made.

The supported stability assumption for the copy: nothing else writes the source
while it is copied. On one hart with S interrupts masked, that excludes all other
ordinary CPU execution; DMA never publishes tags. Linux itself does not give this
for every source: a private mapping of a shared file page that is also writable
through another mapping, for example, is outside what this patch supports. Migration
and compaction lock and unmap the page, and fork COW of private anonymous memory
write-protects it, which is the tested configuration. One hart is required.

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
