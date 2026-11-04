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

The model refuses more than one hart in the virtual profile. General DMA ownership and register preservation across traps
remain OS responsibilities; the model rejects tagged raw publication into a
writable DMA mapping. Tagged swap has no save/restore format. KSM and huge pages
are outside this configuration. Instruction-attempt counts do not measure tag
cache latency, coherence, bandwidth or interrupt latency.

## Status: EXPERIMENTAL, default OFF

`unpinned=0` is the module default and `x-capstone-raw-moves` defaults off in the model; leave both
off. The Linux patch is an explicit platform component this tree does not apply. Nothing here is a
claim about silicon: raw tag moves are an ISA proposal with no RTL, the boards cannot run this mode,
and the flat node table never collects there (Q-27). The pinned path stays fully supported.

What a gate arm runs, and in which mode:

- `gates/run-arm.sh --mode on|off OUT` runs the nine landing gates (hosted, runtime, nodes, malloc,
  pin-ahead, fp, pthread, step-hooks, memcached) with `--unpinned` for `on` in EVERY runner. Each
  runner loads the module itself, echoes the mode the loaded module reports after each insmod
  (`<GATE>_MEMORY_MODE:Y|N|A`; `A` is a module without the parameter, a pinned-only build) and carries a `memory_mode_verified` check. `compare.py` refuses to compare an
  output with no mode file, a gate with no mode line, or a line that is not the requested mode, and
  `--control-mode` / `--arm-mode` pin the pair (an ON arm is compared with an OFF control).
- Optional gates for `on`: `unpinned-mm` (the isolated swap-hook and namespace fixtures, needs
  `UNPINNED_PROBES`) and `unpinned-mutants` (broken modules, below). The arm also checks that the
  production module carries none of the test-hook strings.
- ON has its own resolution oracles, because the pinned page counters read zero by construction. They
  read what Linux says: the kernel's FOLL_PIN counters (step-hooks and pin-ahead require them not to
  move during the unpinned run, after a launch in the same guest has shown that they do move: the
  pthread contract's supervisor frames); the launcher's VmRSS/VmHWM (`CAPSTONE_VM_RSS=1`) for pages given
  back and for the 64 MiB the `vm` launch never touches; per-fault event records against the public
  fault counter; window and event pages that never meet. Every OFF oracle is unchanged.

Negative controls are broken modules, not edited transcripts. A CV_TEST_HOOKS module has parameters
that each remove one guard (`test_skip_vma_clip`, `test_skip_arena_clip`, `test_skip_stop_check`,
`test_skip_resident_stop`, `test_skip_namespace_scrub`, `test_skip_scrub_after_pages`,
`test_skip_scrub_after_granules`). `run-mutants.py` requires a named check to pass on the unbroken module
and fail on the broken one for the VMA clip, the stop check and the stop at the first resident PTE; the
arena clip is not isolated by any scenario a gate can build (the VMA clip or a resident guard page bounds
the window at the same page), so `test_skip_arena_clip` exists but no check can see it. `run-mm.py` does the same for the scrub with a fixture that seeds 21
granules (offsets 0, 16, 2048, 4080) on 10 pages of four mappings. `test-oracles.py` additionally
corrupts real ON transcripts and requires each corruption to break the named check (it proves the parsers
reject contradictory lines, nothing about a module).
The reclaim and tag-clear evidence for devices (DMA, fw_cfg, virtio) is model-side and not part of this
tree.

Not covered by an ON test and rested on the kernel hook tests alone: that a retired thread stack's tags do
not survive its reuse (ON clears at `arch_free_page`, not at retire; the pthread check is an accounting
check), reclaim and swap under memory pressure (only the hook is tested; tagged backing is refused),
page migration and compaction (non-present PTEs make collection and open fail closed and the node table
then grows without recycling unswept IDs), and a huge-page madvise on an arena VMA (the module re-checks
`VM_NOHUGEPAGE` on the faulting VMA when the kernel has THP, which neither platform kernel does).

## Process creation

The refusal is the libc's, not the module's. `__clone` answers `-ENOSYS` for flags that are not a full
thread, and the delegate answers `-ENOSYS` for every syscall it does not serve, so `fork`, `vfork`, raw
`clone` with fork or vfork flags and `clone3` fail before anything is built; `pthread-contract.c` tests all of
them, in both modes, with the thread tests running afterwards. `execve` and `posix_spawn` are delegated: a
Capstone image replaces the whole launcher process (a new mm, so no tag outlives the old namespace), a native
program is refused with the caller intact, and a spawn is forked by the spawner helper that the launcher starts
before it grants any memory, so a child never shares a tagged page.

A native fork of the launcher itself is not refused by the module or the kernel. The child shares the launcher's
tagged anonymous pages copy-on-write; `unpinned_hold_mm` meets a tagged page with `mapcount != 1` and returns
`-EOPNOTSUPP`, so collection and namespace open fail closed until the child exits or execs. That is an
availability limit, not a safety one. Only the torture stimulus (`CAPSTONE_VM_TORTURE`) exercises it.

## Why it is not the default, and what is missing

1. It is a platform, not a flag: the patched Linux, the model with `x-capstone-raw-moves=true`, the
   `capstone,unpinned-v1` device-tree node and `x-capstone-test-csrs=false`. The module refuses `unpinned=1`
   on a stock kernel and never falls back by itself; a default needs detection with a clean fallback to the
   pinned path, or every platform image carrying all of it.
2. No RTL. At most this could be the default on the model; silicon stays pinned.
3. Memcached fixture 24 fails in the dev head, in OFF and in ON alike, and which of its two checks
   (`fx24_prediction` when the header-only condition is created, `fx24_trigger` when it is not) names the
   failure depends on where the allocator places things: one build and platform gave both outcomes across
   repeated runs of the memcached gate. `gates/compare.py` folds that pair into one placement-dependent
   entry; the fixture itself still needs fixing.
4. Reclaim and swap pressure, page migration and compaction, shared tagged backing (refused), a native fork of
   the launcher, and a whole-system leak bound are untested or fail closed with unmeasured availability cost.
5. Device-side rule 1 is a model property the OS cannot restore, and its arms are not in this tree.
6. Single hart, no KSM, no huge pages: the module refuses `num_online_cpus() != 1`; THP is refused at `add`.
