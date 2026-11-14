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

`capstone/kernel/patches/linux-6.1-unpinned.patch` is a complete patch against upstream Linux 6.1
(commit `830b3c68c1fb`). Platform-dev4's Image reports Linux 6.1.0. Its embedded
config has SHA-256
`9c7247993f725f3f4313605a8ec1785177d6fe9906692b3c8f181231f0aa59e1`.
Extract that exact config with `scripts/extract-ikconfig`; do not use a generic
defconfig. KSM and THP are disabled. Swap is configured, but tagged swap is
refused. Keep builds and writable state in a private directory on the build host.

```sh
# In an otherwise unmodified Linux 6.1 source tree:
git apply capstone/kernel/patches/linux-6.1-unpinned.patch
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

`unpinned_fast_sweep=1` is a separate experimental option, default OFF. It
queries each ordinary base page's sticky tag summary under the PTE lock and
takes a reference to a tagged private anonymous page before releasing that
lock. Untagged pages need no GUP or access-bit update. The scan still covers
the entire mm, retains mmap_lock through collection, refuses unsupported
entries and releases every acquired reference on failure. It does not cache
PFNs or skip retired mappings. Tagged compound pages are refused explicitly, and
so is ANY special or device PTE, tagged or not: that is stricter than the old
scanner, which only refused such a page if it was tagged. This option needs its
own qualification before use as a default; `unpinned=1` alone retains the
original GUP scanner. Loading with fast sweep enabled but unpinned mode
disabled returns `EINVAL`. Once per module load the first sweep logs which
scanner ran, and the load banner prints both options, so a guest log shows what
was executed.

The mapcount test exists twice in the fast path: in `next_tagged`, under the PTE
lock and before `get_page`, and in `unpinned_hold_mm`, after the reference is
held (the second also enforces `VM_SHARED`, and is the only one the old
scanner has). Each is sufficient for the fast path; removing both is the
failing mutant. `run-sweep.py` runs the single-site mutants in both modes to
show exactly that: removing only one passes with the fast scanner, removing
only `unpinned_hold_mm`'s fails with the old scanner.

Lock hold. `next_tagged` holds a PTE spinlock (preemption off) across up to 512
PTE reads and one tag-summary query per present non-zero page. The old scanner
took the lock once per resident page for one PTE read. On the QEMU model a query
is a host table lookup. On silicon each query may be an uncached tag-controller
access, which would make the hold up to 512 of them per table; this has not been
measured. Measure it before any default-on decision.

`python3 sweep-double.py` checks the abstract sweep/hold/recycle contract and
requires deliberately incomplete, tag-omitting, reference-omitting and
foreign-namespace-clearing variants to fail. The mutants are branches inside the
model: it is an executable interface double, not evidence about the kernel
module, a Linux race test or a substitute for the fixtures below.

`run-sweep-diff.py` is the completeness check. `sweep-span.c` (a native client
for the namespace-probe fixture) seeds tags in four PTE tables of one mapping,
the first holding only an untagged resident page, and in two VMAs separated by a
third; the namespace scrub must clear them all. Two mutants, each applied only
to the fast scanner, stop the scan early: one at the end of the first PTE table
without a tagged page, one after the first VMA that yielded a page. Both must
fail it (tags survive). A `CV_TEST_HOOKS=1` module also takes
`unpinned_sweep_diff=1`: every sweep then runs the other scanner on the same mm
under the same locks and compares the two held-page lists (count, position by
position, and order-independent digests); the counters are the read-only
parameters `diff_runs`, `diff_active_pages`, `diff_other_pages`, `diff_mismatch`,
`diff_digest_mismatch`, `diff_other_errors`. The runner requires zero mismatches
on the unmodified module in both directions and a nonzero count on both
mutants in both directions. The same test module and parameters can be given
to any guest workload; the counters are then read from sysfs or the log.

`run-sweep.py` consumes the namespace probe fixture and a freshly built
`node-contract.c` image. Build `sweep-shared.c` as a static native client with
the fixture's `namespace-probe.h` in the include path. Pass its directory as
`--probes` together with `namespace-check` and `namespace-probe.ko`. The runner
requires a normal module and five separately built broken modules: `--mutant`
replaces `if (!unpinned_page_summary(page)) continue;` in `next_tagged` with
`if (unpinned_fast_sweep) continue;`; `--shared-mutant` replaces the two sweep
`page_mapcount(page) != 1` predicates (in `next_tagged` and `unpinned_hold_mm`,
not the pinned-mode ones) with `false`; `--shared-next-mutant` and
`--shared-hold-mutant` replace only the first and only the second;
`--config-mutant` removes
`if (unpinned_fast_sweep && !unpinned) return -EINVAL;` from initialization.
These mutations never belong in a production binary. All modules must match
the selected kernel.

The queued run verifies multi-page namespace scrubbing, byte preservation,
refusal to scrub a tagged page shared with a native child, recovery after
the child exits, two 70,000-live-node populations, stale-access denial and
namespace cleanup, and rejection of inconsistent module options. Both page
mutants must fail the same native client that
passes on the normal module. Run with raw moves enabled, test CSRs disabled,
and 32 QEMU supervisor slots; the runner sets and reads back the corresponding
module options. Keep the normal nine runtime gates as a separate comparison.

`run-sweep-refs.py` adds a real-page check without production counters. Build
`sweep-ref-probe.c` as a test-only module and `sweep-refs.c` as a static native
client with this directory and the namespace fixture header in its include
path. From this directory, with a matching configured kernel and native Linux
cross-compiler:

```sh
mkdir -p "$OUT"
cp sweep-ref-probe.c sweep-ref-probe.h "$OUT/"
printf 'obj-m += sweep-ref-probe.o\n' > "$OUT/Makefile"
make -C "$KERNEL_BUILD" M="$(realpath "$OUT")" ARCH=riscv \
  CROSS_COMPILE="$CROSS_COMPILE" modules
"${CROSS_COMPILE}gcc" -static -O1 -I. -I"$NS_FIXTURE" sweep-refs.c -o "$OUT/sweep-refs"
```

The observer takes and releases one temporary page reference under the PTE lock; the client
compares counts including that same temporary reference before and after each
operation. Before sampling, it calls `MADV_COLD` on the untouched middle page:
the matching Linux 6.1 path drains local LRU batches, and that empty range cannot
queue a new deactivation reference. `mincore` verifies the range stays absent.
The client also checks stable PFNs, mapcounts and absolute counts (one mapping plus
one observer, or two mappings plus one observer). The observer avoids GUP's access/activation side effects and cross-checks its
PFN against the namespace probe's independent GUP lookup. Together these checks
prevent an LRU batch drain hiding a leak. It tests a successful scrub through a `PROT_NONE` PTE and reference cleanup
when an exclusive tagged page precedes a shared tagged page in the VMA.

Its `--stage` directory contains the normal module as `good.ko`, the namespace
probe, `sweep-ref-probe.ko`, `sweep-refs`, and two broken modules. `leak.ko`
replaces the `put_page` in `unpinned_release_mm` with a no-op; `omit.ko` is the
tag-omission mutant above. The same client must reject both: a leak changes the
real page's reference count on success and refusal, while omission leaves the
inaccessible page tagged and incorrectly accepts the shared page. The runner
requires those specific failures, not just an arbitrary nonzero exit. This
checks reference balance on an isolated single-hart guest; it does not replace
migration or sibling-thread stress.

Before publishing a fresh namespace in an mm, the same scan removes old tags
left by a prior close/open. A raw load followed by a scalar store rewrites the
original low word, retaining bytes while clearing the storage tag under rule 1.
This avoids scalar-load projection. The global context lock excludes a second
active namespace in the same mm.

These paths still require qualification: "Tests and limits" below lists what each gate and
control establishes, and what is not covered. No result is inferred from successful
compilation.

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
  output with no mode file, a gate with no mode line, fewer or more lines than module loads (`MODE_LOADS`),
  a line that is not the requested mode, a `memory_mode_verified` that failed or is absent (whether or not
  both sides fail alike), or a SUMMARY.txt carrying `MODE_CHECK_FAILED` / `OPTIONAL_GATE_FAILED`.
  `--control-mode` / `--arm-mode` pin the pair (an ON arm is compared with an OFF control). The nine
  ordinary gates may end non-zero (their FAIL sets are compared); the optional ones may not.
- Optional gates for `on`: `unpinned-mm` (the isolated swap-hook and namespace fixtures, needs
  `UNPINNED_PROBES`) and `unpinned-mutants` (broken modules, below). When selected they must succeed (otherwise the arm writes
  `OPTIONAL_GATE_FAILED`, exits non-zero and `compare.py` refuses the output) and their own mode witnesses
  are checked (the seven `MM_ARM` lines; every mutant run's memory mode). The arm also checks that the
  production module carries none of the test-hook strings.
- ON has its own resolution oracles, because the pinned page counters read zero by construction. They
  read what Linux says: the kernel's FOLL_PIN counters (step-hooks and pin-ahead require them not to
  move during the unpinned run, after a launch in the same guest has shown that they do move: the
  pthread contract's supervisor frames); the launcher's VmRSS/VmHWM (`CAPSTONE_VM_RSS=1`) for pages given
  back and for the 64 MiB the `vm` launch never touches; per-fault event records against the public
  fault counter; window and event pages that never meet. Every OFF oracle is unchanged.

### Optional direct fault path: `unpinned_fast_fault` (EXPERIMENTAL, default OFF)

`unpinned_fast_fault=1` (module parameter, read-only after load, valid only with `unpinned=1`: the load is
refused otherwise) resolves an unpinned anonymous fault with `handle_mm_fault` on the VMA `resolve()` already
found under `mmap_read_lock`, instead of `fixup_user_fault`, which looks the VMA up a second time. It applies to
the demand fault and to every fault-ahead page of `unpinned_resolve` (an address outside that VMA, as the broken
`test_skip_vma_clip` module produces, takes the old call). File-backed, shared, IO, PFNMAP, MIXEDMAP and
hugetlb VMAs, and the image text path, keep `fixup_user_fault`; the permission test of `fixup_user_fault` is
replicated for the direct call. Stream training, window clipping, stop checks, counters and diagnostics are
untouched, and with the parameter at its default 0 every fault takes dev's `fixup_user_fault` call.

What a log says about it: the loaded banner (KERN_EMERG) ends in `unpinned=<0|1>, unpinned_fast_fault=<0|1>`; the
first direct fault of a load logs `capstone_vm: unpinned fault path: direct handle_mm_fault`; the read-only
parameter `unpinned_fast_faults` counts the faults the direct path resolved (the hooks module's unload report
carries it as `unpinned_fast_faults=`). `gates/run-arm.sh` takes `FAST_FAULT=on|off` (default off, needs `--mode on`), every
runner passes the parameter and logs `<GATE>_FAST_FAULT:Y|N|A` after each insmod, and `compare.py` refuses an
output whose lines are not the recorded request (`--control-fast` / `--arm-fast` pin a pair).

Negative control: the CV_TEST_HOOKS build has `test_fast_noflags`, which drops `FAULT_FLAG_WRITE` in the direct path
only. `run-fast-control.py` runs step-hooks `base1` and `adapt` with it three ways (fast on and unbroken, fast on
and broken, fast off and broken) and requires the exit checks of all four runs to flip on the broken direct path
and the fast-off broken module to equal the unbroken one. It is not part of `run-arm.sh`.

Not covered: no gate scenario separates the replicated permission test from the caller's own VMA test, since
the window is clipped to the VMA that already passed it; file and text VMAs are not measured; and what the path
saves is instruction attempts in the QEMU model, not cycles on any core.

Negative controls are broken modules, not edited transcripts. A CV_TEST_HOOKS module has parameters
that each remove one guard (`test_skip_vma_clip`, `test_skip_arena_clip`, `test_skip_stop_check`,
`test_skip_resident_stop`, `test_skip_namespace_scrub`, `test_skip_scrub_after_pages`,
`test_skip_scrub_after_granules`). `run-mutants.py` requires a named check to pass on the unbroken module
and fail on the broken one for the VMA clip, the stop check and the stop at the first resident PTE, and only
when both are complete runs of the selected configurations (completed, expected runs with their exit checks,
verified mode, the observations the check reads); `test-mutants.py` plants each way a run can be incomplete,
among them the evaluator's output on an empty transcript. The
arena clip control is NOT achieved: it is not isolated by any scenario a gate can build (the VMA clip or a resident guard page bounds
the window at the same page), so `test_skip_arena_clip` exists but no check can see it. `run-mm.py` does the same for the scrub with a fixture that seeds 21
granules (offsets 0, 16, 2048, 4080) on 10 pages of four mappings. `test-oracles.py` additionally
corrupts real ON transcripts and requires each corruption to break the named check (it proves the parsers
reject contradictory lines, nothing about a module).
Repeated first faults at one address: the no-change bound (`fault_repeat_max`, 64) counts a fault only against the
previous fault in the SAME mapping (the arena id, which is never reused). A program that maps a page, touches it and
unmaps it again and again gets the same address back; each round's first fault has the same address and cause in a new
mapping and is not a resolution that changed nothing. Counted by (address, cause) alone, as it was, the 65th round
ended the domain with a segmentation fault: mallocng does this when its only group is freed and the next allocation
opens one, so the runtime gate's `sparse-churn` and `sparse-stale` cases died in ON mode (and nine more of its checks with
them) once mallocng replaced the registry heap. Negative control: `run-repeat-control.py` boots the contract image's
`map-churn` mode (300 rounds of map, touch, unmap, reusing the address in more rounds than the bound) with the hooks
module twice and requires exit 0 and no refusal on the unbroken module, and exit 139 with the `unpinned-loop` refusal
on `test_skip_repeat_arena=1`, which restores the old key. It is not part of `run-arm.sh`.
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
   entry only when the case's job, output and fault files show the signature of the failed check (not
   created: `header-only=0`, or the earlier bail-out `ia not linked`, then the known no-op mark `18e0018` and exit; created: `header-only=1`, then a
   signal at a cause-24 fault at address 0); any other single failure makes the gate INCOMPLETE. The
   fixture itself still needs fixing.
4. Reclaim and swap pressure, page migration and compaction, shared tagged backing (refused), a native fork of
   the launcher, and a whole-system leak bound are untested or fail closed with unmeasured availability cost.
5. Device-side rule 1 is a model property the OS cannot restore, and its arms are not in this tree.
6. Single hart, no KSM, no huge pages: the module refuses `num_online_cpus() != 1`; THP is refused at `add`.
