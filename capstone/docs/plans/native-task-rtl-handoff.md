# Copyable native Linux tasks: RTL implementation handoff

Developer entry point for **`native/linux-task-integration`**, rebased on
`origin/dev` at `c9611344eeb75ed628d488202a9394a6e44f3db5` on 2026-11-08.
The rebased runtime source snapshot is `a62d1dfe59c3ee1a5786230db30231e3f2abfc46`.
Later handoff and qualification commits do not change that runtime snapshot.
This branch is ready for RTL implementation assessment, not FPGA acceptance.

## 1. Checkout, review and qualification boundaries

```sh
git clone --branch native/linux-task-integration --single-branch \
  https://github.com/project-starch/llvm-capstone.git llvm-capstone-native
cd llvm-capstone-native
git submodule update --init capstone/capstone-qemu
git -C capstone/capstone-qemu rev-parse HEAD
# Expected: dd041bac5ad9c145c3872a50804ca714a26b1c88
```

Read sections 2–3 for the software/hardware boundary and required semantics.
Use the [native Linux task contract](../design/virtual-capstone/native-linux-tasks.md)
and [runtime/build entry](../../runtime/virtual/README.md) for integration.
Other build dependencies are documented there; the checkout above is the
source-review checkout, not a complete compiler/Linux/firmware toolchain.

| Item | Frozen revision |
| --- | --- |
| Rebase/comparison base | `c9611344eeb75ed628d488202a9394a6e44f3db5` |
| Rebased runtime source | `a62d1dfe59c3ee1a5786230db30231e3f2abfc46` |
| Preserved pre-rebase source | `8897411a0f03` on `archive/native-task-pre-rebase-20261108` |
| QEMU at comparison base | `0542c5556c2daddf71795f3d48eec1fb8ae137b3` |
| Required QEMU | `dd041bac5ad9c145c3872a50804ca714a26b1c88` |

The rebased source contains 78 linear commits above the base. The
[current inventory](../../runtime/virtual/tests/native-task/rtl-handoff-current-inventory.json)
groups the complete branch delta and lists the five QEMU commits. There are no
LLVM compiler changes relative to this new base. Compiler changes inherited
from `dev` are present; the new SHRINKTO-global optimization remains default off.
Existing benchmark/compiler binaries retain their separately recorded hashes.

The three rebase conflicts were resolved in the adapter: the old linear remap
path stays removed, native checked-copy fields are retained, and the old ioctl
mapping-pool hooks are removed with the retired ioctl adapter. The upstream
`arena_at` inlining annotation is retained. Upstream legacy launcher pool sources
remain historical; `CAPSTONE_VM_MAPCACHE` is not a native-task feature. No new
native mapping-cache semantics or ISA feature were introduced by this rebase.

Review against the actual base, including the QEMU gitlink:

```sh
git diff --stat --ignore-submodules=none c9611344eeb7 HEAD
git log --oneline c9611344eeb7..HEAD
git -C capstone/capstone-qemu log --oneline 0542c5556c2d..dd041bac5ad9
```

Qualification is snapshot-specific:

- [Rebase regression record](../../runtime/virtual/tests/native-task/rebase-qualification.json):
  the rebuilt runtime and bounded integration/GCC checks on this source snapshot.
- [Rebased full SPEC C-port record](../../runtime/virtual/tests/native-task/spec-rebase-qualification.json):
  all ten existing entries pass on this rebased kernel: 22 original invocations,
  four guest image validators and all 25 official output comparisons. The campaign
  took 236.9 host minutes; this is functional qualification, not a speed result.
  Existing benchmark binaries and inputs were unchanged. Unported C++/Fortran
  entries are outside this gate. The [older record](../../runtime/virtual/tests/native-task/spec-final-qualification.json)
  remains historical evidence for its pre-cleanup image.
- [Read-fault experiment](../../runtime/virtual/tests/native-task/access-fault-experiment.json):
  default off; it regressed GCC and read-then-write workloads. Its measurements,
  like earlier index/array cleanup measurements, describe the pre-rebase images.
- Existing JSON records and source archives keep their original hashes. The
  archive branch preserves those source commits; do not relabel old results as
  rebase qualification. Historical manifests are not current checkout instructions.

The hardware model and interfaces are unchanged by the rebase and recent
kernel-memory cleanups. RTL work still needs explicit lifetime collection,
preemption and raw-cargo interrupt decisions described below. No synthesis,
FPGA performance or multi-hart qualification is claimed.

## 2. Changes grouped by responsibility

| Layer | Change | Start reading |
| --- | --- | --- |
| Processor model | Strict copyable profile, ordinary interrupt routing, checked access, immutable request snapshots, protected export/import and discovery | QEMU `target/riscv/capstone_supervisor.c`, `capstone_table.c`, `insn32.decode`; contracts below |
| Linux memory integration | Tag-aware copy/reuse and unpinned application backing; existing inherited facilities are still required | [page lifecycle](../../runtime/virtual/unpinned/README.md), `linux-6.1-unpinned.patch` |
| Linux architecture/task | Task/mm-owned context; exec admission, accounting, return work, cleanup; no context-owning userspace FD | [task transition contract](../design/virtual-capstone/native-linux-tasks.md#linux-61-first-backend-transition-contract), `kernel-native-task.patch` |
| Kernel runtime | Capability-aware copy protocol, syscall dispatch, descriptor/path grants, ordinary Linux file/status/vector services | `runtime/virtual/module/checked-copy-dom.h`, `native-syscall-dom.h`, `resource-dom.h`, `native-gcc-dom.h`, `native-files-dom.h` |
| Linux helper patches | Held-file and kernel-buffer helpers preserving Linux policy, POSIX locks and errors | `kernel-native-mprotect.patch`, `kernel-native-gcc.patch`, `kernel-native-files.patch` |
| Process, directory and time services | Confined cwd, pipes, default/ignore signals, directory operations and kernel-buffer timers/select | `native-process-dom.h`, `native-directory-dom.h`, `native-time-dom.h`; `kernel-native-process.patch`, `kernel-native-directory.patch`, `kernel-native-time.patch` |
| Native subprocesses | Authenticated child bootstrap, pinned-image exec, ordered file actions and admitted-child wait | `kernel-native-spawn.patch`, `native-spawn-dom.h`, `runtime/common/spawn.c`, `runtime/include/capstone/spawn.h`, inherited `posix_spawn_delegate.c` |
| Application ABI/libc | Launch-v3 feature negotiation; opt-in direct scalar fd and seek/stat services, POSIX locks and read/write/pread/pwrite up to 64 KiB; scalar canonicalization | `runtime/common/launch.c`, `ports/musl-capstone/runtime/native-syscall.S`, `delegate.c` |
| Application portability | Native hush shell and versioned Perl pointer-provenance repair; no new hardware instructions | [hush](../../ports/hush/app/README.md), [Perl provenance](../../ports/perl/provenance/README.md) |
| Evidence | QEMU boundary tests, hostile inputs, lifecycle/race tests, SPEC and SQLite results | [native task qualification](../../runtime/virtual/tests/native-task/README.md), [copyable qualification](../../runtime/virtual/tests/copyable-only/README.md) |

The large source diff includes imported dispatcher/test foundations and raw
measurement records. Its line count is not an estimate of RTL work. Native
task integration and the latest file services add no new processor opcode.

The machine-readable inventory groups every changed path and separately lists
all QEMU implementation and verification changes. The native-only adapter has
no misc device, device ioctl handlers, control FD or scalar trampoline. Build it
in-tree with `CONFIG_RISCV_CAPSTONE_TASK=y`; old external-module recipes are
historical. Native exchange/direct ABIs and shared capability checks remain.

## 3. Hardware contract to implement or demonstrate already present

### Copyable capability profile — changed on this branch

Admitted capabilities are NONLIN. Preserve tags, compressed representable bounds,
permissions, PCC, lifetime namespaces and CDERIVE/CREVOKE. Narrowing a capability
does not create exclusive ownership. CSMINT keeps its six-word v7 descriptor and
preflight-before-publication rules, but rejects linear mint requests. Entry and
import reject tagged non-NONLIN state.

The selected profile removes fifteen old ownership/control/debug encodings at
all privilege levels, including DELIN and legacy CALL/RETURN/CAPENTER. Legacy
REVOKE is distinct from the retained lifetime operation CREVOKE. This is a strict
profile, not a runtime compatibility switch.

Contracts and executable gates: [copyable plan](2026-11-06-copyable-only-prototype.md),
[QEMU decoder/mint tests](../../capstone-qemu/tests/copyable-only/README.md).

### Checked supervisor access — new hardware interface

CSCHECKR and CSCHECKW use opcode `0x5b`, funct3 `1`, funct7 `0x25` and `0x26`.
A paused frame and aligned 48-byte physical descriptor identify request epoch,
register slot, unsigned offset, length, output address and reserved zero.

Authorization uses the immutable escaped capability, not a mutable reply slot.
It checks tag/type, permissions, bounds, current lifetime validity, caller
privilege, translation identity, lifetime namespace and escape epoch. A fresh
nonwrapping epoch is published at frame+656, classifications at +664. Those
classifications are not sufficient authorization. Enabled operations return
success or documented failure causes; unsupported operations fault precisely.
Generation and VALID/FREE must participate in one atomic final lifetime decision.

The checker does not itself hold Linux mappings or prevent later revocation.
The kernel's mapping holds, exclusion window, fault/revocation rechecks and final
drain remain part of correctness. Do not replace them with one unchecked access
after a successful hardware check.

Authoritative layout and tests: [checked-access model contract](../../capstone-qemu/tests/capstone-checked-copy/README.md),
[kernel copy protocol](../../runtime/virtual/tests/checked-copy/README.md).

### Protected execution and context transport — extended interface

CSRUNV uses opcode `0x5b`, funct3 `1`, funct7 `0x24`:

| Action | Required behavior |
| ---: | --- |
| 0 | Enter/resume; a retried fault does not skip its instruction |
| 1 | Discard context |
| 2 | Complete service, consume reply, advance service PC exactly once |
| 3 | Collect the paged lifetime namespace; see implementation decision below |
| 4 | Export a paused context |
| 5 | Import the protected context image |

Actions 0–3 are inherited model requirements; actions 4/5 and the checked
snapshot semantics are part of this branch's QEMU delta. Their presence in
QEMU does not establish that the selected RTL implements them.

The native configuration uses a kernel-owned 4096-byte v6 image. Immutable
escaped GPR slots at `+2400+16*r` are separate from mutable resume/reply slots at
`+64+16*r`. FP and namespace/control state are transported. Successful import
consumes the marker and extension tags; rejected import leaves a usable image.
Trusted kernel rebinding is allowed. This is not anti-replay against the kernel.
**Opaque raw register cargo and RVV are rejected by export**, not transparently
preserved. Broader architecture aspirations must not be read as implemented ABI.

Device MIP is hart state, not context state to restore. Sv39/satp, namespace
changes and TLB invalidation must preserve isolation. Preserve precise service,
fault, page-fault, quantum and node-pressure completion. The current Linux code
also contains explicit fences and immediate srevroot clearing after entry; do
not drop those without a new hardware contract and tests.

One hardware slot is sufficient when export/import is implemented. Multiple
sleeping Linux tasks must not reserve that slot indefinitely. The separate
deterministic race suite uses 32 simulated slots; that is not a minimum FPGA
capacity. See [transport gate](../../capstone-qemu/tests/virtual-capstone-runtime/context-transport.sh)
and `runtime/virtual/module/capstone_vm.c` slot acquisition/eviction.

### Tags and Linux page maintenance — inherited dependencies

CLDRAW/CSTRAW and CPTAG already existed at the common-ancestor QEMU pin. They are
not new instructions introduced by native Linux task entry, but this branch
requires their behavior. Raw transport carries stored bits and an opaque tag
without minting usable authority. CPTAG supplies a conservative per-physical-page
tag summary and verifies emptiness before acknowledgement. Scalar writes clear
tags, and aliases must observe coherent tag publication and summary state.

Raw operations are restricted to ordinary S/M with virtualization off and no
active protected context; protected applications and U/VS cannot execute them.
Linux preserves tags across supported page copies and clears them before frame
reuse. Current admission requires 4-KiB pages, 16-byte granules, one possible CPU,
KSM/THP disabled, and matching discovery. Possibly tagged pages cannot swap.

Contracts: [raw/page boundary](../../runtime/virtual/tests/copyable-only/raw-transport.md),
[raw QEMU instructions](../../capstone-qemu/tests/capstone-raw-moves/README.md).

### Decisions that cannot be hidden in an RTL translation

- **Lifetime table and collection:** the runtime requires paged tables, normally
  30-bit IDs, and action-3 namespace recycling. Older RTL paths documented in
  this tree refuse actions above 2. QEMU's software/model sweep is not a free
  hardware operation; specify its hardware/firmware protocol, ordering and cost.
- **Preemption:** native scheduling currently uses QEMU's 5-ms quantum escape.
  Establish the FPGA timer/escape implementation; direct native timer delivery
  is not already qualified by these runs.
- **Raw cargo during interrupts:** the current copy protocol assumes ordinary
  OpenSBI MTIE/STIP behavior, Sstc disabled and a stable source. An integer-only
  firmware trap save can destroy opaque cargo. This does not cover arbitrary
  machine interrupts, DMA or SMP; define the supported hardware/firmware pair.
- **Scalar operands and bounds:** compare actual selected RTL against virtual
  MOVC and scalar CINCOFFSET/SCC behavior and bounds encoding/writeback. The
  source explicitly records differences from older RTL. Do not assume compiler
  compatibility merely because opcode names agree.
- **Discovery:** advertise the implemented copyable, unpinned, checked-access
  and context-transport versions through the matching platform description.
  Production does not use test CSRs or test-only scalar-S LDC support.

## 4. QEMU acceptance before requesting RTL implementation

The rebased snapshot now passes the full existing ten-entry C-port gate.
[Current qualification](../../runtime/virtual/tests/native-task/spec-rebase-qualification.json)
records every input/platform hash and comparator. The older runs described below
remain historical and are not the evidence used for this new-kernel verdict.

Use the final native kernel, its matching QEMU pin, ordinary OpenSBI and current
launch-v3 SDK. Keep code/input/oracle hashes and failed attempts. Full application
exit plus every official comparator is required; a boot completion marker,
bounded instruction slice or partial output is not a pass.

The current SPEC gate targets all ten entries in the existing C port: eight
application benchmarks plus two specrand entries. It does not claim that the
unported C++/Fortran or speed variants of the whole CPU2017 suite work. Sources
and inputs stay external under scratch storage; existing source adaptations
remain frozen. Test means original full `test` inputs, not train/reference data.

The pre-cleanup kernel `1bd9b8ecf702` independently passes all nine non-Perl entries,
including full x264. Revision 06 of the Perl repair passed focused gates but
failed the complete original workload in `t/op/sort.t` (315/316 files passed).
Revision 07 repairs the freed-scalar metadata lookup and passes the full original
workload and both official comparisons. This is an application-port change,
with no additional kernel or RTL requirement. See
[current non-Perl evidence](../../runtime/virtual/tests/native-task/spec-nonperl-services-qualification.json)
and [Perl port qualification](../../ports/perl/provenance/qualification-07.json).
The [final ten-entry record](../../runtime/virtual/tests/native-task/spec-final-qualification.json)
records 22 benchmark invocations, four guest image validators and all 25 official
comparisons passing. Original inputs and comparators are unchanged.

**Historical snapshot `09b90be908e6`: 9 PASS, 1 FAIL (Perl).** All 22 original test invocations
completed. Of 25 official output comparisons, 24 pass; Perl's test.pl comparison
fails despite exit zero. All four guest image-validation invocations completed, covering one
ImageMagick image and three x264 images. That historical snapshot still had
a named software blocker; the final revision-07 qualification above closes it.

The [machine-readable qualification](../../runtime/virtual/tests/native-task/spec-test-qualification.json)
records platform/SDK/input/object hashes, selected execution evidence and every
comparator result. The [reproduction guide](../../runtime/virtual/tests/native-task/README.md#full-spec-test-input-qualification)
includes the four helpers and their synthetic regression suite. This is
functional qualification; no instruction-count or FPGA speed result is claimed.

### Historical results and diagnosis at `09b90be908e6`

| Existing C-port entry | Historical full test result |
| --- | --- |
| 502.gcc_r | PASS, including generated assembly comparison |
| 519.lbm_r | PASS |
| 538.imagick_r | PASS, including guest image validation |
| 544.nab_r | PASS |
| 997.specrand_fr | PASS |
| 999.specrand_ir | PASS |
| 505.mcf_r | PASS, both output comparators |
| 557.xz_r | PASS, all 12 invocations and comparators |
| 500.perlbench_r | FAIL: makerand passes; test.pl output is incomplete |
| 525.x264_r | PASS, complete 156-frame encode and all three image comparisons |

The [Perl diagnosis](../../runtime/virtual/tests/native-task/spec-perl-blocker.json)
confirmed a missing native `chdir` service (`ENOSYS`). `t/TEST` ignored that
failure, then could not open relative `base/cond.t`. Its outer `test.pl` used
`do 't/TEST'` without checking the exception, so the official invocation exits
zero despite incomplete output. Running `./perlbench_r -I. -I./lib t/TEST`
directly in the staged directory exposes the error and exits 2. This diagnostic
does not change the official workload or its failed qualification status.

The historical native C probe also observed missing pipe/popen, signal disposition
and spawn/wait services. Subsequent commits implement and qualify confined pipes,
default/ignore signals, native subprocesses and hush, as linked above. Signal
refusals alone did not explain the original failure: the successful makerand
case encountered them too. These software repairs add no application-specific
RTL instructions.

## 5. Suggested FPGA implementation/measurement ladder

1. Select and record the actual RTL base and supported board configuration.
   Mark each requirement above present, changed, absent or undecided against
   that base. The superproject's RTL gitlink is not proof of the flashed design.
2. Implement the strict profile and primitive negative tests first. Check exact
   privilege failures, malformed operands, no partial authority publication,
   bounds, lifetime reuse and stale capabilities.
3. Add checked-access and export/import gates, including immutable-request vs
   mutable-reply substitution, namespace/epoch mismatch, rejected import,
   one-slot eviction and sleeping tasks.
4. Qualify Linux page copy/reuse/COW and interrupted raw copies on the selected
   firmware. Boot the native kernel; qualify task accounting, stop/continue/kill,
   blocking I/O, descriptor/path confinement and file-lock interaction with Linux.
5. Run GCC and SQLite correctness first, then the full admitted SPEC test set.
   Keep the same workload and output checks as QEMU.
6. Compare exchange vs direct musl I/O on one bitstream/kernel with matched
   objects and inputs. Measure cycles and retired instructions separately;
   QEMU attempt counts do not predict tag-cache/coherence or memory-bandwidth
   costs. A separate stock-Linux baseline needs its own matched-build contract.

## 6. Reproduction and remaining software scope

Start with [native build/activation](../../runtime/virtual/tests/native-task/README.md#build-and-activation).
The installer stages the runtime into Linux and applies the native task, files,
process, spawn, directory and time patches; prepared Linux also needs the documented unpinned, mprotect and GCC
helper patches. `capstone/tests/capstone-test-env.sh` must be sourced before
build/test commands. Keep licensed benchmark sources and artifacts out of Git.

Perl's full workload additionally needs the exact pinned shell executable grant
`capstone_vm.dom_shell=PATH`, exact device grant `capstone_vm.dom_null=1`, and
confined temporary-directory alias `capstone_vm.dom_local_tmp=1`. These default
off and do not enable broad filesystem grants. See [child-process setup](../../runtime/virtual/tests/native-task/README.md#native-child-processes)
and [private temporary files](../../runtime/virtual/tests/native-task/README.md#private-temporary-files).

The current backend is default-off, root-only and single-hart. Caught signals,
automatic restarts, protected fork/clone, caller-replacing exec, SMP, native audit/seccomp/tracing and
complete arbitrary Linux syscall coverage remain outside the qualified scope.
Unsupported policy configurations are refused. Full native FP/register/raw
platform matrices remain separate gates. Neither SPEC success nor this handoff
waives those limitations or authorizes board flashing.

## 7. Request to the RTL developer

Please assess whether this pinned QEMU/Linux contract can be implemented on the
selected FPGA platform. Return a requirement-by-requirement inventory against
an explicit RTL commit: already implemented, needs modification, missing, or
requires a design decision. In particular, estimate checked access, protected
context export/import, paged lifetime collection and interrupt-safe raw tag
transport separately; include likely storage, critical-path and bandwidth costs.

Propose an implementation order tied to the executable negative tests above,
then synthesis and the application measurement ladder. Identify format/opcode
or firmware changes before implementation. The current QEMU allocations and
frame layout are an experimental platform ABI, not a claim that the permanent
ISA is frozen. An alternative contract needs matching software changes and a
fresh QEMU acceptance record before its FPGA results can be compared.

The SPEC acceptance result must accompany this request. A failing application
must remain a named software blocker rather than becoming an implicit RTL task
or being omitted from the acceptance scope.
