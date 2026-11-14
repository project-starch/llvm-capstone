## Trap-route mapping blocks, 14 November 2026

The step-6 fragment `capstone/kernel/patches/series.s6-mappings` adds depth-1
mapping checks, partial unmap state, last-page CREVOKE, refill and Linux mapping
hooks. Its host state test passes and its missing-depth-check control fails.
The Linux 6.1 Image builds on p13 without warnings. Trap-mode mapping execution
still requires the handle and tagged return-slot streams.
[Design, interfaces and limits](../design/sublet-mappings.md).

## Trap thread build slice, 14 November 2026

On `trap/s8-threads`, the libc clone and futex paths now preserve references.
All 1,349 libc files compile on p13. The two-thread TLS, mutex and join smoke
links and passes disassembly checks. The host binder test passes, and its
incorrect requeue pointer-mask control fails as required. Kernel fragment
`series.s8-threads` adds shared-mm lifetime ownership, per-thread register
slots, reference clear-child-tid and reference futex accesses. The full Linux
Image builds against snapshots of steps 3 and 5. Guest execution
and live-waiter wake counting remain outstanding.
[Integration contract and tests](../../kernel/tests/threads/README.md).

## Trap libc integrated execution, 14 November 2026

On `trap/s5-libc-int`, the setjmp and signal-registration changes are integrated
onto `2443072aa6b`. The exec stack contract and thread runtime are preserved.
All 1,349 musl files rebuilt on p13. The five smoke images linked and passed
disassembly checks, including the clone reference checks. With the integrated
kernel, `setjmp-smoke.dom` preserves the reference tag across longjmp and
siglongjmp and exits 0. `signal-smoke.dom` registers SIGINT, changes its mask,
checks the alternate stack and exits 0. `smoke.dom` writes `ok` and exits 7.
Caught signal delivery remains deferred. The alternate-stack reference is
thread-local to coexist with the integrated thread runtime.
[Build, execution and controls](../../ports/musl-capstone/tests/trap/README.md).

## Rebased native kernel: full existing SPEC C-port gate passes

All ten existing C-port entries pass on branch source `04350015dcad`, runtime
snapshot `a62d1dfe59c3`, kernel SHA-256 `b7de27fb28bc0807053ae3e1c4bcd058e79f601d255a0dcaaa6ce70d46a3815f`.
The full original test inputs complete 22 benchmark invocations, four guest
image validators and all 25 official output comparisons. The sequential campaign
took 236.9 host minutes, including x264 at 133.0 minutes and Perl at 74.8 minutes.
These times are diagnostic, not a hardware or branch performance comparison.

This closes the full-suite software regression gap for the rebased kernel using
the existing frozen binaries. Perl retains its qualified provenance-v2 revision07
port and confined shell/null/tmp grants. Unported C++/Fortran entries, newly built
compiler outputs, multi-hart execution and RTL/FPGA remain outside this result.
[Complete qualification and raw evidence](../../runtime/virtual/tests/native-task/spec-rebase-qualification.json).

## Native RTL handoff rebased onto current dev

`native/linux-task-integration` is rebased onto `c9611344eeb7`, with runtime
snapshot `a62d1dfe59c3`. The preserved pre-rebase branch is
`archive/native-task-pre-rebase-20261108` at `8897411a0f03`. Older qualification
records continue to name their original commits and binaries; their numbers
are not measurements of the rebased image.

The old linear ioctl mapping-pool additions on dev do not restore the removed
adapter entry path. Native checked-copy state is retained; the upstream
`arena_at` inlining annotation is the only adapter source difference from the
pre-rebase runtime. Linux patches and QEMU pin are unchanged. Compiler changes
from dev are inherited, but existing qualified binaries were not rebuilt with
them; SHRINKTO globals remain default off.

Use the [updated RTL handoff](../plans/native-task-rtl-handoff.md), its inventory,
and the [rebase regression record](../../runtime/virtual/tests/native-task/rebase-qualification.json).
Production and race builds, GCC with official comparison, IO/task, directory,
private-tmp, spawn, four memory configurations and both 41-case race groups
pass. The two final gates took 78.9s and 88.4s; each race group observes all 17
deliberate controls firing. The first attempt failed before guest boot due to a
missing test environment variable and remains in the evidence.
The full ten-entry SPEC rerun is now complete as recorded above. Synthesis, FPGA
execution and multi-hart qualification remain outstanding.

## Anonymous read-fault experiment: keep default off

Runtime `2ae1682c54c9` adds an opt-in actual-read policy for lazy anonymous RW
arenas. Eager/bootstrap, file mappings, store/execute handling and checked-copy
policy stay unchanged. Astra xhigh reviewed the scope and test oracles.

Fresh 1MiB read-only traffic saves 41.14% of whole-invocation instruction attempts
and 1,016KiB of peak RSS without fault-ahead. Read-then-write instead costs 28.98%
more; with adaptive ahead64 it costs 2.92x the old policy. Unchanged GCC in fresh
ABBA guests costs 1.617% more without ahead and 1.936% more with ahead64, with no
peak-RSS reduction in either matched cohort. The option remains default off.

Profiling identifies repeated residency scanning: native read-write with the
new policy has 272 flat `next_resident` attempts without ahead versus 1,633,236
with ahead64. Presence is not write readiness: shared zero-page mappings still
need later COW faults. The stream's bounded per-fault scan repeatedly visits
present pages. Access-aware or less repetitive scanning is the next experiment;
no performance or safety claim for such a change exists yet.

All eight GCC outputs match exactly. Eight fixture configurations pass, including
distinct capability stores as the first writes after reads. IO integration and
both 41-case race groups pass with policy readback enabled; each observes all 17
negative controls. Final gates take 174s and 133s. Pilot runs with truncated boot
arguments are excluded; runners now enforce the kernel's 512-byte command-line
capacity and verify guest settings. These are QEMU attempts, not hardware timing;
no new full-SPEC, multi-hart or FPGA claim.
[Artifacts, samples, scope and exclusions](../../runtime/virtual/tests/native-task/access-fault-experiment.json).

## Unpinned arena arrays removed and qualified

Unpinned grants no longer allocate the per-page pin array or duplicate-frame
hash array. Range geometry, IDs, revocation and copy holds are retained; touch
and registration unwind avoid array reads in this mode. All other consumers
remain behind pinned branches. Runtime source: `205b53613621`.

Against the already-qualified index-removal kernel, unchanged GCC/musl binaries
in baseline/candidate/candidate/baseline fresh guests reduce mean total attempts
from 50,348,695.5 to 49,954,678.5 (**0.783%**) and kernel attempts from
26,390,323.5 to 25,996,306.0 (**1.493%**). Program counts differ by only 0.5
attempt on average. Flat savings are concentrated in kernel allocation/free,
zeroing and frame-array sizing. These are incremental measurements, not an
additive estimate of cumulative speedup or hardware time.

Production and race kernel builds, four byte-exact GCC outputs, IO/authority/
entry/task integration, and both 41-case race groups pass within **156.6s**.
Each race group observes all 17 deliberate controls firing, including stale
copies and missing holds. Astra xhigh reviewed all consumers and the immutable
memory-mode contract. The occupied pre-start register-slot unwind was reviewed
statically, not dynamically injected. No new full-SPEC/multi-hart/FPGA claim.
[Exact artifacts, samples and limits](../../runtime/virtual/tests/native-task/arena-arrays-qualification.json).
The actual-access fault-policy experiment above is default off; baseline read
faults and fault-ahead behavior remain unchanged.

## Native arena index removal qualified

The production runtime no longer maintains the unread arena-ID xarray.
Monotonic registration IDs, the address tree, grant retirement and checked-copy
holds are unchanged. The race fixture finds its newly created arena through a
test-only scan of the surviving tree. Runtime source: `af6893dc17a8`.

With unchanged GCC/musl/application binaries, baseline/candidate/candidate/baseline
fresh guests reduce mean total instruction attempts from 50,728,469.5 to
50,344,376.5 (**0.757%**) and kernel attempts from 26,770,595.5 to
26,386,502.5 (**1.435%**). Program attempts are identical at 23,946,835.
The flat reductions primarily involve xarray operations and their locking.
This is a general kernel-runtime saving, not an allocator or GCC modification.

Both kernel builds, four byte-exact GCC runs, and two 41-scenario copy/unmap/reuse
race groups pass in one **134.2-second** gate. Each race group also observes all
17 deliberate negative controls firing. Astra xhigh independently reviewed the
source diff. No new full-SPEC, multi-hart or FPGA qualification is claimed.
The page/frame arrays and fault-access policy remain separate, unimplemented
experiments. [Hashes, samples, checks and limitations](../../runtime/virtual/tests/native-task/arena-index-qualification.json).

## GCC memory amplification and fault-path controls

Follow-up diagnostics tie part of the larger memory demand to concrete GCC ABI
layouts. During compilation `_cpp_clean_line` makes 195 fresh line-note array
allocations in both arms: 200 entries require 3,200 bytes on plain RV64 and
6,400 bytes on Capstone. Another 65 pointer tables request 256 x 8 versus
256 x 16 bytes. Those requests cross the Capstone allocator's 4KiB aligned
allocation threshold. This is actual argument/caller evidence, not an estimate
from binary sizes. It does not yet assign a precise number of extra mappings
to those individual allocation sites.

The mapping trace confirms 118 versus 281 calls, requesting cumulatively
1,912,832 versus 5,140,480 page-rounded bytes during compilation/output (not RSS).
Both mapping and caller diagnostics produce exact reference output. The
allocator's group layout, representability alignment and size classes amplify
larger application objects into different mapping behavior.

Independent existing-option experiments use the same binaries and four fresh
guests per configuration, plus a new same-harness baseline. Disabling adaptive
fault-ahead reduces native total attempts by 0.269% (kernel 0.513%); demand
resolutions rise from roughly 1,227 to 2,002. Reusing the validated VMA via
`unpinned_fast_fault=1` reduces total by 0.984% (kernel 1.866%). These are separate
experiments, not additive savings. All twelve performance guests and four
argument-trace guests match the official output. Every gate finishes below
129 seconds, including lock waits. No runtime default is changed and these
GCC diagnostics are not safety/full-SPEC qualification of either option.

[Exact input hashes, caller arguments, histograms and controls](../../runtime/virtual/tests/native-task/gcc-memory-mechanisms.json).

## Current GCC overhead against ordinary RV64

Matched current Linux/QEMU, plain/native/native/plain fresh guests, all native
routes enabled in both arms: 30.166M plain versus 50.730M Capstone instruction
attempts (+68.17%). Four exact reference matches; authenticated phase boundaries,
phase conservation and a corrupt-fingerprint rejection control pass in 69.4s.
Kernel accounts for 14.698M of the 20.563M extra attempts (71.47%); program and
runtime account for 5.859M. Startup/initialization adds 7.328M, compilation/output
10.290M, finalization/exit 2.946M.

During compilation, first-instruction attempts at Linux do_mmap rise from 118
to 281 and handle_mm_fault from 418 to 959. Capstone heap mappings also number
281, with 239 heap unmaps. This establishes amplified memory-management traffic;
flat deltas include memset, mapping-tree lookup, domain stepping and faults.
At exit, arch_free_page alone adds 428,228 flat attempts. Whole-invocation sysfs
counters observe 1,396 cleared tagged pages versus zero, and about 2,591 freed
pages versus 1,307 (these counters also include measurement shell activity).
The old pinned-page retirement sweep remains zero. No heap-remapping entry was
observed; map-copy-unmap is not an established cause in this workload.

Source mechanisms include mallocng capability layout/alignment, page-owner index
mappings, per-arena bookkeeping arrays allocated even unpinned, and domain fault
handling including 855 fault-ahead pages. Their individual causal contributions
are not yet isolated. Plain and Capstone compiler/libc provenance differs, so
this is an end-to-end comparison on the same patched kernel, not isolated ISA
cost or a stock-kernel result. Counts are attempts, not hardware cycles.

[Inputs, phase means, symbol deltas, counters and limitations](../../runtime/virtual/tests/native-task/gcc-current-plain-attribution.json).
The reproducible runner is `measure-gcc-phases.py` beside that record; raw evidence
is archived under `/tmp/capstone/native-task/gcc-attribution-01/evidence`.

## Current GCC native-route comparison

SPEC 502.gcc_r test/t1.c with `-O3 -finline-limit=50000` now has a matched
current-kernel comparison. The original GCC object hashes are unchanged; one
binary relinked with the native-clock SDK runs with all native feature flags
off/on (masks 0/479), in A-B-B-A order. Two separate guests collect routes.
All six runs match the official assembly reference byte for byte. Relink,
measurement and diagnostics pass in 102.7 seconds.

| Native routes | Total instruction attempts | Kernel attempts |
| --- | ---: | ---: |
| Off | 50,944,339.5 | 26,969,674.0 |
| On | 50,827,707.0 | 26,869,673.5 |

The mean reduction is **0.229% total, 0.371% kernel**: a small
incremental gain, not measured elapsed-time speedup. There are 31 service
completions in each diagnostic run; 11 move to registers and 20 remain exchange.
This isolates current routing options, not cumulative branch-vs-dev changes
or overhead against plain Linux. The kernel remains the native-clock baseline.
[Exact inputs, routes and profiles](../../runtime/virtual/tests/native-task/gcc-current-routes-comparison.json).

## Native basic clock transport qualified

Feature 256 routes clock_gettime through the existing generic checked register
dispatcher. Only CLOCK_REALTIME and CLOCK_MONOTONIC are supported; provider semantics,
launch tick rate and owner/epoch/output-authority checks are unchanged. No
clock-specific fast handler was added.
Every admitted direct operation completes once; errors never trigger replay.
Enable `capstone_vm.native_clock=1` together with `native_io=1` and a rebuilt
SDK/application. The new feature defaults off; old SDKs disable direct routing
on unknown feature bits. No ISA change is required.

Qualification is restricted to the initial time namespace: the inherited provider
lacks the monotonic namespace offset. Output authority is checked before clock ID,
so a bad ID combined with bad output yields EFAULT. These differences from Linux
are not changed by this transport migration. Actual copy faults need not be atomic.

SQLite moves 65 calls from exchange to registers. With previous native
features enabled in both arms, two samples per arm in A-B-B-A order reduce
kernel instruction attempts by 0.117% and total attempts by
0.089%. All 32 reference phases match in every run.
The build/file/SQLite/regression/GCC gate passes in 258.0s;
combined-feature IO and reversed measurements pass in 105.0s.
Each gate has one 300-second deadline.
[Exact evidence and limitations](../../runtime/virtual/tests/native-task/native-clock-qualification.json).

Runtime and negative tests were implemented in parallel and independently reviewed.
This is not a new full-SPEC or FPGA qualification. 37 exchange calls remain
in this SQLite census. Next measure pathname byte counts and repeated checked-copy
setup, then prototype a bounded checked-string operation. Clock namespace/error-order
corrections and bounded vectors remain separate work.

## No external Capstone module in the native deployment

The native kernel contains the Capstone adapter built in. A QEMU boot audit
confirmed no `capstone_vm.ko*` in the root filesystem or staged resources,
no loaded Capstone module in `/proc/modules`, and no Capstone insmod/modprobe
startup in `/etc/init.d` or `/etc/inittab`. The native task/dispatch parameters
are enabled. The audit passes in 3.4 seconds on the qualified native-path kernel.
The rootfs already lacked this artifact; it was not modified.

The obsolete `build-adapter.sh` external-module/launcher recipe is retired;
its old entry point now directs callers to `install-native-task.sh` and an
in-tree `Image` build. Historical module runners remain for old platforms.
The built-in adapter, capability checks and remaining exchange services are
still required. Module-free deployment does not mean all syscalls are direct
or that all Linux semantics have been qualified.
[Boot evidence](../../runtime/virtual/tests/native-task/module-free-qualification.json).

## Native confined pathname services qualified

Feature 128 routes openat/unlinkat directly. Both transports retain the confined
resolver, Linux permission checks and fd publication. Checked bytewise snapshots stop at
the first NUL. Nonzero unlink flags use generic register dispatch before admission,
retaining AT_REMOVEDIR behavior.
Every admitted direct operation completes once; errors never trigger replay.
Enable `capstone_vm.native_path=1` together with `native_io=1` and a rebuilt
SDK/application. The new feature defaults off; old SDKs disable direct routing
on unknown feature bits. No ISA change is required.

SQLite moves 69 calls from exchange to registers. With previous native
features enabled in both arms, two samples per arm in A-B-B-A order reduce
kernel instruction attempts by 0.351% and total attempts by
0.207%. All 32 reference phases match in every run.
The build/file/SQLite/regression/GCC gate passes in 212.9s;
combined-feature IO and reversed measurements pass in 88.3s.
Each gate has one 300-second deadline.
[Exact evidence and limitations](../../runtime/virtual/tests/native-task/native-path-qualification.json).

Runtime and negative tests were implemented in parallel and independently reviewed.
This is not a new full-SPEC or FPGA qualification. 102 exchange calls remain
in this SQLite census. Next measure pathname byte counts and repeated checked-copy
setup, then prototype a bounded checked-string operation. Clock_gettime and bounded
vectors remain separate candidates; arbitrary iovec import remains unresolved.

## Native descriptor close qualified

Feature 64 routes close directly. Both transports share close_fd with grant revocation,
POSIX lock cleanup and restart-to-EINTR normalization. An authorized held-file reference
survives through close; descriptor reuse receives a fresh grant.
Every admitted direct operation completes once; errors never trigger replay.
Enable `capstone_vm.native_close=1` together with `native_io=1` and a rebuilt
SDK/application. The new feature defaults off; old SDKs disable direct routing
on unknown feature bits. No ISA change is required.

SQLite moves 45 calls from exchange to registers. With previous native
features enabled in both arms, two samples per arm in A-B-B-A order reduce
kernel instruction attempts by 0.164% and total attempts by
0.096%. All 32 reference phases match in every run.
The build/file/SQLite/regression/GCC gate passes in 212.1s;
combined-feature IO and reversed measurements pass in 88.0s.
Each gate has one 300-second deadline.
[Exact evidence and limitations](../../runtime/virtual/tests/native-task/native-close-qualification.json).

Runtime and negative tests were implemented in parallel and independently reviewed.
This is not a new full-SPEC or FPGA qualification. 171 exchange calls remain
in this SQLite census. Next qualify confined openat/unlinkat migration.

## Native scalar file services qualified

Feature `CAPSTONE_LAUNCH_NATIVE_FD` (16) routes exactly `fsync`, `fdatasync`,
`ftruncate`, `fchmod` and `fchown` through the register ABI. Both transports
share one held-file Linux provider, preserving VFS/LSM checks, signed truncate
lengths, uid/gid -1 sentinels and restart normalization. Scalar shape, fd grants
and once-only epoch admission precede effects. Close/fcntl/dup are separate.
Enable `capstone_vm.native_fd=1` with `capstone_vm.native_io=1` and a rebuilt
SDK/application. It defaults off; old SDKs disable direct routes for unknown
feature bits. Full measured configuration also enables locks/stat/seek.

SQLite moves 88 fsync and 22 fchown calls out of exchange. Two samples per arm
in A-B-B-A order reduce kernel instruction attempts by 0.39% and total
attempts by 0.23%, with all 32 reference phases matching in each run.
This is a small incremental gain with the other native features already on.
Build, expanded file tests, SQLite, regressions and original GCC comparison pass
in 211.4s; combined-feature IO and reversed measurements pass in
87.7s, each under one 300-second deadline.
[Exact evidence and limitations](../../runtime/virtual/tests/native-task/native-fd-qualification.json).

Runtime implementation and negative tests were developed independently in an
isolated worktree, then reviewed and integrated; QEMU runs remained serial.
The 35 new raw checks include rejected mutation preservation. No ISA change,
new full-SPEC verdict, crash-durability or non-root permission claim is made.

There are 216 remaining exchange calls in this SQLite census: clock_gettime 65,
openat 46, close 45, writev 34, unlinkat 23 and three single calls. Next candidates
are close and confined pathname services. Keep vector work independent: the
virtual provider lacks production nested-capability import for arbitrary iovecs;
a bounded register-vector ABI is a proposal, not implemented support.

## Native seek qualified

Musl routes `lseek` directly when launch advertises `CAPSTONE_LAUNCH_NATIVE_SEEK`.
Both transports use one held-file helper preserving Linux's shared file-position
lock. The handler admits each epoch once, enforces fd grants and scalar register
shape, rejects oversized whence values before narrowing, and normalizes internal
restart errors to EINTR. It has no request-buffer snapshot or copy workspace.
Enable `capstone_vm.native_seek=1` with `capstone_vm.native_io=1` and a rebuilt
SDK/application; the new feature defaults off. Unknown bits disable direct
routing in older SDKs, so preserve their old feature mask when needed.

SQLite moves all 2,283 seek calls to registers. With locks and stat enabled in
both arms, two samples per arm in A-B-B-A order reduce kernel instruction
attempts by 7.37% and total attempts by 4.53%. Every run matches
all 32 reference phases. Build, file/negative tests, SQLite, regressions and
original GCC comparator pass in 210.7s; combined-feature I/O and reverse-order
measurements pass in 96.6s. Each gate has one 300-second deadline.
[Exact evidence and limitations](../../runtime/virtual/tests/native-task/native-seek-qualification.json).

No ISA/QEMU changes or new full-SPEC/FPGA qualification. The independent review's
restart parity fix is included; restarting device callbacks were not injected.
Only 326 exchange calls remain in this SQLite census. Next implement the
independently reviewed scalar-fd group (sync, truncate, chmod/chown); close and
vectors remain separate migrations. Direct arbitrary iovecs lack production
nested-capability import, so do not merely redirect libc readv/writev.

## Native file metadata qualified

The native-I/O SDK now routes `newfstatat` and `fstat` through tagged registers
when launch advertises `CAPSTONE_LAUNCH_NATIVE_STAT`. A specialised handler
avoids the exchange round and generic request snapshot; both transports share
one confined Linux path/stat provider and the existing 128-byte stat conversion.
Pathnames are snapshotted byte by byte through checked accesses up to the first
NUL, preserving tightly bounded strings. Output rights, owner/lifetime checks,
fd grants and the existing supported flags/path policy remain enforced.

Enable `capstone_vm.native_stat=1` with `capstone_vm.native_io=1` and a rebuilt
SDK/application. The new flag defaults off. The specialised path uses the
existing `native_fast=1` default; the checked generic register route remains
available with fast dispatch disabled. Older SDKs disable direct routing for
unknown feature bits, so do not enable this feature for pinned older SDKs when
preserving their direct-I/O/lock performance matters.

SQLite speedtest1 size 1 moves 2,922 newfstatat and 1,463 fstat calls to registers.
With native locks enabled in both arms, two samples per arm in A-B-B-A order
reduce mean kernel instruction attempts by 15.36% and total attempts
by 10.50%. Every diagnostic/profile run matches all 32 reference phases.
Build, metadata/lock tests, SQLite, regressions and original GCC comparison pass
in 205.9s; the combined-feature I/O gate and reversed measurement repeat
pass in 90.6s. Each gate has one 300-second deadline.
[Exact evidence and limitations](../../runtime/virtual/tests/native-task/native-stat-qualification.json).

No ISA or QEMU change is required. These are incremental QEMU instruction
results, not FPGA timings or a new full-SPEC verdict. Exchange compatibility
remains. Next candidate: scalar `lseek`, still 2,283 exchange calls in SQLite.

## Native POSIX lock path qualified

Musl now sends F_GETLK/F_SETLK/F_SETLKW directly through the register ABI
when launch advertises `CAPSTONE_LAUNCH_NATIVE_LOCKS`. The specialised service
uses the existing checked-copy/owner helper and Linux POSIX lock helper,
without an exchange round or generic request snapshot. Input/output authority,
fd grants, epoch admission and Linux contention semantics remain enforced.
Enable `capstone_vm.native_locks=1` together with `capstone_vm.native_io=1`;
the new switch is default off. The specialised path uses `native_fast=1`
(the existing default); with it disabled, the checked generic register path
remains available. Rebuild the native-I/O SDK and relink applications to use it.

SQLite speedtest1 size 1 moves all 5,862 lock calls from exchange to register
routing. Two samples per arm, using the same binary and kernel in A-B-B-A
order, reduce mean kernel instruction attempts by 18.12% and total
attempts by 12.39%; every run matches all 32 reference phases.
The combined build, lock tests, SQLite, regressions and original GCC comparator
gate passes in 210.8s. The new-I/O-feature test and reverse-order
measurement repeat pass in 92.2s, each gate under 300 seconds.
[Exact inputs, counts and limitations](../../runtime/virtual/tests/native-task/native-locks-qualification.json).

This is a QEMU instruction result, not hardware timing or a full-SPEC verdict.
No ISA or QEMU changes are required. Exchange compatibility remains; next
consider metadata calls (`newfstatat`/`fstat`) using the same bounded workflow.
Older launch decoders disable direct routing on unknown feature bits; they
do not reject the whole program. Keep the new feature off for pinned older
SDKs when preserving their direct-I/O performance matters.

## Native broker census: next migration candidate

On the session-teardown kernel, SQLite speedtest1 size 1 passes all 32
reference phases in each of two fresh guests: a route census and a separate
untraced instruction profile. The combined gate takes 37.7 seconds under a
300-second deadline. No runtime or ABI changes are included.
[Measured inputs and results](../../runtime/virtual/tests/native-task/native-broker-census.json).

Remaining exchange calls include 5,862 POSIX file-lock requests, 2,922
newfstatat calls, 2,283 lseek calls and 1,463 fstat calls. The untraced profile
counts 352,883,780 kernel instruction attempts; cv_copy_check accounts for
10.25%, cu_copy for 6.26%, and ns_dispatch_request for 6.17% of those attempts.
These are flat symbol costs, not inclusive costs per syscall family or FPGA
cycles. A single run is a baseline, not a speedup result.

Next migrate POSIX file locking end to end: musl marshalling and the native
kernel service, preserving capability permissions, fd policy and Linux lock
semantics. Compare matched old/new SQLite binaries and negative lock tests
within bounded gates. Resolve feature negotiation first: current launch
parsing accepts exactly feature mask 1, so adding a bit unconditionally would
disable direct routing in existing binaries. Remove exchange handlers only after callers have
migrated; the broker has not yet been removed.

## Native session teardown encapsulated

Final session cleanup now lives beside session creation in
`native_session_destroy()`. The loader delegates the same three frees in the
same order; constructor failure unwind and task ownership are unchanged.
The incremental build, both I/O dispatch settings, task/kill, resource policy,
directory/dup/time, spawn and original GCC comparator gate passes in 104.0s
under one 300-second deadline. Adapter text/data/BSS sizes are unchanged.
[Qualification](../../runtime/virtual/tests/native-task/session-teardown-qualification.json).

This is a responsibility cleanup, not a performance optimization. Existing
negative admission tests cover policy failure on a partially initialized
context; they do not inject session allocation failures or prove leak absence.
No new full-SPEC or race-profile verdict is transferred from earlier kernels.
Keep future lifecycle or allocation-policy changes separate and use bounded
gates; the final-only destructor is not a reusable session reset API.

## Native generic copy workspace reuse qualified

`native_copy()` now reuses its session's existing kmalloc-backed copy workspace,
retaining both native-owner and Linux task-ownership checks. The checked-copy
engine, fault handling, mapping holds and other copy callers are unchanged.
Build, both ABI/fast settings, resource/spawn gates, original GCC comparison
and matched instruction measurements pass in 163.4s under one 300-second
budget. A separate 52.2s race gate passes 164 scenarios: pipe and file,
both fast settings, with safe and deliberately broken controls.
[Qualification](../../runtime/virtual/tests/native-task/copy-reuse-qualification.json).

Against the preceding authorization-reuse kernel, the fixed 1 MiB/4 KiB copy
loop uses 1.97% fewer kernel attempts in exchange and
2.53% fewer in generic direct mode. Fast direct remains a control.
These are workload-specific QEMU counts, not a GCC or FPGA speedup.
Keep any session teardown encapsulation or broader scratch reuse as a separate
change with its own ownership proof and bounded gate. Flat tables remain intact.

## Native authorization scratch reuse qualified

The generic capability preflight now reuses the existing kzalloc-backed native
context workspace through one offset-aware helper shared with the fast path.
Ownership checks and the checked-access protocol remain unchanged. The final
kernel's incremental build, both fast settings and ABIs, resource/spawn gates,
original GCC comparator and matched instruction measurements pass together in
160.6 seconds under one 300-second deadline. A separate 25.2-second
race gate passes 41 pipe and 41 regular-file scenarios, including negative controls.
[Qualification and measurements](../../runtime/virtual/tests/native-task/authorization-reuse-qualification.json).

For the fixed 1 MiB copy in 4 KiB chunks, generic direct requests use
2.42% fewer kernel instruction attempts; exchange requests use
1.90% fewer with fast dispatch disabled. Fast direct requests are effectively
unchanged. These are QEMU instruction counts, not GCC or FPGA speedups.
Next keep session teardown encapsulation separate; broader copy-state reuse
and slot accounting require their own proof and bounded tests. No new task
ownership mechanism is needed, and flat-table support remains untouched.

## Native helper calls: direct and typed

The built-in adapter now calls required Linux helpers directly through one
shared internal header. Twelve obsolete module exports are removed. Build,
native task/I/O, directory/dup/time, private-tmp, spawn lifecycle and original
GCC with its official comparator pass together in 93.1 seconds (300-second
budget). Adapter text shrinks by 986 bytes; no runtime speedup is claimed.
ISA, QEMU and flat-table support are unchanged. Full SPEC remains separately
qualified on its older frozen kernel; do not transfer that verdict.
[Direct-helper evidence](../../runtime/virtual/tests/native-task/direct-helper-qualification.json).
Continue in small changes using the bounded quick gate; a further migration
should select one remaining adapter responsibility and its regression scope.

## Native iterative feedback: bounded five-minute gate

The quick gate passes in 83.6 seconds on the cleanup kernel: incremental build,
native I/O/task, directory/dup/time, private-tmp policy, spawn lifecycle, and
original GCC test input with its official comparator. Its one 300-second budget
includes all configured work and lock waits; a timeout is never a pass. Seven
process-level supervisor tests pass. Full SPEC remains a separate qualification.
[Quick-gate evidence](../../runtime/virtual/tests/native-task/quick-gate-qualification.json).

## Native-only adapter cleanup: focused qualification passed

The legacy device/ioctl entry and scalar userspace trampoline are removed from
the committed native-task adapter. Context execution, checked copying, resource
grants and native direct/exchange service dispatch remain. The adapter now
requires an in-tree `CONFIG_RISCV_CAPSTONE_TASK=y` build; external-module builds
and legacy fallback admission are refused. New kernel `e2dca4c948f5` passes native I/O/task, directory/time/duplication,
private-tmp and spawn gates. The task gate explicitly proves no old device,
misc registration, trampoline mapping or control FD. The long remote SPEC rerun was cancelled to provide a five-minute feedback
loop; it is not a full-pass result. Local x264 continues independently. [Focused evidence](../../runtime/virtual/tests/native-task/native-only-cleanup-qualification.json). The complete ten-entry result below still
identifies the pre-cleanup kernel and must not be relabeled as a new-kernel pass.

## 2026-11-07 — Native C-port SPEC qualification complete

All ten existing C-port entries pass on the frozen final QEMU/Linux platform:
22 original benchmark invocations, four guest image validators and all 25
official output comparisons. Perl revision 07 completed in 4,831.9 host seconds,
with empty stderr and no logged kernel fault. The nine other entries use their
independently verified runs on the same final kernel.
[Final qualification](../../runtime/virtual/tests/native-task/spec-final-qualification.json).
The [RTL handoff](../plans/native-task-rtl-handoff.md) is ready for implementation
assessment; this result establishes neither FPGA operation nor performance.

## Native subprocess and shell qualification

Native capability-image spawn and admitted-child wait now pass focused QEMU
lifecycle gates in both exchange and direct SDK modes. These cover ordered
file actions, inherited grants, failed exec cleanup, reply-publication failure,
autoreaping, parent death and ambient-child exclusion. Bootstrap-only ptrace
races and cancellation during blocked bootstrap I/O remain unqualified.
[Exact scope and artifact hashes](../../runtime/virtual/tests/native-task/spawn-qualification.json).

A real BusyBox hush port uses native posix_spawn and passes exchange/direct
QEMU gates including large pipes, substitutions, functions, redirection,
background waits, a 100 KB heredoc and repeated descriptor rollback. It receives
an exact pinned shell image and optional exact `/dev/null` grant. General fork,
caller-replacing exec and caught signal handlers remain unsupported.
[Shell qualification](../../ports/hush/app/results/native-task.json).

The next kernel adds confined mkdir/rmdir, path chmod/truncate, hardlinks,
symlink inspection, directory iteration, correct destination handling for dup3,
and native nanosleep/clock_nanosleep/pselect. Each SDK passes 94 directory,
36 duplication and 33 time/select checks. The dispatcher passes 299 host checks
and ten negative controls; all six Linux patches install on a fresh fixture.
[Service qualification](../../runtime/virtual/tests/native-task/directory-time-qualification.json).

The original Perl test now reaches its child tests. Musl's hard-coded `/tmp`
is supported through a default-off alias to a private directory inside the
original launch grant. Four policy modes pass in both SDK profiles; unchanged
`perlio.t` completes all 48 results with zero unexpected failures, including
actual passes for the previously failing temp-file assertions 23–29.
[Private storage qualification](../../runtime/virtual/tests/native-task/local-tmp-qualification.json).

The old Perl build lost pointer tags during endian reversal, unaligned storage
and UTF-8 conversion, faulting at assertion 243 of `pack.t`.
[Original diagnosis](../../runtime/virtual/tests/native-task/spec-perl-pack-blocker.json).
The new `capstone_provenance_v2` interpreter port retains exact capabilities in
private scalar metadata and explicitly propagates or invalidates that metadata.
Frozen revision 06 passes the unchanged 14,708-assertion pack test, 89 pointer
checks and 101 independent controls in native direct mode; exchange mode passes
the same 89+101 controls. Host sanitizer and allocation-failure probes pass too.
It changes no ISA, kernel, SDK, benchmark input or comparator.
[Port and focused qualification](../../ports/perl/provenance/README.md).
All nine non-Perl C-port benchmarks now pass their original complete test
invocations and official comparisons on the final private-tmp kernel
`1bd9b8ecf702`, including all 156 x264 frames.
[Current non-Perl qualification](../../runtime/virtual/tests/native-task/spec-nonperl-services-qualification.json).
The full original Perl diagnostic on the preceding services kernel finished
with two failing test files: perlio (fixed and separately qualified above) and
pack (now repaired in the separately qualified interpreter). The complete original
Perl workload on the final kernel and revision-06 interpreter completed in
4,380.8 seconds: 315 of 316 test files passed, but `t/op/sort.t` failed and the
official comparison rejected the output. The provenance adapter inspected the
body of a scalar marked freed before Perl could raise its normal exception.
Revision 07 guards that marker and preserves the original error path; its
unchanged sort test matches all 183 baseline results on the host and passes
the focused native direct gate. Revision 07 subsequently passed both original
invocations and official comparisons. The obsolete partial diagnostic was stopped after confirming the missing services, not
reported as a full execution. The final record above supersedes this failed
revision-06 attempt. These service additions require no ISA/QEMU/compiler change; hush is a newly built native application.

## 2026-11-07 — Native cwd, pipe and simple signal repair

The native task branch now implements capability-confined cwd changes and
bounded parent traversal, transactional pipe creation, and default/ignore
signal dispositions. Both exchange and direct QEMU fixtures pass, including
concurrent renames, authority escapes, 128 partial payload failures and 64
fatal metadata-publication failures per path, descriptor exhaustion/recovery,
and real signal delivery. The host dispatcher passes 275 checks plus ten
negative controls; the installer applies the new Linux helper patch cleanly.
[Implementation and qualification](../../runtime/virtual/tests/native-task/README.md#native-cwd-pipe-and-signal-repair).

A focused original SPEC rerun passes GCC's exact reference and Perl makerand.
Perl test.pl now finds all 317 tests but cannot start their child processes
(SPAWN returns ENOSYS); its official output comparison still fails despite
exit zero. [Focused evidence](../../runtime/virtual/tests/native-task/spec-process-repair.json).

That repair added no ISA/QEMU/compiler/SDK requirement. At that snapshot,
`popen` still needed a confined shell and native subprocess support; wait was
not yet implemented. The subsequent subprocess qualification is recorded
above. The older full SPEC campaign below belongs to its frozen kernel; its nine passing benchmarks
must not be silently attributed to the repaired kernel.

## 2026-11-07 — Full native SPEC test gate and RTL handoff

The implementation snapshot `09b90be908e6` on `native/linux-task-integration`
now has a complete functional check of the ten existing C-port entries: **9
PASS, 1 FAIL**. All 22 original test invocations completed; 24 of 25 official
output comparisons pass. x264's full 156-frame encode and three image checks
pass. This does not qualify unported C++/Fortran entries or FPGA performance.

Perl makerand passes, but test.pl exits zero with incomplete output. A separate
QEMU diagnosis confirms native chdir returns ENOSYS; the test wrapper hides the
subsequent file-open exception. Pipe/popen and signal service gaps also remain.
No production code or benchmark source was changed during this qualification.
[Results and reproduction](../../runtime/virtual/tests/native-task/README.md#full-spec-test-input-qualification).

The [RTL handoff](../plans/native-task-rtl-handoff.md) inventories the cumulative
change, separates new interfaces from inherited hardware requirements, and
records the unmatched origin/dev commits. Its software acceptance gate remains
failed until Perl is closed; hardware implementation and synthesis are not
claimed.

## 2026-11-06 — Native Linux task architecture

`native/linux-task-integration` starts from measured `a13e2ffaa9a2` in a separate
worktree. The selected endpoint is a Linux-owned Capstone task with protected
context/mm namespace state, architecture entry/return, capability-aware syscall
access, and Linux signal/restart/scheduling integration. CSRUNV remains the
initial privileged entry primitive; direct hardware traps are separately gated.
A new syscall that only wraps dom_ioctl and a vDSO are not substitutes.

A default-off in-tree task/architecture-return prototype now runs without a
scalar trampoline or context-owning control FD. It passes GCC exact output,
51 authority/ABI checks, one-slot blocking transport, compute STOP/CONT/KILL,
user-time accounting, admission-exception recovery and 41 copy-race scenarios.
The concrete Linux 6.1 transition contract and reproducible installer are present.
A matched four-run GCC comparison shows 0.2125% more total attempted instructions
(0.4019% more kernel work), so this first implementation is not a speedup.

Syscall microbenchmarks now isolate entry and argument ABI: 128 valid windows
show direct-register specialised read/write saving 61–74% versus exchange on
the task backend. A generic-dispatch control also improves, while task entry
alone remains more expensive. Cached musl getpid never enters Linux and is
kept distinct from a forced scalar request. These probe results are separate
from the implemented first slice: opt-in read/write/pread64/pwrite64 with nonzero
requests up to 64 KiB, preserving caches and checked-copy rules. Select once
before effects, return direct short results without refill, and require a
validated loader feature announcement. Larger calls and vectors keep their
existing paths. See the [reviewed migration plan](../plans/native-linux-task-integration.md#reviewed-first-musl-register-abi-slice).
[Syscall measurement](../../runtime/virtual/tests/native-task/README.md#syscall-path-and-direct-register-measurement).

The first opt-in musl migration now implements that reviewed slice, with a
validated launch-v3 feature announcement and strict scalar canonicalization.
Fast/generic and feature-absent tests pass, as do 41 pipe and 41 regular-file
race scenarios. Matched GCC objects produce exact output in four ABBA runs;
total attempts change +0.0411%, kernel +0.0833%: no observed speedup. Separate
route tracing finds only one 69-byte read eligible in this GCC input; its three
writev calls and other recorded services remain exchange. Attribute remaining
kernel work by service/phase before broadening the migration. Global v3 activation
requires rebuilt SDKs; full signal/cancellation and Linux policy gates remain open.
[Implementation, reproduction and evidence](../../runtime/virtual/tests/native-task/README.md#opt-in-musl-small-io-migration).

The short SPEC follow-up adds two runs per SDK variant for ImageMagick, lbm
and nab. All twelve runs pass original SPEC comparisons; input hashes match
original test data. Mean total attempts decrease 0.0375%, 0.0900% and 0.0035%
respectively (kernel decreases 0.2332%, 0.6355% and 1.3880%). This reinforces
that the small-I/O migration is functional but not a large total-work saving
for these inputs. [Results and raw evidence](../../runtime/virtual/tests/native-task/README.md#short-spec-test-input-comparison).

A syscall-heavy resident-file copy now tests the actual musl integration:
1 MiB in 64-byte blocks makes 32,768 read/write calls and saves 65.91% total
attempts, 64.96% kernel attempts. Larger 4-KiB/64-KiB buffers save 61.08%/46.47%
total attempts. All twelve measured windows pass byte-for-byte output checks.
These CPU-work savings are workload-dependent and are not hardware speedups.
[Reproducible workload](../../runtime/virtual/tests/native-task/README.md#syscall-heavy-workload-through-real-musl).

SQLite file-backed speedtest1 now completes in both SDK variants after adding
real native POSIX locks, bounded fcntl duplication, sync, truncate, unlink and
fd metadata services. All 32 host-reference SQL phases match in each of four
ABBA measurement runs and two separate route-diagnostic runs. Direct small I/O
reduces mean total attempts from 657.856M to 630.508M (4.1571%), kernel attempts
from 433.044M to 410.436M (5.2207%). The same two application objects and unchanged
SQLite defaults are used. This is a tmpfs CPU-work measurement, not disk latency
or crash-durability evidence. The earlier failed admission remains historical.

Real Linux/domain contention, blocking wakeup, killed waiters, duplicate-limit
failure without lost locks, close-release semantics, confined unlink and raw
flock capability checks pass in both modes. Host dispatcher checks pass 275/275
under ASan/UBSan plus ten existing negative controls. Existing I/O, authority,
entry and lifecycle regressions pass. Review also corrected shared-position
locking around native lseek. [Implementation, reproduction and records](../../runtime/virtual/tests/native-task/README.md#native-file-services-and-sqlite-comparison).

The measured direct arm migrates 1,460 reads and 823 writes. Its 5,862 POSIX lock
requests, 2,283 seeks and metadata requests still use the exchange argument ABI,
though Linux services them inside the native task backend. Next attribute those
costs and qualify further ABI migration; do not infer savings for those services
from the small-I/O result or label this complete Linux syscall coverage.

Next close the remaining policy and qualification gates: native syscall
audit/seccomp/tracing (currently rejected), full register/FP/bounds and raw-platform
negative matrices, exhaustive lifetime/failed-exec race tests, then capability
signal/restart semantics and direct register ABI. Current repeated start/kill
checks do not prove absence of all leaks. This is quantum-assisted, single-hart,
root-only work, not complete Linux compatibility.
[Implementation and verified evidence](../../runtime/virtual/tests/native-task/README.md).

[Architecture](../design/virtual-capstone/native-linux-tasks.md) ·
[Milestones and acceptance gates](../plans/native-linux-task-integration.md).

## 2026-11-06 — Preliminary native GCC measurement

`experiment/native-gcc-preliminary` extends frozen `ea6464a9b2a5` with native
file/status/vector/system syscall coverage and kernel-buffer Linux helpers.
The unchanged GCC SPEC test input now produces the exact reference in all four
order-balanced runs. Mean total QEMU attempts decrease 7.499% (58.461M to
54.077M), kernel plus module 10.855% (32.541M to 29.008M). Both modes pass 20
new coverage checks and 51 authority/ABI/protection checks.

Plain follow-up: all six orders of plain/launcher/native pass exact output
validation (18 runs). Mean total attempts are 32.893M / 58.402M / 54.029M.
Native saves 7.488% versus launcher and remains 1.643x plain. This compares the
existing RV64/musl and Capstone builds on the same patched measurement kernel;
compiler revisions are not matched for an isolated ISA-cost claim.
[Three-arm record](../../runtime/virtual/tests/copyable-only/results/gcc-three-arm.json).

Phase follow-up (fresh guest per invocation): native compilation/output itself
costs 24.205M attempts versus plain 13.186M. About 55% of the native-minus-plain
excess occurs in startup/initialization and finalization/exit, 45% during
compilation/output. Kernel+module accounts for about 74% of the latter excess.
Next inspect allocation/mapping/fault work inside compilation; the native task
entry alone cannot be assumed to remove it. All six outputs and phase-count
conservation pass, including a wrong-boundary negative control.
[Phase evidence](../../runtime/virtual/tests/copyable-only/results/gcc-phases.json).

Memory follow-up: eight reference-exact fresh-guest runs isolate two existing
fault switches. Direct anonymous fault resolution plus disabled fault-ahead
saves 2.02% total, 3.97% in compilation. Mapping creation remains 281 native
versus 118 plain, all 281 native creations matching heap-mapping entries;
malloc implementation entries differ only 3.16%. Next isolate allocator
layout/alignment/group mapping amplification, preserving bounds and lifetimes.
No runtime defaults changed. [Memory diagnosis](../../runtime/virtual/tests/native-integration/README.md#gcc-memory-work-diagnosis).

Allocator attribution clarifies the cause: wider GCC pointers/records double
195 line-note allocations (3200->6400 bytes) and 65 pointer arrays (2048->4096).
Those two cohorts localize 124/163 extra mapping creations. Their raw allocator
requests have no extra alignment padding. Total malloc request volume is 1.87x,
not nearly equal despite the similar call count. Built mallocng matches pristine
1.2.5 plus patches 0009/0010; group release-policy functions are unchanged.
Four exact-output argument traces and repeated-sequence checks pass. Next isolate
reuse for these cohorts while preserving object bounds/lifetimes; do not assign
this mapping delta to brk or native task entry.
[Allocator diagnosis](../../runtime/virtual/tests/native-integration/README.md#why-the-gcc-heap-creates-more-mappings).

Ordinary-musl replay now isolates request-size effects: plain and Capstone
allocation/free histories produce 118 and 296 mappings in normal RV64 Linux
processes. Doubling only the two plain cohorts raises 118 to 245; halving them
in the Capstone history only lowers 296 to 277. Thus request growth is causal,
but allocator history/other sizes interact: the earlier 124/163 localization is
not an independently removable fraction. Eight fresh-guest replay runs pass;
plain reproduces GCC's original 118 mappings / 94 unmaps. This uses the same
patched kernel and is not a full computation/access-pattern replay. No production
defaults changed. [Replay evidence](../../runtime/virtual/tests/native-integration/README.md#ordinary-linux-allocator-replay).

This is a bounded experiment, not completion of native Linux compatibility.
Five SIGACTION and one IOCTL_BUF request are still refused in this unsignalled
run; GCC tolerates them. The exchange ABI and ioctl entry remain. Next qualify
native caught-signal/ioctl behavior and larger workloads before extrapolating
this startup-inclusive small-input result or undertaking the full task rewrite.
See [scope and reproduction](../../runtime/virtual/tests/native-integration/README.md#preliminary-native-gcc-result).

## 2026-11-06 — Qualified kernel syscall prototype

On `prototype/copyable-only`, the narrow Point 2 kernel consumer is implemented
and measured. Both dispatchers pass 51 ABI/authority/protection checks plus 10
entry checks, including two blocked processes sharing one hardware slot; both
pass all 41 deterministic RT cases (17 negative controls). Launcher regressions
still pass 22 checks each pinned/unpinned, and raw platform admission passes.
The default-off native path requires matching checked-copy/transport features,
a private FD broker and the normal-Linux mprotect wrapper. No legacy ownership
instructions return. Gemini Argon assisted review.

Median total QEMU attempts versus launcher: read 1 byte -54%, read 64 KiB -74%,
write 1 byte -40%, write 64 KiB -68%. Cached launcher getpid is faster; small
writes increase kernel/module work despite reducing total work. See the
[qualified integration and reproduction](../../runtime/virtual/tests/native-integration/README.md)
and [source import boundary](../design/virtual-capstone/syscall-import.md).

GCC follow-up: the existing launcher passes the unchanged SPEC test input twice,
but both native dispatch modes exit before producing output. Diagnostics expose
missing openat/newfstatat/writev plus signal, sysinfo and prlimit64 requests.
These syscall families need safe support and a reference-validated GCC rerun;
there is currently no GCC speedup measurement for Point 2. See the
[GCC attempt record](../../runtime/virtual/tests/copyable-only/results/gcc-native-check.json).

Next milestone is Point 3: native Linux task/architecture-return integration,
with explicit signal/restart/process semantics. The current ioctl trampoline
is only a prototype entry mechanism. Before broad workloads, extend unsupported
syscalls and requalify their authority/FD policies; measure remaining bounce
copies separately. MC's pinned mapping pool remains deferred. No full native
SPEC, SMP, DMA or RTL qualification is claimed.

## 2026-11-06 — Copyable-only prototype branch

`prototype/copyable-only` boots ordinary OpenSBI/Linux and runs copyable capability
applications with legacy ownership/control instructions rejected in M/S/U. Both
pinned and unpinned modes pass the directed runtime, fault, ABI and shutdown gates.
The existing tag-aware Linux patch also passes physical reuse, timer-copy and COW
movement checks with this model. Three matched allocation pairs show a 12.085%
reduction in instruction attempts with unpinned payload backing; no hardware
speedup is claimed. This is a one-hart QEMU branch qualification, not a landed
production or RTL baseline. [Evidence and commands](../../runtime/virtual/tests/copyable-only/README.md).
Full SPEC test inputs now pass four runs each for lbm, ImageMagick and nab.
Order-balanced mean instruction reductions from unpinning are 1.7905%, 0.7220%
and 0.0069%; mcf was stopped and excluded. This is a quick application check,
not a hardware timing result.
The page-lifecycle admission boundary is now independently enforced in Linux
and the module: one possible CPU, matching platform/kernel support, no spoofable
activation readback. Raw-operation privilege/precise-fault, platform-refusal,
runtime and page-lifecycle checks pass on the rebuilt kernel/module; see the
[hardening qualification](../../runtime/virtual/tests/copyable-only/results/hardening.json).

The fixed NS dispatcher, checked-copy loop and hosted context reference are now
imported as a provider-independent foundation for the syscall milestone. Their
hosted suites and 20 negative controls pass; the existing mapping/tag peer
fixtures compose successfully. No native kernel path or new application ABI is
activated. See [selection and prerequisites](../design/virtual-capstone/syscall-import.md).

# Current Capstone state

## 2026-11-10 — Normalized third-party defect classification

The 425 declared bug-corpus records now have a normalized, auditable classification artifact:
418 unique active defects, two duplicate records and five withdrawn records.  The active split is
263 temporal, 139 spatial, 11 other memory-safety, four non-memory-safety and one independently both.
Every record has a root CWE or an explicit unassigned value, plus manifestation CWEs, evidence,
confidence and review status.  Two independent Gemini passes, disagreement adjudication, 70
authoritative sanitizer-CWE controls and five direct provenance reviews leave zero unresolved rows.
See [the method and summary](../../bug-corpora/CLASSIFICATION.md) and
[`classification.json`](../../bug-corpora/classification.json).

## 2026-11-05 — Perl work-bounded comparison

On the same `ae47030ceeb6` SDK baseline, unchanged Perl makerand slices of
10,000/20,000 fixed-seed random draws measure +50.57%/+50.59% program attempts.
The completed makerand sub-workload from the subsequently cancelled longer
run matches SPEC's expected output in both arms and measures +50.60%.
This is not complete Perl harness validation. Over 99.9% of the short-slice
delta is in interpreter functions, not allocation. See the
[Perl evidence and attribution](../../ports/spec-cpu2017/results/2026-11-05-perl-slices/README.md).
Further instruction comparisons use matched work slices; full-run correctness
qualification stays separate. No source/input or allocator-policy changes.

## 2026-11-05 — Fresh SPEC/PHP instruction baseline

The SDK at `ae47030ceeb6` was rebuilt and all eight SPEC C applications plus
PHP 5.0.0 relinked. Six SPEC test workloads validate in both static-musl arms:
GCC +32.77%, mcf +3.48%, lbm +1.62%, ImageMagick +2.39%, nab +5.32%, xz +2.02%
in program instruction attempts. Perl/x264 time out at 600 seconds; the matched
100-call x264 slice is -1.04%, not a complete validated workload.
The compiler revisions differ; default Capstone libc uses -O1 versus plain
musl -O2. A libc-only -O2 diagnostic reduces GCC counts by 0.50%.

PHP's unchanged cache threshold excludes its widened Bucket/HashTable objects,
raising outer malloc calls. Its churn workload measures +38.94% for protected
musl/stock Zend and +73.89% with nested Zend protection versus plain; three
repeats per arm agree exactly. Direct derive/revoke attempts explain only a
small part of both the GCC allocator delta and PHP's nested delta.
SPEC originals (3,399 files) and frozen source packages pass integrity checks;
no input stubs, source edits, or free-payload scrubbing. The obsolete Perl
input-mutation helper is removed from the measurement tools. See the
[measurements and attribution](../../ports/spec-cpu2017/results/2026-11-05-minimal-mallocng/README.md)
for full counts, failed/replayed attempts, compiler limits and evidence hashes.

## 2026-11-05 — Minimal protected mallocng

The minimal port replaces `runtime/virtual/heap.c` and its object registry with
musl mallocng plus mapping-level authority and direct derive/revoke hooks.
Public raw context creation now rejects calls with ENOTSUP. Application
threads use pthread_create; the private context primitive is retained for
musl's clone backend and directed runtime tests. See the
[guard qualification](../../runtime/virtual/results/raw-thread-guard/qualification.json).
No freed-payload scrubbing. PHP retains its malloc parent in a prefix before
the unchanged Zend header. See the
[implementation contract](../../runtime/virtual/mallocng.md) and
[qualification](../../runtime/virtual/results/minimal-mallocng/qualification.json).
The qualification covers allocator denial/positive controls in both bounds
profiles, regular pthreads, PHP and fresh GCC/ImageMagick SPEC test workloads.
It does not requalify every application port or RTL. Existing SPEC sources and
inputs remain unchanged apart from the previously frozen GCC src.alt.

## Read this first (2026-10-13)

This file was 2,030 lines and its companion 1,560, most of it instructions and
results that had been superseded for weeks. Everything dated before 2026-10-01
is now in
[`history/13-10-2026_01-00-00_state-journal-through-2026-09-29.md`](../history/13-10-2026_01-00-00_state-journal-through-2026-09-29.md),
verbatim and with an index of all 58 sections. Nothing was rewritten or
dropped. The cut is by date alone, so a section that stayed is not thereby
current and a section that moved is not thereby wrong -- the dated entries
below say which is which, and for any individual defect `ref/ISSUES.md`
outranks this file.

**Where the work stands, as of 2026-10-12.** Four things are live at once:

- **Allocators.** The Sublet lifetime extension (`CDERIVE`, two-operand
  `CREVOKE`, node-scoped MANAGE) is qualified in QEMU and merged; QEMU #21 is
  `7dbfdd534eea` and runtime #216 pins it. The migration of all eight nested
  allocator ports onto managed lifetimes, and the removal of the physical
  Sublet heap, sit on `ports/nested-allocators-rebased` with fresh
  qualification -- 19/19 allocator, 43/43 runtime, 16/16 application checks --
  and are **not** merged into `dev`. One check is open: memcached fixture 24.
- **The virtual address space.** musl mallocng is the one system allocator; it
  executes in the application's virtual C context. The physical Sublet heap is
  retired, and virtual builds reject physical heap selectors.
- **Silicon (RTL lane).** The resident bitstream carries the S-16 fix; a
  gp-captable delegated application runs byte-exact on it. `fpga-testing-dev`
  was rebuilt as one linear history on 2026-10-10.
- **CheriBSD.** Our PoisonCap adapter is removed from every port except
  PostgreSQL, which goes with its rewrite. Stock CheriBSD remains the
  comparison vehicle.

What to do next is [`current-next-step.md`](current-next-step.md), not this
file. This one records what was established and when.

## 2026-10-17 — The virtual profile's RTL is complete for the two-bitstream plan; both builds wait on the lead's go

vp1 (`virtual-profile` 9ed9a2594: translation in C mode) and vp2 (`virtual-profile-2-d` bf3db3175: the flat per-context
lifetime table, CSRUNV with one saved context and FP state, CSRETIRE/CSNODE, the policy list, the collection strobe, the
reserve, Q-15's representability refusal, Q-18) are gated in simulation (lint at baseline by name, the 95-test sweep, the
152-test supervised list, every virtual-profile arm as pre-registered), pre-registered on dev before their handoffs, and
handed to the synth lane, which built both in parallel on 2026-10-18 on the lead's direct go: vp1 sealed at routed WNS -15.615
(sha256 01072623...2905), vp2 at -19.019 (sha256 776cdd8c...7475), LUTLP-1 0 on both, graded in the plan doc; the flashes wait on
the lead's own word to the board lane. The
monitor's CSMINT emulation (the board lane's routine, included at a pinned hash) reads every pre-registered value on the
vp2 RTL through the shipped trap entry. Silicon still runs `caplifive_sublet_d93a8d66b.bit`. Details and the next step:
[`current-next-step.md`](current-next-step.md) and the plan's rows for items A, B, D and E.

## 2026-10-12 — Virtual allocator cleanup qualified on the migration branch

`ports/nested-allocators-rebased` uses managed lifetimes in outer musl mallocng
too. Its allocation policy is unchanged; linear heap loans and their final
carved-corpus consumer are removed. Virtual build metadata names `mallocng`,
physical heap selectors are rejected, and cache keys include the musl
representation sources. The [qualification](../../runtime/virtual/results/managed-mallocng/README.md)
records 19/19 allocator checks, 43/43 runtime checks, 16/16 normal port checks,
paired FFmpeg 13/13 and PostgreSQL 5/5 corpus results with safe controls.
memcached passes 81/81 outer-only checks and 82/83 with protected pools;
fixture 24 remains unresolved. Recomputed CPython archived sharded outcomes
agree for all 2,183 stock/protected tests. Full upstream campaigns and older
measurements retain their original scope. These cleanup commits are on the
migration branch, not claimed merged into `dev`.

## 2026-10-10 — CheriBSD PoisonCap removed (branch `nested/poisoncap-removal`)

Our CheriBSD PoisonCap adapter is gone from every port except PostgreSQL, which goes with
its rewrite: the FFmpeg, wmem, memcached, APR and Perl SV-head backends, their runners,
the `poisoncap-*` arms of the fourteen FFmpeg, tshark, memcached and httpd corpora, the
two PoisonCap columns of `catch-tables.py`, and the study's `nested-poisoncap` planner
and build paths. The platform recipe (`ports/ffmpeg/buffer-pool/host/cheribsd/poisoncap/`)
and its libc fix stay, because the stock CheriBSD runners of the httpd and memcached corpora
boot that image. Every PoisonCap entry below is history; its results stay in `results/`.

## 2026-10-12 — Sublet lifetime extension merged; allocator migration in review

`CDERIVE`/two-operand `CREVOKE`, node-scoped MANAGE and explicit parent links
are implemented and qualified in QEMU with C wrappers. The format has
30-bit IDs, four rights bits and 16-byte node records. Rebuild the Linux
adapter with the new header; its capacity limits change too. Compiler
backend changes are unnecessary. See the [qualification and commands](../design/virtual-capstone/development.md#sublet-lifetime-qualification).
QEMU #21 landed as `7dbfdd534eea`; runtime #216 landed as `baa1ab2faded`,
which pins that integrated QEMU commit. The allocator migration branch builds
on this baseline. Its archived port and workload measurements retain their
original source and binary identities; rebasing does not requalify those runs.
This is QEMU evidence, with no new performance or RTL claim.

## 2026-10-10 — `fpga-testing-dev` rebuilt as one linear history (RTL lane)

- capstone-ariane's default branch is now 44 commits above the fork `e1b3db6ba`, one per logical change, tip `5382266b8`;
  its RTL is byte-identical to `776d9d859` (the current bitstream). Old tip `0bf09b1d6` = tag
  `backup/fpga-testing-dev-2026-10-10`. Lane branches, the parent's submodule pointer and R-51/R-52's branches are
  untouched. Method, table of hashes and the carried arms' readings:
  `docs/history/10-10-2026_08-40-00_fpga-testing-dev-rebuilt-linear.md`.

## 2026-10-08 — mallocng executes locally in virtual Capstone

Branch `virtual-musl-local` replaces the virtual buddy heap with Capstone-compiled
musl 1.2.5 mallocng. malloc/free and ownership operations execute in the
application's virtual C context. Actual syscall/VM, node-pressure, fault and
scheduling paths cross the existing trusted Linux adapter. Upstream allocation
policy is retained; patch 0009 ports capability representation and lifetime
hooks. The fixed allocator block/arena arrays are removed.

ABI v5 requires QEMU's opt-in exact physical-shadow bounds profile, pinned at
`24b1e95b1dac`. This is extra QEMU metadata, not a tag-bit-only or RTL design.
The default physical processor profile and physical allocator selection retain
their previous behavior. Linux core and firmware are unchanged.

The [qualification](../../runtime/virtual/results/local-mallocng/README.md)
records 19/19 allocator, 42/42 runtime, 10/10 pthread, 16/16 other normal-port
and 73/73 memcached checks, plus 202 processor checks. It includes 200,000
lifetimes, 70,000 live objects, exact-site spatial/temporal denials and a
control detecting an inserted allocation-loop service. All eight application
ports are relinked with the new SDK. The compiler and platform are reused;
the two long sparse runtime cases and the CPython inner / FFmpeg-tshark safety
corpora are not requalified here. The memcached fixture-17 setup now searches
for actual reissue; its original temporal-fault prediction is unchanged.

The withdrawn native-heap branches are not the integration target. Merge the
node-growth stack, then QEMU [#19](https://github.com/project-starch/capstone-qemu/pull/19),
then the matching local-mallocng runtime. Rebuild SDK and applications together.
The [allocator contract](../../runtime/virtual/mallocng.md) explains the
representation, metadata cost, VM remap transaction and remaining limits.

## 2026-10-07 — growing virtual node tables

The `review/virtual-node-growth` branch builds on `dev` at `08cf07d75ad2`
and the application build repairs in `review/virtual-build-repairs`. Its QEMU
pin includes the `b649187a989c` UNINIT diagnostic fix and a paged guest node
store with a stable physical root,
unchanged 31-bit IDs and 16-byte records, and chained collection requests.
The trusted Linux module allocates base pages on demand and reuses IDs only
after the existing complete stale-tag sweep. The SDK requests node capacity
before heap allocation. Linux core, firmware, upstream musl and the legacy
host-tree algorithm are unchanged. Applications should be relinked with the
new SDK to use proactive growth.

The [node gate](../../runtime/virtual/results/node-growth/nodes.json) passes
15/15 checks: two rounds of 70,000 simultaneously live structural nodes,
140,004 reclaimed IDs, stable capacity at 70,656 slots (1,142,784 metadata
bytes), stale denial, bounded quota failures and teardown. Processor gates
pass 69/69 virtual-runtime, 60/60 M1 and 69/69 U-access checks. The actual C
node backend matches the forest model for 8,949 prefixes per binding
(host, flat guest, paged guest), with 19 directed paging checks and mutation
controls. The [pthread gate](../../runtime/virtual/results/node-growth/pthreads.json)
passes 9/9.

The [complete runtime gate](../../runtime/virtual/results/node-growth/runtime.json)
passes 46/46, including both 200,000-allocation recycling cases, exact-site
stale denial, spatial checks, SQLite persistence, mruby and Perl smoke. The
sparse positive case uses 348 KiB of node metadata, reclaims 188,631 IDs and
leaves unused payload pages unpopulated. All processor and runtime gates use
the same QEMU executable. See the [qualification manifest](../../runtime/virtual/results/node-growth/qualification.json)
for hashes, reused compiler/musl/platform inputs and application relink scope.
The rebase also passes 87/87 native CTest checks under ASan/UBSan. An additional
physical LINEAR/UNINIT corpus attempt fails at `create_dom` before its probes;
the cause remains unresolved and physical compatibility is not requalified.

This is one-hart QEMU evidence. The 65,536 allocator block-record limit is
separate and unchanged. Node capacity is bounded by RAM, an optional module
quota and the ID field; individual published pages are retained until exit.
See the [runtime mechanism and reproduction commands](../../runtime/virtual/README.md).

## 2026-10-05 — B0: a gp-captable delegated application runs byte-exact on silicon

- **b0-hello passes on `caplifive_supcall_715bdd1fe.bit`, byte-exact under a `cmp` oracle, N=2** (boots 13 and
  13b, preflight off for a documented reason).
  - Four delegated rounds: HELLO, `ioctl` -> -25, `writev` -> 51, `exit_group`.
  - Identity was verified by hash. fw e411395bde2c embeds Image d5ab7fad, whose initramfs carries b0run.sh
    cc858ec4, b0-hello.dom cdd82e56, capstone-exec 1dca5b19 and module 0332e2d6.
- **What runs:**
  - a full-LTO gp-captable musl application, launched by capstone-exec through the process-ABI module;
  - stepped by the FPGA monitor's supervised CALL under CSR events;
  - with no fabricated gp;
  - with its domain torn down at exit (census `live_domains 0`).
- **Found on the way, all QEMU-silent:**
  - **C-76:** capstone-c's `stc` spill of a `__rev` argument, moved to cnull by silicon's STC. It wedged every
    process-ABI step.
  - **A delegate-runtime copy loss** in R-29's family, worked around with 8-byte copies; an RTL fix for R-29 and S-10b is
    FIXED ON SILICON (2026-10-05, caplifive_supcall_776d9d859.bit: b0-memcpy unguarded 94/96 -> 0/96, s06agg 66 -> 64; ISSUES R-29). The first attribution
    (R-29's stale-refill mechanism) was retracted. The RTL lane then reproduced the actual path in simulation: a
    granule copy of a plain-store pair still in the write buffer reads the high half as 0.
  - **M-13:** a monitor-internal trap reads as cause 24 at `_cap_trap_entry`+4.
  - **The board kernel cannot unload modules**, and a module built out of tree needs `-march=rv64g`.
- Record: `docs/plans/b0-silicon-delegated-runtime.md`. The module's `vm_flags_set` port is on caplifive-buildroot
  `capstone-bootstrap` (3f485c4).

## 2026-10-04, evening — the S-16 fix bitstream on silicon: S-16 fixed, the R-43 list unchanged, and the real workload preempts 1,277 times at +0.69 %

- **The resident bitstream is `caplifive_vp2_bf3db3175.bit` since 2026-10-17** (sha256
  776cdd8c40d985708a09d4d43ab1c5106414ae1347cc4436a711381d2c197475; routed WNS -19.019, 3.404 behind vp1 and inside the
  one-sample null band of 3.74 ns, LUTLP-1 0). capstone-ariane `virtual-profile-2-d` bf3db3175: the FULL virtual
  Capstone profile on vp1's base (item B translation in C mode, Q-17, R-56, Q-19) plus item A's flat per-context
  lifetime table, item D's CSRUNV with the frame as the one saved context's storage (kinds 1-5, FP state per context,
  CSRETIRE in hardware, CSNODE for the monitor's CSMINT emulation), items E and H, Q-15's exactness refusal, Q-18, and
  the decoder and CSR-write fixes. Flashed by the board lane on the lead's own word after vp1's first session (dev
  987bf254ee40, `tests/rtl-smoke/vp2-bare-2026-10-17/README.md` "Flash authorisation"): the canary 9 of 9 against
  d93a8d66b, all 25 pre-registered arms equal to their vectors simulated on bf3db3175, corpus set 7 33 of 33. The
  attribution of its issue_read_operands growth (the DYN unit's unregistered result chain, real logic from item D) and
  the un-built vp2b timing contingency are in `docs/plans/2026-10-16-virtual-profile-on-silicon.md`. Known RTL/QEMU
  divergences carried: Q-16, Q-20's software rules, Q-21 (0xcc0 absent), Q-22 (0x5c0 absent). Registered fallbacks:
  `caplifive_vp1_9ed9a2594.bit` (sha256 01072623f8f24b2081556c193113ca906d68316fdc1de15a55ae8267ddf82905, its first
  session clean, dev c5e696ac7328) and `caplifive_sublet_d93a8d66b.bit`.
- **Earlier resident: `caplifive_sublet_d93a8d66b.bit`, 2026-10-15 23:54 to 2026-10-17** (sha256
  89ba6d816c23b1115585ff6bcc2d5ceeb4523967731a84baf4f8955256077430; routed WNS -15.856, outside the band, and it runs:
  the S-16 bare set identical to the FP respin's). It carries the Sublet extension on top of fp-fixes 4a4261bb4: the
  capability word {revnode_id 30 | perm 4 | cap_type 3 | bounds 27} (MANAGE = perm bit 3), CDERIVE / CREVOKE with the
  node parent link, MREV deepening, CREVOKE in R-45's flush, the non-linear SPLIT keeping its node. Its decoder does
  not read the old word: FP-era images need rebuilding under the full format. Confirmed on silicon 2026-10-16 by the
  bare floor (capstone/tests/rtl-smoke/sublet-bare-2026-10-15, five arms, every reading as pre-registered): the Sublet
  instructions and their refusal causes, deepening, the shared-node split, a supervised domain deriving and ending
  children, and that every load and store through a revoked capability is refused with cause 25. It needs the R-55
  monitor (explicit cursors after every carve) like the FP respin. Registered fallback: `caplifive_fpb_afcd19bea.bit`.
- **Earlier resident: `caplifive_fpb_afcd19bea.bit`, 2026-10-15 01:31 to 23:54** (sha256 d8d2b641da70207c74d41a69ef60a561426fc573b6767fd03bd6d3c28a5fb3f8; routed WNS -15.310, outside the -13.335..-5.855 band, and it runs: every FP acceptance row as pre-registered). It carries fp-fixes afcd19bea: R-53, the R-52 residual, R-54's renormalisation in residue form, the encoder without the cursorless scheme. The full-scheme encoder needs the R-55 monitor (caplifive-opensbi df0e9a8 / capstone-sbi 6dad416 on the board line): explicit cursors after every carve. Registered fallback: `caplifive_r51r52_0568f93a9.bit` (resident 2026-10-11 to 10-15). The paragraph below describes the earlier resident 715bdd1fe and is kept for its workaround history.
- **Earlier resident: `caplifive_supcall_715bdd1fe.bit`.**
  - capstone-ariane sup-call 715bdd1fe: R-49, R-50 and the S-17 apertures 219..222.
  - Flashed 18:41-18:43 on the lead's direct word with `drivers/flash-bitstream.py`. The sha256 matched the sealed
    hash, and the name was read back after the power cycle.
  - Set `FPGA_BITSTREAM=caplifive_supcall_715bdd1fe.bit` for every board run. The previous resident,
    caplifive_supcall_36a641e0b.bit, stays registered on the console.
- **S-16 is fixed on silicon.** Audited wording and limits are in the S-16 entry in ISSUES and in its folder.
  - Bare: 14 of 14 previously-hanging arms complete without a fence.
  - **The FPGA monitor's supervised SQLite speedtest:** the C5u firmware, byte-identical and with no fence, completes
    1,277 preemptions with the oracle hash at **+0.69 %** over the plain call (quantum 2,000,000). On 36a641e0b it
    hung after 212. Preemptive supervision of a real workload now works on silicon.
- **The R-43 acceptance list passes unchanged**, a1..a10 on the same images
  (`tests/fpga-repros/R43-revocation-cache-false-deny/results/board-715bdd1fe.result-lines.txt`).
  - Identity readings are exact: instret 17/17, hashes, Sublet counts, a10's trap and refusal record.
  - Cycles are within noise.
- **S-17 does not reproduce** (3 of 3), mechanism unknown. The new apertures are armed.
- **The trap log's 0x83 residue on bare hangs is the JTAG load's debug-module `ebreak`** (mepc 0x35c). It means no
  trap since the load.
- **Next:** B0, the delegated hello-world, on this bitstream (branch `b0-silicon-runtime`). It already passes in QEMU
  with no fabricated gp.

## 2026-10-04 — the app ports' and corpora's last 24 registered cells measured, two retractions, and the ledgers closed

- **What was asked:** audit the defect evidence for memcached/tshark/FFmpeg against the paper, then
  finalize everything.
- **The audit's answer: count was never the problem.** 22 upstream corpus cases across the three, **all
  22 nested-allocator**, now **22/22 with a measured capability arm**; plus 59 app-port fixtures carrying
  24 measured nested-discriminating cells. Only **1 of 22** is live at the pin, which matches the paper's
  own wording ("previously fixed bugs"). The gap was **printed numbers without backing runs** — see
  `/tmp/capstone/2026-10-04-paper-defect-evidence-audit.md` (not committed; the paper is the lead's).
- **Measured, all as pre-registered** (predictions committed before every run):
  - **FFmpeg pool corpus, fixtures 40-47, both app-port arms: 16/16 cells.**
    `ports/ffmpeg/app/results/20261004-qemu-pool-corpus-40-47/`. `poolsublet` faults cause 24 at
    `case.c:85 / :48 / :33 / :104`, each with non-zero `value_hi`; `poolstock` completes every defect
    with DEFECT-REPRODUCED. **Case 3's `sublet-port` arm was the last unmeasured arm of the 22.**
  - **FFmpeg case 3's component-port pair, probe slot 39: 2/2.**
    `ports/ffmpeg/buffer-pool/results/measurements/20261004-vvc-case3-probe39/`.
  - **tshark `chunks` fixtures 2-9: 9/9** (fixture 1 as the in-boot control).
    `ports/wireshark/app/results/20261004-qemu-chunks-2-9/`. Read as a consistency check: these are
    plain-heap probes on a nested arm, not new nested evidence.
  - **memcached corpus, 5 cases × spatial/sublet: 10/10, plus the suite-wide negative control** (exit 0 =
    every oracle fired). `ports/memcached/allocators/results/20261004-qemu-corpus-defects/` — the
    corpus's first committed bundle; it gitignores its own `results/` by policy.
  - **ASan blindness to pooled reuse, 4 arms**, with upstream's own `e6255fb822` as the positive control
    and arm C showing the `free_entry_cb` gate leaves the MPVPicture pool uncovered even on master.
    `bug-corpora/ffmpeg/pool-repros/results/2026-10-03-native-asan-pool-blindness/`.
- **New instrument: `ports/common/application/run-fixtures-9p.py`.** `capstone-vm` is the app ports' only
  runner and needs ssh; no riscv64 dropbear exists here to give it. This stages images and the pinned
  launcher/module over 9p, issues **the same guest command `capstone-vm` issues**, merges the launcher's
  fault record into the result as `capstone-vm`'s host side does, and judges with the **committed**
  oracles (`check-safety.py`'s `classify()`/`matches()`, or an external `--verdict` script). Negative-
  tested: forced against a wrong row it reports DIFFERS and exits 1.
- **TWO RETRACTIONS, both mine, both from generalising a partial enumeration:**
  1. *"The app SDK gate refuses every toolchain on this host"* (`5208789e4b9e`). It does not: the
     `llvm-capstone-cc` worktree's `build-release`, commit `7d01722aab88`, passes it end to end, and all
     16 FFmpeg app images were built through the gate with it. I had globbed `*/llvm/cmake-build-*`;
     `ffmpeg-live-defect-triage.md:281-284` already named the qualified compiler.
  2. *"No replacement harness exists"* (`dedab28c4a2c`). One existed on disk and had run app-SDK
     fixtures the previous day; the sibling bundle said so outright. Propagated into the four files that
     still carried the first claim (`49c4903ca277`).
  Rule captured as the `blocked-is-an-enumeration-claim` memory.
- **Also corrected:** `dedab28c4a2c`'s `g_refpool` caveat named the **wrong bundles** — the affected one
  is `bug-corpora/ffmpeg/pool-repros/results/20260929-qemu-sublet-port/`, which now carries it in-tree
  rather than only in a commit message; and that port bundle's claim that the oracle's FAULT branch is
  "structurally unreachable" (it is reachable — the host merges the fault record).
- **Still open, deliberately:**
  - **PostgreSQL's 8 spatial/sublet cells.** The paper prints Sublet catching 8 of 8 while all 8
    `case.json` say the arm has **not** been run. Highest-value single measurement for the paper; out of
    scope by the lead's decision (report only).
  - **`INDEX.md` / `index.json` are stale on `dev`** from the board lane's fpga-repros and ISSUES
    additions. Only this lane's own lines were applied; the lead asked that the two lanes squash and
    merge one consolidated regeneration.
  - PoisonCap and CheriBSD for memcached and FFmpeg: that platform is absent from this host.

## 2026-10-04 — Supervised CALL under the FPGA monitor on silicon: a real workload preempts and resumes hundreds of times, then hits S-16 (localised, mechanism traced, an RTL fix needed); S-17 found beside it

- **The FPGA monitor runs supervised CALL on silicon.**
  - Where it lives: capstone-sbi `monitor/supcall-fpga`, behind the opt-in `CAPSTONE_SUPERVISOR_CSR_EVENTS`.
  - The C5 boot hang was the monitor's 21.5 KiB of globals carved from an 8 KiB FPGA `dom_stack`. It is fixed in the
    wrapper and gated in `build-fpga-fw.sh`.
  - The merged monitor with supervision compiled out passed C3.
  - Folder: `tests/rtl-smoke/supmon-2026-10-03/`.
- **Silicon supervision forbids every M-level CSR inside a domain, by design.**
  - The legacy my_first_domain-family glue saves and restores mcause/mtval, and every cycle bracket reads mcycle.
  - `sup-static-audit.py` finds every such instruction (controls both ways).
  - `patch-sup-glue.py` replaces exactly those words. The patched speedtest gives the oracle on a plain call at
    -0.04 % cycles.
  - The musl glue is clean.
- **The real workload under supervision.**
  - The SQLite speedtest resumed 212 times (C5u), then the switch hung.
  - With a fence before every domcall, it resumed 552 times (C5f), then hung again, on the escape side.
- **S-16 (`tests/fpga-repros/S16-supervised-switch-never-finishes/`).** A domain switch that starts while the
  store buffer's commit queue is full never finishes.
  - The switcher's walk-idx-7 write is never acknowledged; commit count 4, WAIT_STORE_READY.
  - Reproduced bare within 16 resumes.
  - 12 stores before the CALL complete; 18 and 24 hang.
  - The RTL lane's simulation matches every silicon read: the store unit acknowledges the switcher's write on the
    previous store's queue ready, so the write is pushed over the full commit queue's head. **The oldest committed
    store is lost**, and the 4th later write starves.
  - **Every switch kind is exposed (bare, 2026-10-04, the RTL lane's `sup-s16-stores.S`).**
    - The armed CALL, a supervised RETURN right after a domain store, and an escape hang at walk idx 7.
    - A PLAIN CALL hangs too, at idx 4, its exchange write.
  - **A fence before the CALL covers only that CALL's own switch.** The RETURN and escape arms hang with it.
  - **A fence before every CALL AND every RETURN completed all four twins that hung**, so runs without quantum
    preemption have a software workaround.
  - A quantum escape lands wherever the domain is. C5f's hang was on the escape side, and so is bare
    `s16st-esc-n8-retfence`. **Preemptive supervision on this bitstream needs the RTL fix**; that lane has one
    passing in simulation.
  - **Withdrawn:** the memcached plan's "unsupervised contexts do not depend on S-16".
- **About 22,000 bare supervised resumes without the swap sequence never hung.**
- **S-17 (`tests/fpga-repros/S17-ldc-after-supervised-switch-lsu-stuck/`).**
  - After a finished switch, an LDC right behind `ccsrrw sp <- cscratch` leaves the LSU not ready. A scalar `ld` there
    completes 8,552 escapes.
  - A plain CALL is affected too.
  - The simulation hangs at the same LDC in a different state; open.
- **Instrument added:** the bare runner `sup-resume-2026-10-03/run_sup_bare_wedge.py` reads the switcher and LSU
  apertures (192-195, 224-229, 238-240) after a hang. The commit-pc aperture during a stuck switch is stale scoreboard
  slot 0, shown by a positive control.
- **Corrections along the way, recorded in the lane PREREGs:**
  - the k800 entry path;
  - "the hot loop never misses";
  - "MSWAP is void" (my x26/s10 register clash);
  - the S-17 prediction ldsync = 1;
  - the C5f prediction "all supervised runs complete".
- **Still open:**
  - the RTL fix for S-16 and its bitstream. A reflash is the lead's call.
  - S-17 on silicon.
  - Supervised contexts on silicon (memcached Part 4b) stand behind S-16.

## 2026-10-03 — Atomics through a capability address work on silicon: the first memcached prerequisite

- **The question.** memcached.dom links 171 atomics: 80 `lr.w`, 80 `sc.w` (musl's `a_cas`), 9 `amoadd`, 2 `amoswap`. No
  image proven on silicon contained any, and no RTL directed test ran one through a capability in capability mode.
- **The test.** `tests/rtl-smoke/cap-atomics-2026-10-03/`, bare on `caplifive_supcall_36a641e0b.bit`, pre-registered
  and pushed before the boot (49d7f98067c3). It has 22 architectural readings.
- **PASS, all 22 exact**, cited by image hash:
  - the control, call-retpc a05ca464, exact (N = 4 on this bitstream);
  - amoadd/amoswap .d/.w, including the 32-bit wrap;
  - lr/sc .w/.d success;
  - SC failure with no reservation and on another granule;
  - a musl `a_cas` loop in one attempt.
- **Not covered:**
  - AMO bounds or permission enforcement;
  - the I4 tag residual (an AMO over a capability granule keeps its tag; the RTL lane's
    `s06sec-amo-no-resurrect.S`);
  - multi-hart contention.

## 2026-10-04 — S-16 root-caused: the switcher's first SAVE write pushed into a FULL store-buffer commit queue; fixed in RTL with a second switch-path defect (R-49, R-50); the resident bitstream needs a `fence` before every CALL and every RETURN, and preemption needs the reflash

- **The resume hang under the FPGA monitor (C5u/C5q, the board lane's S-16) is a store-path defect, not the adapter.**
  The store unit acknowledges a switcher write on the room of the PREVIOUS store's queue, so with four committed
  stores still queued the write lands over the commit queue's head: one committed store is lost and the queue
  deadlocks four writes later, at walk id 7. The board lane's wedge reads (idx 7, WAIT_STORE_READY, commit count 4,
  the write request valid and unacknowledged) and the simulation reproducer `sup-s16-stores.S` (N = 32 stores before
  an armed CALL, memory delay 12 and 40) agree on every aperture. Fixed on capstone-ariane `sup-call` 192a5e624
  (registry R-49). **Workaround on the resident bitstream** (the board lane, bare on silicon, narrowed 2026-10-04):
  - A `fence` before every CALL covers that CALL's own switch only.
  - A fence before every RETURN as well completed every non-preemptive twin that hung.
  - A quantum escape stays exposed. See the S-16 entry above.
- **A second defect, found on the way (R-50, 429c60b32):** the load unit's dom-switch flush exemption outlives the
  switcher's read, so the first LDC after an escape, when dispatched in the swap-in CCSRRW's flush cycle, runs
  un-killed and its stale completion pops the re-issued LDC unexecuted; the DYN unit waits forever. Deterministic in
  `sup-mswap-noploop` on the resident logic; fixed by adding the entry's valid bit to the exemption. Not the silicon
  S-17 (different aperture signature; S-17 stays open).
- **Synthesized and sealed (2026-10-04 evening, the synth lane):** `sup-call` 715bdd1fe (the two fixes plus the
  S-17/S-16 LED apertures 219..222) -- route clean in ~21 min, LUTLP-1 0, WNS −9.949 in band, one TIMING-23 loop
  (arc relocated within the same cluster), census unchanged; TNS +49% with the LED apertures carrying the worst-path
  shift (0.36 ns from WNS) but not the endpoint growth. Bitstream sha256
  a7add4c019d3067627ba984634e96de564f023bcec50e2c29fd0f668c0a81311. The board lane's acceptance batch (17 bare images
  plus the monitor without the fence) is staged; the reflash is the lead's word. Details in the plan's step-11 result.
- **Verification so far:** matched pairs for both fixes; `sup-mswap-noploop/base/postread` 78 switches on the fixed
  tree; lint gate at the committed baseline; 95-test sweep: 95-entry sweep on the committed tree against the clean baseline, seed 20260922, memory delay 12: 92 identical in taken exceptions, CAPPRINT readings and retired-instruction counts, 0 differ; the 3 random-generator entries produce no log on either side, as in every previous sweep. Sim-only tracers (every retirement, write-back,
  LSU queue event, load/store-unit transition, the switcher's busy edges, and the board's apertures emulated every
  4000 cycles) in b576635be/d92093828. **Next:** the synthesis candidate is `sup-call` HEAD; synthesis and the
  reflash are the lead's word; then the board's S-16 arms (armdep-nt, s16sd24) must complete without the fence, and
  the FPGA monitor's supervised speedtest runs again.

## 2026-10-03 — Supervised-CALL ladder complete on silicon: 12 images exact to simulation, the real timer escape, S-11's refusal; N = 3 on the key tests

- **The rest of the RTL lane's ladder ran bare on the board** (`caplifive_supcall_36a641e0b.bit`), one session per image,
  pre-registered and pushed before the boot (`tests/rtl-smoke/sup-bare-2026-10-03-ladder/`, 73093f0bd558).
  - **Equal to simulation, every reading:** fullswitch (the 67-slot full exchange), arm, armclose, guards (84/84), and
    all seven S-11 arms. Below 1024 B or off 16-byte alignment the seal is refused with cause 29; at 1024/2048 B
    aligned it seals with every canary intact.
  - **sup-mtip:** the real CLINT timer interrupt escapes a supervised domain (kind 1, cause 7) twice, with MTIP still
    pending in the monitor, and the quantum escape follows. Exact except the reference spin count, a board timing
    quantity (122 against simulation's 243).
  - **sup-strip x5:** exploratory. It completes within its invariants. Its in-switch detector exists only in
    simulation.
  - **Repeats:** sup-escape v2, sup-quantum v2 (q16, q64) and call-retpc now stand at N = 3, identical every time;
    74 preemptions at both quanta.
- **Still open on silicon:**
  - sup-hostile (it needs a register-preserving recorder);
  - anything through the OpenSBI monitor or Linux. Next: the FPGA monitor's supervised_invoke, in the board lane.

## 2026-10-03 — upstream memory defects for the three app ports: 4 measured on the arms, 18/18 cells as pre-registered, 3 retractions

- **What was asked:** find memory defects (plain `malloc` **and** nested allocators) for memcached,
  tshark and FFmpeg on the upstream trackers, run them through the emulator, check they are caught.
- **Measured, all exactly as pre-registered** (predictions pushed before every build):
  - **tshark** fixtures 14/15 — ZigBee ZCL Touchlink (CVE-2026-95391) and http2 `GRegex` — on
    `level0`/`shrink`/`sublet`/`chunks`: **8/8 cells**. `ports/wireshark/app/results/2026-10-03-qemu-upstream-live-defects/`.
  - **memcached** fixtures 17/18 — `204019d` realloc cursor and `e779381` logger write-after-free — on
    all five arms: **10/10 cells**. `ports/memcached/app/results/2026-10-03-qemu-plain-heap-contrast/`.
- **The memcached pair is the one that changes an argument.** Fixtures 12–16 end their objects inside
  memcached's own allocators, so the runtime's `free` is never called and `sublet` is registered as the
  negative control that must RETURN — that silence is the nested-allocator blindness claim. 17/18 end
  theirs with the platform's `free`/`realloc`, and `sublet` faults. Same heap, same runtime, opposite
  outcome: the variable is **where the lifetime ends**, which makes 12–16's silence attributable.
- **Three retractions, all surfaced and recorded:**
  1. **`ops_dispatch` (FFmpeg fixture 25) is NOT live at our pin** — backported as `716d2a47c5`. The
     retracted method was *"Ancestry decides; the pinned tree proves"*: ancestry is insufficient for
     cherry-picking release tags, and the "proof" was a grep for a line *shape* that matched a site
     where the object is never freed. Replacement test, with both controls:
     `git log <tag> --grep='cherry picked from commit <sha>'` plus reading the enclosing function.
  2. **Neither tshark defect is reachable in the port's own binary** — http2's block is inside
     `#ifdef HAVE_NGHTTP2` which the port undefines (and it is preference-gated upstream), and the
     zbee dissector is not linked (0 symbols against 353). They are live in the **v4.6.8 source**;
     "live in the version we compile" and the triage doc's "reachable and not latent" are withdrawn.
  3. **Three attribution/control overstatements in the tshark bundle**: the load base was given as
     `pc − symbol` (`0xa01f0014`) and is `0xa01f0000`; a symbol-delta argument was **vacuous** (an
     algebraic identity); and "both controls fired in every boot" was false — fixture 5 ran in none of
     the four decisive FAULT boots.
- **Search yield, honestly:** Wireshark `v4.6.8..origin/master` is **4321** commits (I had searched 82
  on the release branch) → 24 lifetime-worded → 5 live in source → **1** new usable. FFmpeg: all 10
  remaining candidates read, **none usable**; its nested column is empty because the only nested-class
  commit was backported, and the four "HW data freed early" pool commits are destruction-ordering on a
  **VkSemaphore**, not pool storage. memcached: **0** fixed-upstream-after-the-pin, now verified
  two-sided (`origin/next`, which memcached actually develops on, is one unrelated commit ahead).
- **Instrument lesson worth carrying:** a lifetime-vocabulary grep over memcached's history returns 13
  commits and is **not** a superset of the 9 the subsystem search found — intersection 2, and three of
  the five built corpus cases would never have been found by wording. Treat such a filter as a sampler.
- **Still open:** memcached's nested column needs port work (`--disable-proxy --disable-extstore`
  everywhere, page mover not hooked); no native/ASan arm for the four new fixtures; the "revocation on
  free" mechanism is inferred rather than counted in the faulting runs, and the missing control is named
  in the tshark bundle.

## 2026-10-02 — Supervised CALL RUNS ON SILICON: quantum preemption with exactly-once resume, and fault-event delivery (bare M-mode, one run each)

- **The RTL lane's directed tests ran on the board** (`caplifive_supcall_36a641e0b.bit`), bare M-mode, JTAG-loaded at
  `0x80000000` with no OpenSBI. The tests are byte-identical to capstone-ariane `36a641e0b`; only a harness header
  differs. All three board reading vectors are **bit-identical to the RTL lane's simulation of the same tree**
  (113/113, 67/67, 8/8):
  - `sup-quantum`: 53 quantum preemptions, each resumed. The domain's 400 instructions net to **exactly 400**. The
    domain's x1, the monitor's resume count and its mie survived, and no trap was taken.
  - `sup-escape`: ecall, illegal, `csrw mepc` (CSR gate), `mret` (decoder guard) and a misaligned load each deliver a
    fault EVENT to the monitor with the right cause, epc and tval. The seal comes back SEALED.
    - It does NOT show the trap stripped: the no-strip mutant reads the same 67 values, because mepc, mcause and
      mstatus are walked. **Settled by the v2 tests** (capstone-ariane 7564c0945, repaired by the RTL lane;
      `tests/rtl-smoke/sup-bare-2026-10-02-v2/`, pre-registration committed before the boot): sup-escape v2 reads
      the domain's saved mcause slot as 0 in all five arms, and equals simulation 72/72. **The trap is stripped on
      silicon.** sup-quantum v2 (an acc := acc*7 + i ladder) is exact at both quanta. At q16 it equals simulation
      155/155. At q64 the board took 74 preemptions against simulation's 73: a pre-registration miss on a timing
      count, with every correctness reading exact.
  - `call-retpc`: correct values. This is not a demonstration of R-47 fixed on silicon, because the board's timing
    was not shown to create the trigger.
  Folder: `tests/rtl-smoke/sup-bare-2026-10-02/` (self-contained harness, images, pre-registration, audited result
  lines).
- **These two tests discriminate** against any bitstream without the extension, where `cssupervise` and the event
  CSRs do not exist. This is the first silicon evidence for the mechanism the delegated runtime needs.
- **Not yet on silicon:**
  - repeatability (one run each);
  - the MTIP escape, the guards, the hostile domain, and S-11's refusal;
  - anything through the OpenSBI monitor or Linux.
  The delegated runtime on silicon still needs the runtime lane's FPGA monitor, built to the plan's contract.
- **Instrument:** the runner's GDB halt fallback does not work on this board (`dmstatus` allrunning; the halt is not
  honoured). The UART report is the only channel.

## 2026-10-01 — Supervised CALL implemented in RTL (capstone-ariane `sup-call`), verified through step 7 in simulation; R-47 found and fixed; S-11 amended

- **The supervised-CALL platform extension** (the primitive the delegated application runtime needs on silicon)
  is implemented on `capstone-ariane` branch `sup-call` per `docs/plans/supervised-call-silicon.md` (revisions 1.1
  and 1.2): `cssupervise`, the escape at commit, the SAVE/RESTORE walks around the existing 8-register exchange,
  the quantum, the guards, the event CSRs (csupquantum 0x7C3, csupctl 0x7C4, csupstatus..csnodefree 0xFC0..0xFC4).
  Steps 2-7 of the ladder read as pre-registered in simulation (five synchronous faults, exactly-once resume
  under two quanta, a real timer interrupt escaping the domain twice, all ten guards, a hostile domain and a
  foreign RETURN), with ten mutant positive controls, an adversarial after-audit whose four findings are fixed
  with discriminating arms, and the 92-test neutrality sweep on the committed tree (91 identical, `revocation`'s
  phantom mid-switch traps gone). Lint at baseline. Commits: d38887426, 1dbf379b1, 727ea6e93, 03b70667e on
  `sup-call`, pushed, not landed on the submodule's shared branch. **Not synthesized, not on silicon**: synthesis
  is the lead's call; the FPGA monitor build (`supervised_invoke` with the CSR-based events, csupctl.resume
  before a resume, interrupts masked between cssupervise and CALL) is the runtime lane's.
- **R-47 (new, fixed on `sup-call`, OPEN on every bitstream):** CALL parks the wrong return pc when a jump or
  branch reaches issue before the dyn unit responds -- the caller resumes past that instruction, or
  mid-instruction after a compressed one. Nothing resident is hit only because the monitor's and the runtime's
  CALLs are followed by an `ldc` that holds issue. Registry entry on dev.
- **S-11 amended on dev** (the Anvil relational-precedence fold, read from the compiler's own grammar): QEMU now
  raises on an undersized SEAL (since 2026-09-26, minimum 528 B); the switcher's instance of the fold is fixed
  on `sup-call` and the full-mode slot map measured; a 64-byte seal makes the switch write 24 bytes past the
  capability's end (measured); `cssupervise` CANNOT check a seal's size (a SEALED capability carries no end
  bound in a register), so the 1 KiB minimum is SEAL's alone -- the resident monitor's 1536-byte seals are the
  guarantee until S-11 is fixed. A deterministic lint for the fold is on `sup-call`.
- **Two more decisions on the lead's delegation, same bitstream, audited:** a CALL of another seal while armed
  fails closed (trap, the arm persists across the trap and across every exception on a CALL head, as QEMU; one
  predicate for the block and the trap, debug mode included); S-11 is fixed in RTL at 1024 B / 16-B aligned with
  R-32's off-by-one corrected AND the sealed cursor set to the region's start -- the audit showed a SEALED
  capability's base IS its cursor, so the size check alone bounded nothing (every known sealer complies; QEMU's
  528 should be raised). Both tested with mutant controls; sweep4 on the final tree 91/92 identical to the unmodified RTL (revocation -3 phantom traps, as before) and 92/92 identical to the tree before the audit's changes. RETRACTED
  with it: "a SEALED capability reads back untagged after STC/LDC in simulation" and "status 1 is unreachable" --
  `sup-arm.S` had overwritten the seal with an integer before the store; repaired, the seal survives the round
  trip and the dead-node status (1) is measured.
- **Supervised CALL synthesized (2026-10-02):** `sup-call` 36a641e0b sealed by the synth lane, exit 0; routed loop arc
  identical to the resident's, LUTLP-1 = 0, WNS -8.189 in band, FF +1,090 explained row by row, two ORDER-like
  pre-registration misses with mechanisms (plan, step-9 result). Bitstream sha256 016219b2..., NOT flashed.
- **Supervised CALL confirmed on silicon (2026-10-03, the board lane):** the whole plan ladder except the hostile domain
  ran bare on the resident bitstream exact to simulation -- full switch, arm, the fail-closed arm, guards, all seven
  seal-size arms (S-11 fixed on silicon), MTIP, strip x5, N=3 repeats of escape/quantum/call-retpc; folder
  `tests/rtl-smoke/sup-bare-2026-10-03-ladder/`, plan step-10 result. Next: the FPGA monitor's `supervised_invoke`
  against the plan's CSR contract (the board lane), then the first `capstone-exec` program.
- **Resident bitstream: `caplifive_supcall_36a641e0b.bit`** (sha256 016219b2...5d045), flashed 2026-10-02 15:55 by the
  board lane on the lead's direct confirmation, name read back after the power cycle; the previous resident was
  `caplifive_r43_8f6a0af.bit` (the name the console reports). **ACCEPTED 2026-10-02 16:12-18:05: the R-43
  acceptance list a1..a10 reads as pre-registered on every row, on the same images as 2026-09-29**
  (`tests/fpga-repros/R43-revocation-cache-false-deny/results/board-36a641e0b.result-lines.txt`; audited, every number
  re-derived from the primary logs). Highlights:
  - live512 sums 17408 (R-43 holds), and P1 cell 6 completes with 112006 38bb59fd and the exact Sublet counts;
  - the R-35 probe still traps 25 at +0x4354 with the refusal record LATCHED, id 95, probe-DEAD;
  - the capability ladder's instret is identical on 17/17 rungs, and P1 moves -0.01..-0.05 %.
  **Identity shown by behaviour, not only by the console label:** a one-instruction domain reading csnodefree (CSR
  0xFC4, which exists only in 36a641e0b; on 8f6a0af98 the read is an illegal instruction) RETURNED 0xFFCD, inside the
  pre-registered range. That is the first silicon execution of the census CSR. **Read the acceptance as "no observable
  change on these ten workloads", NOT "inert until cssupervise"** (that wording was retracted before this landed).
  Several changes run without cssupervise, every boot:
  - commit drops exceptions while any domain switch is busy;
  - every CALL parks its pc per R-47;
  - every SEAL applies S-11's size/alignment check and sets cursor := start; the monitor's seals passed.
  The supervised path itself, and any input that would trip S-11 or R-47, have not run on silicon. Set
  `FPGA_BITSTREAM=caplifive_supcall_36a641e0b.bit` for every board run. Every Part A bundle so far (M1 v2, S1,
  repro322) was measured on 8f6a0af98, before this flash.

Minimal snapshot. Read first in every session.


---

Everything before 2026-10-01 is in
[`history/13-10-2026_01-00-00_state-journal-through-2026-09-29.md`](../history/13-10-2026_01-00-00_state-journal-through-2026-09-29.md).
