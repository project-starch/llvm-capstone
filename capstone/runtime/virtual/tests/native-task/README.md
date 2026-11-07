# Experimental Linux task-return backend

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
[Exact evidence and limitations](native-seek-qualification.json).

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
[Exact evidence and limitations](native-stat-qualification.json).

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
[Exact inputs, counts and limitations](native-locks-qualification.json).

This is a QEMU instruction result, not hardware timing or a full-SPEC verdict.
No ISA or QEMU changes are required. Exchange compatibility remains; next
consider metadata calls (`newfstatat`/`fstat`) using the same bounded workflow.
Older launch decoders disable direct routing on unknown feature bits; they
do not reject the whole program. Keep the new feature off for pinned older
SDKs when preserving their direct-I/O performance matters.


## Session teardown ownership

`native_session_destroy(c)` lives next to `native_session(c)` and owns the
existing session, bounce-buffer and native-context frees. `dom_context_destroy`
invokes it after race-fixture disarming and before delegated-resource and VM
context cleanup. The function is for final teardown only; it adds no reset,
idempotency, lock or reference-count semantics. Constructor failure unwind
remains local and unchanged. [Qualification](session-teardown-qualification.json).

The incremental build, both fast settings/ABIs, task exit/kill, private-tmp
policy errors, directory/dup/time, spawn and official GCC comparison pass in
104.0s under one 300-second deadline. Code/data/BSS sizes are unchanged;
no performance claim or new profiling/race campaign is made. Existing policy
admission failures are not targeted allocator-failure injection or a leak proof.


## Generic copy workspace reuse

`native_copy()` calls `cv_checked_copy_reuse()` with `n->copy` after both
`native_owner()` and `riscv_capstone_owns()` succeed. The old wrapper's task
ownership check is preserved. Scratch is initialized per copy; the same checked
copy engine retains holds, rechecks, fault retry and partial-result semantics.
No busy flag, lock or reference count was added: synchronous per-task calls
cannot overlap this workspace, and the test racer dispatches no native copies.
Other allocating copy paths, vectors and metadata replies remain unchanged.
[Qualification](copy-reuse-qualification.json).

The build/regression/GCC/measurement gate passes in 163.4s, and all 164
race scenarios in a separate 52.2s gate. Each has one 300-second budget.
Production and race images use separate destinations; the build scripts refuse
to overwrite a frozen image with different bytes. The final production Image
is retained locally as well as on the test host.

Matched against the preceding authorization-reuse kernel, ABBA order with two
repeats per kernel/setting, same 1 MiB resident-file copy and 4 KiB requests:

| ABI / dispatch | Before: kernel attempts/call | After | Reduction |
| --- | ---: | ---: | ---: |
| exchange / generic | 18680.4 | 18313.3 | 1.97% |
| direct / generic | 14500.1 | 14132.8 | 2.53% |
| exchange / fast enabled | 18680.6 | 18312.0 | 1.97% |
| direct / fast enabled | 8594.1 | 8589.5 | 0.05% |

Exchange uses the generic dispatcher in both settings. Fast direct already
reused scratch and is the control. Measurements use the unchanged
`measure-helper-reuse.py`, with tracing disabled. These are attempted QEMU
instructions, not cycle timings or a GCC speedup. Full SPEC remains separately
qualified on its historical frozen image.


## Shared authorization workspace

`cn_authorize()` and the direct fast path share `native_authorize_reuse()`.
Every invocation resets the existing kmalloc-backed workspace, preserving its
physical descriptor alignment and all offset, ownership and access checks.
This removes one allocation/free pair per generic authorization. It does not
merge task/context identities, change the general checked-copy wrapper or
change the ISA. [Final qualification](authorization-reuse-qualification.json).

The final combined build/regression/GCC/measurement gate passes in 160.6s;
the separate race profile passes all 82 scenarios in 25.2s. Each gate has
one 300-second deadline. The earlier race script path error and superseded
production artifact are recorded; the final Image is independently qualified.

Matched old/new kernels in ABBA order, two repeats per kernel/setting, resident 1 MiB copy,
4 KiB requests (512 read/write calls per window), tracing disabled:

| ABI / dispatch | Before: kernel attempts/call | After | Reduction |
| --- | ---: | ---: | ---: |
| exchange / generic | 19042.4 | 18680.3 | 1.90% |
| direct / generic | 14859.0 | 14500.0 | 2.42% |
| exchange / fast enabled | 19038.1 | 18678.9 | 1.89% |
| direct / fast enabled | 8589.8 | 8594.0 | -0.05% |

Exchange uses the generic dispatcher in both settings.
Fast direct already reused scratch; its near-zero change is a control, not a
meaningful speedup. These counts include kernel work inside the marked loop,
not setup/output validation; they are not hardware timing or a GCC speedup.
`measure-helper-reuse.py` accepts `--qemu`, `--baseline` and `--candidate` image
directories (each with Image, System.map and rootfs.ext2), `--firmware`,
`--plugin`, `--probes` (copy-exchange.dom/copy-direct.dom) and fresh `--out`.
Use absolute paths and run it inside `run-quick-gate.py`; the hashed final plan
in the qualification record provides the exact commands.


[Direct-helper qualification](direct-helper-qualification.json): the build and
focused gate, including original GCC and its official comparator, pass in 93.1
seconds. This is not a full SPEC rerun or a measured runtime speedup.

## Direct internal Linux helper interface

The built-in adapter calls its required Linux helpers directly. The shared
`kernel-capstone-helpers.h` is installed as
`include/linux/capstone_native_helpers.h` and included by both callers and
helper definitions. Spawn/time retain their existing typed headers. Missing
helpers now fail compilation/linking rather than returning a missing-symbol
fallback at runtime; syscall authority, lock placement and resource cleanup
are unchanged.

`install-native-task.sh` applies `kernel-native-helpers.patch` **after** the
existing native task/service patches and installs the staged header. This final
patch removes twelve obsolete module exports; the remaining export APIs and
flat-table support are outside this change. Use the installer on a fresh,
prepared tree. For an incremental tree already containing the earlier patches,
apply the final helper patch once, refresh the staged adapter and install its
`capstone_native_helpers.h` into Linux `include/linux/` before building.
The earlier prerequisite memory/mprotect/GCC patches remain unchanged so their
historical standalone profiles remain reproducible.

## Iterative feedback: at most five minutes

Use `run-quick-gate.py` to supervise a configured build and regression plan.
The single monotonic budget includes build time, QEMU lock waits and all test
steps; it defaults to 300 seconds and refuses larger values. A timeout stops
the owned process group, preserves logs and returns 124 with `TIMEOUT`, never
`PASS`. A failed command stops later steps. Remaining steps are `NOT_RUN`.
Commands must not daemonize or escape their process group.
The [qualified example](quick-gate-qualification.json) completes the incremental
build, native regression gates and original GCC comparison in 83.6 seconds.

Source the environment, then run a fresh plan/output pair:

```sh
source capstone/tests/capstone-test-env.sh
python3 capstone/runtime/virtual/tests/native-task/run-quick-gate.py \
  --plan "$CAPSTONE_TMP_ROOT/native-quick/plan.json" \
  --out "$CAPSTONE_TMP_ROOT/native-quick/result" --budget-seconds 300
```

The plan is a JSON list of objects with a unique `name`, an `argv` string list
and optional `cwd`. Commands execute directly; there is no implicit shell or
environment-variable expansion. Put the actual build command in the plan so it
shares the deadline. Use fresh output paths for all constituent gates.

Choose tests for the change: native task/I/O, directory/dup/time, private-tmp
policy and spawn lifecycle gates, plus the unchanged original GCC test and its
official output comparator. Selecting GCC alone means copying that entry from
the original comparator manifest into a separate subset manifest, preserving
all commands and references, and using that same subset for execution and
validation. Record both original and subset manifest hashes. Do not report
other benchmarks as tested. Full Perl/x264/SPEC qualification is a separate
release/background campaign and does not block iterative feedback.

`test-quick-gate.py` exercises deadline, process-group cleanup, early failure
and invalid-plan behavior with real short-lived subprocesses. A configured
quick-gate PASS is not a full SPEC or hardware qualification.


The current branch removes the legacy device ioctls, control FD and scalar
trampoline. [Native-only cleanup qualification](native-only-cleanup-qualification.json)
records the new built-in kernel and passing task/I/O/resource/spawn gates.
Its full ten-entry SPEC rerun is pending; `spec-final-qualification.json` remains
the complete pre-cleanup result on its original kernel. Historical ioctl-based
measurements and launchers below require their pinned historical revision.


For the cumulative diff against dev and the hardware work, start with the
[RTL handoff](../../../../docs/plans/native-task-rtl-handoff.md). Its SPEC gate
is separate from the earlier short performance comparisons.

Branch `native/linux-task-integration`. This is a working first implementation,
not completion of the full [acceptance plan](../../../../docs/plans/native-linux-task-integration.md).
The [transition contract](../../../../docs/design/virtual-capstone/native-linux-tasks.md#linux-61-first-backend-transition-contract)
explains context tracking, exec failure and synchronous lifetime rules.

The kernel owns the protected task context and mm namespace. `.dom` exec installs
no scalar trampoline mapping or context FD. Architecture return enters CSRUNV
and loops through Linux return work after each transition. Existing checked-copy,
exchange ABI, confinement and providers remain. No linear ownership instructions
are introduced. The model's 5 ms quantum supplies escape opportunities; this is
not direct hardware timer delivery. Caught handlers, full restart semantics,
general clone, caller-replacing exec, SMP and native audit/seccomp/tracing remain outside this slice.

## Native child processes

The native task backend now accepts the existing packed `SPAWN` ABI for
capability images and supports exact-child/any-admitted-child `wait4` with
options zero or `WNOHANG`. The [process qualification](spawn-qualification.json)
records independent parent/child tests in both SDK profiles. This does not
implement general fork/vfork, caller-replacing exec, job control, or caught
signal handlers.

`kernel-native-spawn.patch` adds a dedicated bootstrap whose exact entrypoint,
request ownership and clone flags are checked by `copy_thread`. The temporary
shared mm is never exposed to scalar userspace: successful bootstrap must have
a fresh mm and an attached native context. The request/completion is refcounted;
exit inside exec also completes it. Parent interruption cancels and reaps the
exact child before returning. This can wait for uninterruptible filesystem I/O.

The frontend snapshots the complete bounded spawn block through checked
access. A child gets private Linux files/fs/sighand, the original immutable
directory grants, its current cwd, and only admitted descriptors. File actions
run in order against that evolving child state. CLOEXEC is applied by real exec
and reconciled with the admitted descriptor set. `dup`/`dup3` also preserve the
FD boundary; replacing an unadmitted descriptor is refused.

Execution uses a pinned file through Linux's `AT_EMPTY_PATH` bprm lifecycle,
retaining execute permission, noexec, write exclusion and LSM checks. Native-only
format dispatch refuses ordinary scalar ELF images, scripts and interpreter
chains. No path is checked and then reopened by name. In-place exec remains
refused because the architecture callback still owns the caller's context.

Child bookkeeping holds PID objects, not just numeric IDs. Wait-any scans only
those admitted children, so a Linux child inherited across initial exec is not
implicitly granted. Autoreaped/detached PID entries are collected before new
spawns; zombies retain their status until wait. Like Linux wait4, reaping occurs
before checked status copy-out. Failed spawn reply publication cancels and
reaps the newly created child before the caller can resume.

A shell is an explicit optional executable grant: `capstone_vm.dom_shell=PATH`
pins that file at initial admission and maps `/bin/sh` execution to it. Descendants
inherit the same pinned grant. `capstone_vm.dom_null=1` separately grants the
exact `/dev/null` device; neither option grants a directory. Both default off.
The [hush port](../../../../ports/hush/app/README.md) supplies a noninteractive
capability shell using native spawn and existing NOMMU state serialization.
Its own qualification and the original SPEC comparisons are separate gates.

Build the fixtures with `build-spawn.sh` and `build-spawn-lifecycle.sh` after
sourcing the test environment. The latter needs both SDK compiler paths and
`CROSS_COMPILE`; run `spawn-driver` on each resulting `.dom` image in the native
QEMU guest. The scalar driver is a test supervisor, not an application syscall
or spawn fallback. Raw artifacts remain external. Bootstrap-only ptrace races
and cancellation during blocked bootstrap filesystem I/O are explicitly
**not** established by the post-exec parent-death and ptrace tests.

## Private temporary files

`capstone_vm.dom_local_tmp=1` maps logical `/tmp` to a pinned `.capstone-tmp`
child of the original launch directory. Trusted launch setup must create that
directory with mode 0700 and matching ownership before native exec. Admission
rejects symlinks, mount crossings, unsuitable filesystems, an initial root of
exactly `/tmp`, and simultaneous `dom_wide_grants`. The option defaults off;
`run-spec-tests.py --local-tmp` creates the reserved directory and refuses a
pre-existing entry. No ambient `/tmp` or device authority is added.

Children inherit the pin even after a cwd change. The original directory grant
remains the authority boundary: moving the scratch directory outside it refuses
subsequent alias, relative and directory-FD lookups and `getcwd`. Existing regular
file descriptors retain their object authority. This does not provide atomic
revocation during racing renames. `getcwd` reports the physical directory name;
the alias does not create a chroot namespace. The separate `/dev/null` grant
remains available when explicitly enabled.

The [local temporary-file qualification](local-tmp-qualification.json) records
four modes in both SDK profiles: disabled, enabled, enabled with `/dev/null`, and
conflicting wide grants. The enabled modes verify actual `tmpfile` I/O,
unlink-while-open, child inheritance, host sentinel preservation, escape
refusals, pinned-object replacement and move-out behavior. This supports libc
implementations whose `tmpfile` uses a fixed `/tmp` pathname without changing
libc or application sources. On the same kernel, a focused run of the unchanged
original `t/io/perlio.t` and Perl binary completes all 48 TAP results: 38 passes,
seven original skips and three original TODO failures, with no unexpected
failures. The previously failing temporary-file assertions 23–29 all pass
without skip or TODO annotations. This direct focused invocation does not
establish a passing full SPEC workload or resolve the separate pack-serialization
failure.

```sh
source capstone/tests/capstone-test-env.sh
# Set CAPSTONE_EXCHANGE_CC, CAPSTONE_NATIVE_CC, CROSS_COMPILE and platform paths.
bash capstone/runtime/virtual/tests/native-task/build-local-tmp.sh "$PROBES"
python3 capstone/runtime/virtual/tests/native-task/run-local-tmp.py \
  --qemu "$QEMU" --images "$IMAGES" --firmware "$FIRMWARE" \
  --probes "$PROBES" --out "$CAPSTONE_TMP_ROOT/native-task/local-tmp-run"
```

The scalar fixture is a trusted test supervisor that prepares and moves the
directories; native application requests still use the confined frontend. The
runner requires the common QEMU lock and preserves the complete serial logs,
positive assertion counts, command lines and artifact hashes.

## Native directory, duplication and timer services

The [service qualification](directory-time-qualification.json) records 94 directory,
36 descriptor-duplication and 33 timer/select checks in each SDK profile, all
passing on the same frozen kernel. These services close observed syscall gaps;
they do not establish a passing SPEC Perl comparison.

`native-directory-dom.h` resolves checked pathname snapshots under the existing
pinned grants. It supplies mkdirat, directory removal, path chmod/truncate,
symlink creation, final-link metadata/readlink, regular-file linkat with flags
zero, and getdents64 on admitted descriptors. Opening or traversing a symlink
remains refused. Hard links retain Linux's protected-hardlink, mount and LSM
checks; path chmod refuses device pins. Directory iteration preserves Linux's
record layout, continuation offsets, shared file position and partial results.
The duplication tests verify the requested destination return value, descriptor
authority, replacement, CLOEXEC and same-descriptor behavior.

`kernel-native-time.patch` and `native-time-dom.h` use Linux timers and readiness
polling with checked kernel buffers. The guest tests cover relative/absolute
expiry, invalid timeouts, pipe readiness, descriptor refusal and temporary mask
restoration. Interrupted output marshalling has separate host tests; caught
signals and automatic restart remain outside this qualification. The dispatcher
passes 299 sanitizer checks and ten negative controls. Fresh installation of all
six kernel patches and 20 qualification-helper tests also pass.

Build and run all three fixtures with explicit SDK and platform paths:

```sh
source capstone/tests/capstone-test-env.sh
# Set CAPSTONE_EXCHANGE_CC, CAPSTONE_NATIVE_CC, PROBES, QEMU, IMAGES and FIRMWARE.
bash capstone/runtime/virtual/tests/native-task/build-directory.sh "$PROBES"
bash capstone/runtime/virtual/tests/native-task/build-dup.sh "$PROBES"
bash capstone/runtime/virtual/tests/native-task/build-time.sh "$PROBES"
python3 capstone/runtime/virtual/tests/native-task/run-directory.py \
  --qemu "$QEMU" --images "$IMAGES" --firmware "$FIRMWARE" \
  --probes "$PROBES" --dup-probes "$PROBES" --time-probes "$PROBES" \
  --out "$CAPSTONE_TMP_ROOT/native-task/directory-services-run"
```

The runner uses the shared QEMU lock and refuses a missing test environment.
`--runner PATH` selects another current `run-staged.py` when the build host has
separate tool copies. The qualification retains prior harness failures and the
corrected zero-size readlink fixture expectation: musl returns zero through its
private one-byte probe, while the raw syscall returns EINVAL. Raw logs, binaries
and source snapshots remain outside Git under
`$CAPSTONE_TMP_ROOT/native-task/directory-time-qualification-final`.

## Native cwd, pipe and signal repair

The first repair after the full SPEC gate adds `chdir`, live `getcwd`, bounded
parent traversal, transactional `pipe2`, and default/ignore signal dispositions.
The [qualification record](cwd-process-qualification.json) records the new
kernel, source hashes and QEMU checks. It does **not** qualify subprocesses or
turn the historical 9/10 SPEC result into a full pass.

The launch directory stays an immutable authority root; changing cwd does not
change that grant. Relative paths and directory FDs resolve against pinned
objects. Parent traversal rejects grant escapes, symlinks and mount crossings;
ambiguous concurrent renames can return `EAGAIN`. `getcwd` follows trusted
renames, rejects a cwd moved outside its grant, and never returns an unchecked
outside pathname. Linux's cwd is kept synchronized after successful changes.

`pipe2` reserves both descriptor numbers within RLIMIT_NOFILE and the frontend's
1024-FD bound before creating files. It copies the result through checked access
and, for exchange calls, publishes the reply before installing either file.
Copy failures release both reservations and file references. This avoids
accidentally closing an existing descriptor or releasing unrelated POSIX locks.
The direct ABI retains its exact capability/scalar register shape and epoch
validation. Both paths support only `O_CLOEXEC | O_NONBLOCK` flags.

The SIGACTION service accepts default/ignore dispositions through Linux's
`do_sigaction`; handler capabilities remain in the SDK and caught handlers
return `EOPNOTSUPP`. SIGKILL/SIGSTOP cannot be changed. Only the child-disposition
flags affect default/ignore actions; handler-only flags have no effect. This is
not protected caught-signal delivery, cancellation, or full restart support.

Install with `install-native-task.sh`, which now also applies
`kernel-native-process.patch`. No QEMU/ISA, compiler, SDK, or benchmark changes
are required for this repair. Build the probes after sourcing the test environment:

```sh
source capstone/tests/capstone-test-env.sh
# Set CAPSTONE_EXCHANGE_CC, CAPSTONE_NATIVE_CC and CROSS_COMPILE to built SDKs/tools.
bash capstone/runtime/virtual/tests/native-task/build-cwd.sh "$CAPSTONE_TMP_ROOT/cwd-probes"
bash capstone/runtime/virtual/tests/native-task/build-process.sh "$CAPSTONE_TMP_ROOT/process-probes"
```

Run `cwd-driver ./cwd-exchange.dom` and `cwd-driver ./cwd-direct.dom`, then
`process-driver ./process-exchange.dom` and its direct counterpart in the native
QEMU guest with narrow grants and one hardware slot (the same platform flags as
the full SPEC helper). The cwd driver creates and concurrently renames its own
temporary grant. The process driver exercises real signals, both descriptor
limits, 128 partial payload failures per path, and 64 fatal metadata-publication
failures per path. It checks descriptor reuse and Linux file-count growth.
Expected metadata failures kill the protected child with SIGSEGV before resume;
they are deliberately negative cases, not benchmark failures.

The [focused original SPEC rerun](spec-process-repair.json) passes GCC's exact
reference and Perl makerand. Perl test.pl now finds the tests and reports all
317 as unable to start (`SPAWN -> ENOSYS`); it still exits zero, and the official
output comparison correctly fails. This confirms repair of the cwd blocker
while exposing the subprocess blocker. The other eight entries were not rerun
on this repaired kernel, and the original full-campaign record stays unchanged.

That repair did not yet supply subprocesses. The subsequent native child
process implementation above closes the spawn/wait slice; general clone and
caller-replacing exec remain separate work. The original SPEC inputs and
comparator remain the application acceptance criterion.

## Full SPEC test-input qualification

This gate covers the ten entries in the existing C port, with every original
`test` invocation and official output comparator. It is not a CPU2017 reportable
run, a whole-suite port, or a performance measurement. Exit status alone does
not qualify a benchmark. In particular, the Perl test harness can exit zero
with incomplete output.

The [current final qualification](spec-final-qualification.json) reports
**10 PASS, 0 FAIL** on kernel `1bd9b8ecf702`, using all 22 original benchmark
invocations, four guest image validators and all 25 official comparisons.
Perl revision 07 uses the pinned [source-selection recipe](../../../../ports/perl/provenance/README.md#build-and-run-the-explicit-v2-package).
The final Perl run completed in 4,831.9 host seconds; this is not a hardware
performance estimate. The preceding failures below are preserved history.

The [historical qualification record](spec-test-qualification.json) reports **9 PASS,
1 FAIL** across 22 completed invocations and 25 official output comparisons
(24 pass, one fails). GCC, lbm, ImageMagick, nab, both specrand entries, mcf,
all twelve xz inputs and the complete 156-frame x264 test pass. ImageMagick and
x264 include their guest image validators and official host comparisons.
The x264 encoder took about 65 minutes on the test host; this is not a hardware
performance estimate.

Perl's makerand output passes, but test.pl produces incomplete output despite
exit zero. The [confirmed diagnosis](spec-perl-blocker.json) is a refused native
`chdir` followed by an exception hidden by the outer test wrapper. Pipe/popen
and signal services also remain incomplete; a cwd-only patch is not sufficient
to declare this benchmark supported. No production code or benchmark sources
were changed to obtain these results.

The initial GCC attempt omitted assembly from the exported artifacts; that
harness failure is retained, followed by a passing retry. The final GCC,
ImageMagick and integer specrand rows select fresh runs through the repository
helpers on a separate host with identical QEMU/kernel/firmware/rootfs hashes.
The other rows use the original campaign; x264's three image checks run in a
separate fresh guest against the encoder's exact output hashes. An interrupted
remote transfer caused by scratch inode exhaustion was corrected before the
official comparisons. These infrastructure issues are separate from Perl's
reproducible application failure.

Raw evidence is kept outside Git as
`$CAPSTONE_TMP_ROOT/native-task/spec-test-qualification-evidence.tar.gz`, with
a copy in the build host's task scratch directory. SHA-256:

```
ec686cf64691a6c76ee721db01c2bed068c18746f76b5c42444cd7c05c959da1
```

It contains original preparation inventories, binaries/inputs, invocation and
comparator records, serial logs, outputs, the separate x264 image run, Perl
diagnosis and tool snapshots. Duplicate staged trees, sparse stage disks and
original command files containing environment snapshots are omitted. Parsed
commands and their source-file hashes are retained. This licensed evidence
archive is separate from the source-only RTL handoff bundles.

Four helpers keep licensed sources and generated artifacts outside Git:

1. `prepare-spec-tests.py` verifies original input checksums and frozen adapted
   sources, hashes existing application objects, and relinks with the selected
   SDK. Specrand is compiled from its original sources. This is not a fresh
   compiler or complete application rebuild.
2. `setup-spec-validation.py` asks the original SPEC harness for fresh commands
   and comparators, verifies the invocation arguments, and regenerates x264's
   input with the supplied host decoder. Its `no_md5` control is not a checksum;
   byte identity with that fresh generated input is checked separately.
3. `run-spec-tests.py` runs the applications and image validators in QEMU with
   one hardware slot, the native task/checked-copy backend and narrow grants.
   It retains indexed exit receipts, serial logs, image data and comparator
   output files. It fails on missing cases or nonzero results.
4. `validate-spec-tests.py` independently checks execution evidence and runs
   every official host `specdiff`. It fails on missing benchmarks, images,
   references or output mismatches, even when the application exited zero.

Source `capstone/tests/capstone-test-env.sh` before these commands. Select fresh
scratch directories for each attempt; the preparation, setup and runner refuse
to overwrite existing results. The build host needs licensed CPU2017 1.0.2,
the existing port's build directories and `frozen.json`, the matching SDK and
the ordinary host x264 input decoder. Original SPEC setup roots must identify
one run per benchmark; provide narrower roots first if multiple variants exist.

```sh
helpers="$PWD/capstone/runtime/virtual/tests/native-task"
python3 "$helpers/prepare-spec-tests.py" \
  --spec "$SPEC_ROOT" --builds "$SPEC_BUILDS" --sdk "$NATIVE_CC" \
  --frozen "$FROZEN_SOURCES" --run-roots "$ORIGINAL_RUN_ROOT" \
  --out "$CAPSTONE_TMP_ROOT/spec-native/prepared"

# On the licensed host, after sourcing that SPEC installation's shrc:
python3 "$helpers/setup-spec-validation.py" \
  --spec "$SPEC_ROOT" --prepared "$CAPSTONE_TMP_ROOT/spec-native/prepared" \
  --cc "$NATIVE_CC" --ldecod "$HOST_LDECOD" \
  --out "$CAPSTONE_TMP_ROOT/spec-native/official"

python3 "$helpers/run-spec-tests.py" \
  --prepared "$CAPSTONE_TMP_ROOT/spec-native/prepared" \
  --comparators "$CAPSTONE_TMP_ROOT/spec-native/official/comparators.json" \
  --qemu "$QEMU" --images "$NATIVE_IMAGES" --firmware "$FIRMWARE" \
  --out "$CAPSTONE_TMP_ROOT/spec-native/execution" --timeout 7200

python3 "$helpers/validate-spec-tests.py" --spec "$SPEC_ROOT" \
  --results "$CAPSTONE_TMP_ROOT/spec-native/execution/runs" \
  --comparators "$CAPSTONE_TMP_ROOT/spec-native/official/comparators.json" \
  --out "$CAPSTONE_TMP_ROOT/spec-native/validation.json"
```

For split build/QEMU hosts, transfer the prepared tree and comparator manifest
unchanged to the QEMU host, then return `execution/runs` to the licensed host
for validation. Comparator reference paths refer to that licensed host. Keep
both inventories and all serial/receipt files, not just stdout. Host elapsed
times include scheduling and the shared QEMU lock; they are not speed results.

The qualification tools have a self-contained regression suite:

```sh
python3 "$helpers/test-spec-qualification.py" -v
```

Its synthetic fixtures test missing/duplicate/nonzero completion evidence,
wrong commands, stale or missing outputs, generated-input checks and image
reference provenance. Compiler, SPEC setup and QEMU execution are mocked; it
does not replace the real application/output gate.

## Build and activation

Use a disposable Linux 6.1 tree with this branch's
`../../unpinned/linux-6.1-unpinned.patch`, `../../kernel-native-mprotect.patch`
and `../../kernel-native-gcc.patch` already applied, plus its prepared platform
configuration. Matching QEMU is pinned at `dd041bac5ad9c145c3872a50804ca714a26b1c88`.
Do not apply the experimental patch to the measured baseline tree.

```bash
source capstone/tests/capstone-test-env.sh
# KERNEL_BUILD: disposable prepared Linux tree; CROSS_COMPILE: Linux compiler prefix.
bash capstone/runtime/virtual/install-native-task.sh "$KERNEL_BUILD"
make -C "$KERNEL_BUILD" ARCH=riscv CROSS_COMPILE="$CROSS_COMPILE" olddefconfig
make -C "$KERNEL_BUILD" ARCH=riscv CROSS_COMPILE="$CROSS_COMPILE" -j12 Image
```

The installer applies all six native task, files, process, spawn, directory and
time patches, then stages matching runtime sources
as a built-in driver. It enables `RISCV_CAPSTONE_TASK`, disables SMP and selects
generic virtual CPU accounting with forced user context tracking. The Kconfig
option defaults off and execution additionally requires `capstone_vm.native_task=1`.
For an external matching QEMU source checkout, set `CAPSTONE_QEMU_SOURCE` before
staging. Kernel test variants use `make CV_TEST_HOOKS=1 CV_RACE_TEST=1 Image`.
These deliberate fault/race hooks are absent from the ordinary build.

Boot arguments (no insmod):

```
capstone_vm.native_task=1 capstone_vm.dom_binfmt=1
capstone_vm.native_dispatch=1 capstone_vm.checked_copy=1
capstone_vm.native_slot_transport=1 capstone_vm.unpinned=1
capstone_vm.supervisor_slots=1
```

Use the shared `../../run-staged.py`, ordinary OpenSBI, 4 GiB guest memory and
matching QEMU CPU properties:

```
x-capstone-supervisor-slots=1,x-pcrel=false,x-capstone-raw-moves=true,x-capstone-test-csrs=false,x-capstone-checked-copy=true,x-capstone-context-transport=true
```

Build these fixtures with `build.sh OUTPUT`, setting `CAPSTONE_NATIVE_CC` to the
copyable SDK compiler and `CROSS_COMPILE` to the Linux compiler prefix. Build
current `../native-integration` probes with its build.sh. Stage `gate.sh`,
`compute.dom`, `task-driver`, `probe.dom`, `blocked.dom`, `entry-driver` and the
external GCC artifacts `gcc-test.dom`, `t1.c`, `reference.s` in one directory.
GCC is SPEC 502.gcc_r test/t1.c, `-O3 -finline-limit=50000`; no suite is vendored.
Require `NATIVE_TASK_EXIT:0`, zero fixture failures and the exact-output marker;
run-staged's completion marker alone is not a passing oracle.

For deliberate admission refusal, use the test kernel, write `1` to
`/sys/module/capstone_vm/parameters/task_refuse_once`, then execute `compute.dom`.
It must die with SIGSEGV; subsequent ordinary qualification must still pass.
The invalid action raises an S-origin exception before domain activation,
exercising the early context-tracking recovery and exception-table fixup.

For the existing race suite, use **32 slots** in both kernel and QEMU; its
injected racer requires an additional context. Apply each row of
`../native-race/scenarios.txt`, run `pipe-fill race.dom` with that row's arguments,
and use the unchanged `../native-race/judge.py` oracle. One-slot blocking/task
transport is tested separately by `entry-driver` and `task-driver`.

## Verified evidence

[Qualification record](qualification.json): 51 ABI/authority/protection checks,
10 entry/blocking checks, compute STOP/CONT/KILL, ptrace refusal, absence of
trampoline/control FD, twelve repeated blocked-task kills, admission-exception
recovery, reference-exact GCC, and all 41 copy-race scenarios (17 negative
controls). Compute fixture reports 0.501139 seconds user time and zero rounded
system time. Scalar Linux tools continue to run. The external-module variant also
builds; shared native-core C90 declaration warnings remain.

The initial stale probe binary expected ENOSYS for a now-confined pathname;
the rebuilt current source passes all 51 checks. The initial one-slot race suite
refused its racer setup; this was not counted as passing. Full register/FP/bounds
transport matrices, all raw-platform negatives, exhaustive leak/failed-exec race
injection and the remaining Linux policy integration gates are still open.

## Matched preliminary GCC comparison

`measure-gcc.py` takes paths to QEMU, Image/System.map/rootfs, firmware, capprof,
profrun, GCC, input and reference; see `--help`. It runs fresh guests in
ioctl/task/task/ioctl order. The only mode change is `capstone_vm.native_task`;
both use the same built-in kernel, config, binary, input and output oracle.

[Four-run record](gcc-comparison.json), mean attempted instructions:

| Entry | Total | Kernel including built-in adapter |
| --- | ---: | ---: |
| Existing ioctl | 50,928,947.5 | 26,954,892 |
| Architecture task return | 51,037,165.5 | 27,063,220.5 |

Task return is **0.2125% more total work**, about **0.4019% more kernel work**
in this small test. Other work is effectively unchanged. This first version
returns through Linux work checks after every transition and accounts domain
execution as userspace; it does not yet optimize/batch those transitions.
The result supports the architectural change, not a performance win. Do not
compare these absolute counts directly with older 54M runs: this kernel has
changed SMP/accounting configuration and builds the adapter in-tree.

Raw logs, profiles, kernel images/configuration and Argon review are archived at
`$CAPSTONE_TMP_ROOT/native-task/native-task-evidence.tar.gz`
(mirror `p12:/var/tmp/capstone-native-task/native-task-evidence.tar.gz`).
SHA-256: `b8286ed6a5252441d0589fa1a9631fc20b468d508a7e064a797cbd19047008fd`.
Astra xhigh reviewed entry/context tracking, task lifetime and ptrace/exec races;
Argon independently reviewed the patch. Review findings were checked against
Linux 6.1 source; neither review substitutes for the runtime gates.


## Syscall path and direct-register measurement

[Full factorial record](syscall-comparison.json): 128 valid profile windows across
eight fresh guests, two observations per entry/ABI/case/fast-path setting.
All return-count checks and output-byte checks pass, with no kernel diagnostics.
`build-syscalls.sh OUTPUT` builds the existing budget fixture for ordinary RV64,
Capstone exchange, forced scalar exchange and direct register calls.
`measure-syscalls.py --help` lists external artifacts; run with `--fast 1` and
`--fast 0` to separate specialised dispatch from the direct argument ABI.
The latter still changes parsing/copy/provider paths, so it is not a pure cost
measurement of register moves alone.

Mean system-wide attempted instructions **per call**, new task backend:

| Operation | Ordinary Linux | Exchange | Registers, generic dispatcher | Registers, specialised paths |
| --- | ---: | ---: | ---: | ---: |
| getpid, forced kernel request | 601 | 10,715 | 5,264 | 1,345 |
| read, 1 byte | 1,697 | 15,908 | 9,603 | 4,205 |
| write, 1 byte | 2,667 | 16,929 | 10,560 | 5,194 |
| read, 64 KiB | 39,996 | 215,131 | 86,462 | 69,969 |
| write, 64 KiB | 80,050 | 293,606 | 126,287 | 113,427 |

The existing direct-register specialised path saves 73.56% for 1-byte read,
69.32% for 1-byte write, 67.48% for 64 KiB read and 61.37% for 64 KiB write
against exchange on this backend. Kernel work for 1-byte read falls from
13,845 to 4,175 attempts (about 70%); most of the reduction is kernel work,
with application marshalling also disappearing. Checked copies and bounds/
permission enforcement remain. This is a probe ABI comparison; GCC/musl has
**not** been switched wholesale to that ABI.

Important scalar control: ordinary Capstone musl `getpid()` uses launch metadata,
measuring 255 instructions and **zero kernel instructions** here. Calling
`syscall(SYS_getpid)` would still meet that SDK shortcut. The forced-exchange
probe calls `__capstone_delegate_ints` to bypass the shortcut deliberately.
The 10,715-to-1,345 comparison is therefore a transport comparison, not an
argument to replace cached getpid: preserving that cache is cheaper still.

The task-entry change by itself remains a cost: the direct getpid path takes
609 attempts through the old ioctl service loop versus 1,345 through task return
with userspace accounting. The cheap repeated service loop need not exit its
ioctl on every request. These are separate effects: direct argument dispatch
helps substantially; changing ownership/return integration alone does not.

Small cases use 100 calls per window; 64 KiB cases use one call. Buffer pages
are touched before profiling, stdin is a prefilled pipe, writes target a tmp
file, and setup/diagnostic output are outside the window. Counts include loop
and wrappers, plus system-wide interrupt work. Only two repeats per cell were
run; they are preliminary instruction counts, not latency/cycle estimates.
The ordinary static RV64 reference uses the same patched kernel, not stock
Linux with a separately matched libc/compiler. Preserve those distinctions
when using the table to choose the next optimisation.

Syscall logs, profiles, input hashes and probes:
`$CAPSTONE_TMP_ROOT/native-task/syscall-evidence.tar.gz`
(mirror `p12:/var/tmp/capstone-native-task/syscall-evidence.tar.gz`), SHA-256
`91b3e92a7fb0eeec926ba9c04fadba93cb34d3e432cf87211565b315e18ac00f`. The kernel/model artifacts are the same as the task GCC record.

## Opt-in musl small-I/O migration

`CAPSTONE_MUSL_NATIVE_IO=ON` now routes `read`, `write`, `pread64` and `pwrite64`
through the direct-register ABI **only for original requests of 1–65536 bytes**.
Selection happens once before effects. A direct short count/error is final:
no refill, fallback or replay. Zero and oversized original requests, including
small trailing chunks, retain exchange semantics. Checked copies, bounce buffers,
buffer capability authority, identity/time caches and the existing completion
lifecycle remain. This is still a limited prototype, not all-native libc.

The SDK option defaults off and is rejected for physical applications. Build
both comparison SDKs from this source with identical options and libc archive,
changing only this option. The SDK manifest reports `native_io`. Reinstall this
branch's built-in adapter and boot with `capstone_vm.native_io=1` in addition to
the task flags above. It requires `native_task=1` and advertises launch wire v3:
a 16-byte `{abi=1, features=1, max_io=65536, reserved=0}` extension after the
88-byte header. The application image stamp remains v2. New SDKs accept launch
v2 without direct routing; unsupported v3 feature contracts disable routing.
No speculative ecall probes are used. **The global v3 boot option requires rebuilt
applications**; old startup decoders reject v3 and can fail after exec's point
of no return. Keep it off for mixed old/new SDK deployments.

Integration exposed an ABI detail: scalar operations in the pinned model write
the low 64 bits but can leave stale upper payload. A computed `strlen` count
therefore failed strict native shape checks despite constant-count probes passing.
The libc bridge now canonicalizes actual scalar fd/count/offset values in an
explicitly cleared register. LCC type checks preserve malformed capability
arguments for rejection. The assembly stub also clears the number and selector
registers. Kernel upper-word, raw-tag and buffer authority checks are unchanged.
Tagged fd/offset regressions pass; this fixture does not exercise a tagged count
with an admitted low cursor. Moving its bounded capability to such a cursor
clears the tag under compressed-bounds rules, so that is not a valid negative test.

[Qualification](musl-qualification.json) covers v3 specialised/generic dispatch
and v2 fallback. Each includes file/boundary/dirty-scalar tests, short pipe/EAGAIN,
51 authority checks, 20 syscall-coverage checks, blocking and task lifecycle.
Both 41-scenario pipe and regular-file race suites pass the unchanged oracle,
including all 17 deliberately broken controls. Regular-file short results remain
short; libc does not refill after the injected copy fault. Host ASan/UBSan launch
tests cover extension/table truncation, offset boundaries and unknown features.
Astra xhigh and Gemini Argon reviewed the implementation and scalar correction;
findings were checked against pinned model code and generated assembly.

Reproduce fixtures and gates (set external artifact/compiler paths):

```bash
source capstone/tests/capstone-test-env.sh
# CAPSTONE_EXCHANGE_CC and CAPSTONE_NATIVE_CC: matching SDK capstone-cc wrappers.
# CROSS_COMPILE: ordinary Linux compiler prefix.
bash capstone/runtime/virtual/tests/native-task/build-musl-io.sh "$PROBES"
python3 capstone/runtime/virtual/tests/native-task/run-musl-io.py io \
  --qemu "$QEMU" --images "$IMAGES" --firmware "$FIRMWARE" \
  --probes "$PROBES" --out "$OUT"
# Repeat into fresh output directories with --fast 0 and with --feature 0.
# Modes race and race-file require the CV_RACE_TEST/CV_TEST_HOOKS kernel.
# The runner selects one slot for IO/lifecycle, 32 for the injected racers.
```

### GCC result and route attribution

Relink the **same GCC object files and final link arguments** with each SDK.
`measure-musl-gcc.py --help` lists the required external artifacts; the two GCC
arguments are `--gcc` (exchange) and `--gcc-direct`. It runs four fresh guests in
exchange/direct/direct/exchange order with the same task kernel and feature
announcement. Profiling has route tracing disabled. All four exact-output checks
pass. [Measured record](musl-gcc-comparison.json), mean attempted instructions:

| Musl routing | Total | Kernel including adapter |
| --- | ---: | ---: |
| Exchange | 51,037,402.5 | 27,063,328 |
| Small direct I/O | 51,058,377 | 27,085,867.5 |

That is **+0.0411% total, +0.0833% kernel**: no observed GCC speedup in this small
preliminary sample. Userspace/firmware other work decreases 1,565 attempts. Do not
interpret interrupt-sensitive totals or two repetitions as a latency result.

Repeat the command with `--routes` and a new output directory for two separate
**diagnostic** guests. [Route record](musl-gcc-routes.json): each arm records 31
service completions; only one 69-byte `read` moves to registers. Three `writev`
requests, pathname/status/limit/signal and other recorded services stay exchange.
The trace is not an exhaustive accounting of allocator/mapping/lifetime work;
`count` is the transport's third argument, not always bytes (for writev it is the
vector count). The direct-I/O probe savings therefore do not predict a speedup
for this GCC test. Further work should attribute the remaining kernel work by
service and phase before expanding the ABI solely on these microbenchmarks.

Raw final logs, profiles, binaries, kernel images, input hashes, source diff and
Argon reviews are archived at
`$CAPSTONE_TMP_ROOT/native-task/musl-io-evidence.tar.gz`
(mirror `p12:/var/tmp/capstone-native-task/musl-io-evidence.tar.gz`), SHA-256
`08df4188697990d7f14556c04e1ba61bf606267aeb9be8f6014ff186beccdac1`.
The root filesystem and firmware are external artifacts identified by hashes;
the archive does not vendor SPEC sources or duplicate the base root filesystem.

### Short SPEC test-input comparison

[Matched four-program record](musl-spec-short-comparison.json) extends the same
SDK comparison to ImageMagick, lbm and nab, selected from workloads previously
estimated to take at most five minutes. Each new benchmark has two complete
fresh-guest runs per variant, with order reversed for the second pair. GCC reuses
its unchanged four-run record above. All three new programs were relinked from
identical objects with the two existing SDKs; no runtime change was made.

| Test input | Exchange total attempts | Direct total attempts | Total change | Kernel change |
| --- | ---: | ---: | ---: | ---: |
| GCC | 51,037,402.5 | 51,058,377 | +0.0411% | +0.0833% |
| ImageMagick | 112,356,920 | 112,314,757.5 | −0.0375% | −0.2332% |
| lbm | 6,657,715,796 | 6,651,726,854.5 | −0.0900% | −0.6355% |
| nab | 7,242,697,263.5 | 7,242,441,522.5 | −0.0035% | −1.3880% |

Values are means, with kernel including the built-in adapter. All twelve new
runs pass the original SPEC comparators: lbm absolute tolerance `1e-7`, nab
relative tolerance `0.01`/`skipreltol=2`, and ImageMagick's image validator plus
its `0.01` relative-tolerance output comparison. Image validation is outside the
profile window. GCC retains its exact-output oracle. All eight staged data files
for the three new programs match the original SPEC input/compare files by hash.

The initial nab staging omitted its nested input directory and failed before
completing the benchmark; those attempts are retained in the raw archive and
excluded from the table. Corrected runs copy the complete directory tree.
Elapsed host durations include shared-runner lock waits and must not be used
as benchmark timings. These remain preliminary instruction measurements, not
hardware speedups or reportable SPEC scores. The largest observed reduction in
total work is only 0.09%; this slice works but does not materially reduce total
work for these test inputs.

Raw profiles, serial logs, output validation, input audits, object/binary hashes
and the linking/running/validation scripts are in
`$CAPSTONE_TMP_ROOT/native-task/spec-short-evidence.tar.gz`
(mirror `p12:/var/tmp/capstone-native-task/spec-short-evidence.tar.gz`), SHA-256
`dacb6be869ac79b416a8557e37faa3b6dfc9ad5ca39f01015677f289f53f8be6`.
Scripts record the external artifact paths used for this run; the QEMU, kernel,
SDK and GCC artifacts are the same as the preceding musl migration record.
No benchmark sources or data files are committed to the repository.

### Syscall-heavy workload through real musl

`musl-copy.c` copies the same resident 1 MiB file using ordinary musl `read` and
`write`, with 64-byte, 4-KiB and 64-KiB buffers. One application object is linked
against the exchange and direct-I/O SDKs. File creation/population, buffer setup,
full output comparison, final offset/length checks and diagnostics are outside
the measured loop. Both files are populated before measurement, so this measures
CPU work in the cached I/O path, not disk latency. Loop/return-count checks remain
inside the window; capability checking and kernel copy rules are unchanged.

[Four-guest record](musl-copy-comparison.json): exchange/direct/direct/exchange,
reversed case order in the second pair, twelve valid windows, every return count
and full copied output verified. Means of two observations per cell:

| Buffer | read/write calls | Exchange total attempts | Direct total attempts | Total reduction | Kernel reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64 bytes | 32,768 | 555,982,127 | 189,516,342 | 65.91% | 64.96% |
| 4 KiB | 512 | 12,367,353.5 | 4,813,633.5 | 61.08% | 53.62% |
| 64 KiB | 32 | 4,989,767.5 | 2,670,841 | 46.47% | 20.85% |

This confirms that the libc integration retains a substantial saving when the
workload spends its time on eligible I/O. The 64-byte loop uses about 2.93 times
fewer instruction attempts, not a measured 2.93x hardware speedup. Buffering also
matters: either 64-KiB variant performs much less total work than either 64-byte
variant. These results explain why the limited call mix in the short SPEC tests
does not show a comparable whole-program gain.

Reproduction, with the same SDK and external platform artifacts as above:

```bash
source capstone/tests/capstone-test-env.sh
# CAPSTONE_EXCHANGE_CC / CAPSTONE_NATIVE_CC name the matching SDK wrappers.
bash capstone/runtime/virtual/tests/native-task/build-musl-copy.sh "$PROBES"
python3 capstone/runtime/virtual/tests/native-task/measure-musl-copy.py \
  --qemu "$QEMU" --images "$IMAGES" --firmware "$FIRMWARE" \
  --plugin "$CAPPROF" --probes "$PROBES" --out "$OUT"
```

Raw profiles, serial logs, binaries, common object and source scripts:
`$CAPSTONE_TMP_ROOT/native-task/musl-copy-evidence.tar.gz`
(mirror `p12:/var/tmp/capstone-native-task/musl-copy-evidence.tar.gz`), SHA-256
`8b502739784acb0c673d413673782c5129dc4c6194039d18392a23fa260f3e32`.
Platform artifacts are identified by hashes and are unchanged from the musl
migration record. No wall-clock or hardware-latency claim is made.

### SQLite speedtest1 admission: file workload blocked

The existing ordinary-program SQLite 3.22.0 port was built with its Unix VFS,
normal allocator, original defaults and existing 32-phase result oracle.
One engine object and one speedtest object were linked with the two SDKs.
No journal, locking or sync option was changed to make the test pass.

[Admission record](sqlite-admission.json), diagnostic tracing enabled:

| Workload | Exchange | Direct small I/O |
| --- | --- | --- |
| `speedtest1 --size 1 st.db` | Exit 1, first CREATE TABLE, zero completed oracle phases | Same failure |
| `speedtest1 --size 1 :memory:` | All 32 phases match native oracle | All 32 phases match native oracle |

The file run reaches open/stat/read successfully, including one direct-register
read in the direct SDK. Its first POSIX lock request uses the existing
`CAPSTONE_NR_FCNTL_LOCK` wire service (`0xc0de0003`); native dispatch returns
`-ENOSYS`, and SQLite reports `disk I/O error` for the first CREATE TABLE.
`unlinkat` is also refused at startup. The buffered ioctl query used to configure
diagnostic output is refused but nonfatal. Source inspection additionally finds
no native providers for scalar fcntl, fsync/fdatasync or ftruncate. These latter
calls are prerequisites to audit, not a claim that this early failed run reached
them or discovered every remaining dependency.

The older delegated runtime's SQLite qualification does not establish coverage
on this task backend. The memory controls establish that this build and both
SDKs can execute the complete SQL workload; they do not validate file locking,
journaling, crash durability or a file-backed speedup. No performance number is
reported for these diagnostic runs. A successful diagnostic gate exit is not a
successful benchmark exit; the admission record checks each program separately.

Before comparing the file workload, add real native file services with descriptor
confinement and checked flock marshalling, including competing-writer lock tests
and correct release/error behavior. Preserve journal and sync semantics. Then
repeat file-backed admission against all 32 oracle phases before collecting
untraced performance windows and separate route diagnostics.

Raw traces, binaries, common objects, build command, host reference outputs and
analysis script are archived at
`$CAPSTONE_TMP_ROOT/native-task/sqlite-admission-evidence.tar.gz`
(mirror `p12:/var/tmp/capstone-native-task/sqlite-admission-evidence.tar.gz`), SHA-256
`ac9d7fed30701ba8d4728c5faf9545340b29b94ac962a5b7370412557f133484`.
Source hashes identify the existing prepared SQLite app port. Platform artifacts
are unchanged from the musl migration record. The source archives are external.


### Native file services and SQLite comparison

The earlier admission failure is now closed for the size-1 file workload. The
[Linux patch](../../kernel-native-files.patch) exports narrow held-file helpers
for real Linux POSIX locking, bounded descriptor duplication, truncate and fd
metadata. The [provider](../../module/native-files-dom.h) adds fcntl scalar and
32-byte flock operations, fsync/fdatasync, ftruncate, fchmod/fchown; unlinkat uses
the existing pinned directory grant and component walk, now with the path LSM
hook. These are kernel changes plus the built-in runtime provider, not a new
userspace trampoline or a replacement syscall calling into ioctl.

GETLK requires a readable and writable flock buffer; SETLK/W only read the
snapshot and never copy it back after changing locks. Linux retains ownership,
conflict reporting, interruptible waiting, deadlock detection and close-release
semantics. vm_lock is released across blocking Linux operations. F_DUPFD and
F_DUPFD_CLOEXEC reserve below the 1,024-descriptor grant limit before installing
anything: allocating an out-of-range duplicate and closing it would release all
of the caller's locks on that inode. Native lseek now takes f_pos_lock, matching
read/write on shared open-file descriptions. Existing Linux VFS and LSM checks
remain; unsupported fcntl commands fail. Caught signals, automatic restarts, OFD
locks and arbitrary device ioctls are not newly supported.

[File-service qualification](files-qualification.json):

- Two real domain/Linux contention runs check conflicting and shared locks,
  GETLK ownership/ranges, blocking wakeup, SIGKILL of a blocked waiter (confirmed in /proc/locks), duplicate
  exhaustion retaining locks, and closing another same-inode fd releasing them.
- Exchange/direct domain fixtures pass 69/77 checks, with 56 Linux supervisor
  checks each. They also check shared offsets/flags, CLOEXEC, truncate zero-fill,
  sync failures, metadata updates, inherited-fd refusal, and unlink confinement
  including a symlink escape and unchanged outside sentinel. The direct fixture
  checks read-only, undersized, null and invalid-command raw flock requests.
- The production dispatcher passes 275 ASan/UBSan host checks and all ten
  existing negative-control mutants. SETLK/W tests revoke input authority after
  snapshot and deliberately mutate the backend buffer, detecting any copyout.
- Existing musl I/O (170/176 regular-file checks and four pipe checks per SDK),
  51 authority/ABI checks, 20 GCC-service checks, entry and task lifecycle gates
  pass with the final production kernel. The new installer also passes on a
  fresh prepared source fixture. Astra and Gemini Argon reviewed the changes;
  findings were checked against Linux sources and the actual tests.

[SQLite result](musl-sqlite-comparison.json): same SQLite 3.22.0 engine and
speedtest objects as the original admission, two SDK variants, size 1, default
main testset, file database and unchanged journaling/sync/locking options. Four
fresh guests run exchange/direct/direct/exchange. Every run matches all 32
stock-host-reference SQL phase outputs; two additional diagnostic guests also
match. No diagnostic tracing is enabled in the measured guests.

| Mean attempted instructions | Exchange | Direct small I/O | Reduction |
| --- | ---: | ---: | ---: |
| Whole process window | 657,856,152.5 | 630,508,392.5 | 4.1571% |
| Kernel, including built-in adapter | 433,043,686.5 | 410,435,927 | 5.2207% |
| Other: userspace and firmware | 224,812,466 | 220,072,465.5 | 2.1084% |

The window includes process startup, application work, file operations and SQL
oracle calculations. Database/journal files live on guest tmpfs: this establishes
CPU-work savings, not storage latency, hardware speedup or crash durability.
Only two repeats per arm were taken. This is exchange versus direct **within the
same native-task kernel**, not a stock Linux processor baseline.

Diagnostic routes show exactly 1,460 reads and 823 writes moving from exchange
to registers. The 5,862 lock requests, 2,922 newfstatat calls, 2,283 seeks,
1,463 fstat calls, 88 fsync calls and other services still use exchange arguments.
All lock and sync calls return success. Normal absent-path probes return ENOENT;
one open is denied by confinement and one buffered diagnostic ioctl remains
ENOSYS in each arm. Both are nonfatal and are recorded, not silently called full
Linux coverage. The remaining service/argument costs are the next attribution
target before broadening the direct-musl migration.

Reproduction uses external prepared sources, the two existing SDKs and platform
arguments documented above. Nothing from the benchmark suite is vendored.

```bash
source capstone/tests/capstone-test-env.sh
# Set CAPSTONE_EXCHANGE_CC, CAPSTONE_NATIVE_CC and CROSS_COMPILE as above.
bash capstone/runtime/virtual/tests/native-task/build-files.sh "$PROBES"
python3 capstone/runtime/virtual/tests/native-task/run-files.py \
  --qemu "$QEMU" --images "$IMAGES" --firmware "$FIRMWARE" \
  --probes "$PROBES" --out "$FILES_QUALIFICATION"
bash capstone/runtime/virtual/tests/native-task/build-sqlite.sh "$SQLITE_SOURCE" "$SQLITE_BUILD"
python3 capstone/runtime/virtual/tests/native-task/measure-musl-sqlite.py \
  --qemu "$QEMU" --images "$IMAGES" --firmware "$FIRMWARE" --plugin "$PLUGIN" \
  --profrun "$PROFRUN" --build "$SQLITE_BUILD" --reference "$HOST_REFERENCE" \
  --out "$SQLITE_COMPARISON"
# Repeat the last command with --routes and a fresh output directory.
```

Raw final profiles, serial logs, test fixtures, new kernel image/configuration,
source snapshots, prepared SQLite sources/objects, host reference and review
outputs are archived at
`$CAPSTONE_TMP_ROOT/native-task/native-files-sqlite-evidence.tar.gz`
(mirror `p12:/var/tmp/capstone-native-task/native-files-sqlite-evidence.tar.gz`),
SHA-256 `b60fadc864d0d0165f1181f8b6615e663332d0b7e0dfee29544f457beff49aac`.
Unchanged QEMU, firmware, rootfs and SDK provenance are those of the preceding
musl migration/admission records; the comparison records their hashes.
