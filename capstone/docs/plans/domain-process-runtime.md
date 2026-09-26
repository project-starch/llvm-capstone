# Domain applications as Linux commands

Status: QEMU application lifecycle implemented, 2026-09-26. The design below
records the original goals; the implementation and verified limits are in
[applications.md](../../runtime/applications.md). The matched platform changes
live on the `domain-process-runtime` branches in LLVM, Buildroot, OpenSBI,
capstone-sbi and QEMU. This does not claim FPGA implementation or full POSIX.

## Delivered implementation

| Layer | Implemented and tested |
|---|---|
| Application | Shared musl CRT, versioned argv/env/cwd block, immutable ELF snapshot, inherited streams, ordinary main/exit |
| Platform | Protected CALL continuation, typed return/preemption/fault, no-yield cancellation, guest operand faults instead of host assertions |
| Ownership | Per-open driver owner retained by dup/fork/VMAs; forced close cleanup; retained roots above domain and transferred heap; scrub and reusable slots |
| Resource reuse | Bounded retained physical pool with explicit live/cache accounting; stale-tag sweep before node reuse; emergency cleanup reserve and recoverable node exhaustion |
| Host/guest | Installed Buildroot package, QMP lifecycle, SSH shell/streams, token-checked cancellation, waitpid result collector, explicit restart |
| Port build | Shared CMake application/SDK interface; generic compiler driver; Perl's three private execution/compiler/entry files removed; Perl and mruby exercised |

The managed path avoids the old stack-pop/CPMP lifetime bugs through its own
stable ownership and revocation protocol. **M-7/M-9 are not declared globally
fixed for legacy loaders.** Legacy global IDs cannot be mixed with managed
execution in one module lifetime. The retained pool is not memory returned to
the Linux page allocator; its capacity and accounting are explicit.

Follow-up platform work is FPGA/ISA implementation and validation of the
supervised continuation and node-reclamation protocol. Follow-up application
work is gradual migration of remaining legacy entry protocols and actual missing
OS services demanded by upstream tests. The Perl upstream subset already exposes
unsupported fork and another domain fault; those are not silently counted as
passing tests. A complete POSIX/fork/thread ABI and a hostile-code security audit
remain outside this delivery.

## Outcome

Boot a development guest once, enter its ordinary Linux shell, and run real
Capstone applications. A supported application fault ends its Linux launcher
with a signal and leaves the shell usable. Normal exit, fault and cancellation
all converge on resource destruction, so the next application can start without
a reboot. The execution layer has no application-name dispatch table.

Inside the guest, the intended interface is:

```sh
capstone-exec ./perl.dom -e 'print "hello\n"'
capstone-exec ./python.dom script.py
capstone-exec ./sqlite3.dom example.db 'select 1;'
capstone-exec ./filter.dom < input.txt > output.txt 2> errors.txt
```

These examples describe the interface, not the current completeness of any
application port. The guest command must also work without our host CLI, for
example from an interactive console or a Linux test harness. Application build
systems and upstream test harnesses remain their own projects. The target is
ordinary program execution; trace formats and replay drivers are not its API.

## Original baseline and gaps (before this implementation)

| Existing code | What it establishes | Remaining work |
|---|---|---|
| [Generic recovery](../../runtime/domain-faults.md) and its [recorded tests](../../runtime/tests/fault-recovery/README.md) | On a matching QEMU, a cooperative CALL/REGION_SHARE client can fault, return to Linux and cause actual SIGSEGV in its launcher; a later healthy client runs in the same boot | Integrate application entry/yield paths; complete destruction; independently trusted fault termination |
| [musl entry](../../ports/musl-capstone/runtime/start-musl.S) and [HostCall runtime](../../ports/musl-capstone/runtime/hostcall.c) | Resumable execution of real applications across service requests | This is a different continuation ABI from the generic recovery entry; installing its recovery define alone is insufficient |
| [Host services](../../ports/musl-capstone/runtime/host_service.h) | Shared file-service implementation, including the bounce buffer required by observed 9p I/O failures | Separate stdout/stderr, working stdin, descriptor semantics, reusable library ownership |
| [Former Perl application entry](../../ports/perl/musl/README.md) | Argument/environment setup before main, currently using fixed files | Shared startup ABI with environment established before constructors, no application-specific argument files |
| [Guest runner](../../tests/runtime-qemu/run-domain-smoke.py) | Linux boot, shared-directory mount, module loading and commands | It owns a whole boot and powers off afterward; no persistent session interface |
| [Shared host support](../../ports/common/host/port_support.py) | Input staging, identities, results and execution serialization | Extract reusable host infrastructure without introducing another test framework |

In this baseline, the musl bridge has no stdin service, sends stdout and stderr
through WRITE_STDOUT, keeps file positions in the domain, and implements local
pipe queues with restricted nonblocking behavior. These are real compatibility
limits: a new command name cannot provide ordinary Linux I/O semantics by itself.

The driver/library currently have create/call and region operations, but closing
the device does not destroy the domain. Region reuse already has recorded defects
([issue registry](../ref/ISSUES.md), M-7 and M-9). The current table and ownership
model must be repaired, not hidden behind larger CMA reservations. Record the
actual monitor, module and QEMU revisions used by implementation tests; directory
names or an older result bundle do not identify the running platform.

## Architecture and language choices

```text
development host: Python CLI ----- VM start/status/console (QMP + serial)
                     |
                     +----------- ordinary guest execution (SSH when enabled)
                                      |
guest Linux shell / upstream harness --+--> capstone-exec (C, one Linux process)
                                              |
                                  Linux execution library + host services (C)
                                              |
                                  per-execution driver context + monitor
                                              |
                                  shared domain CRT (C + small assembly entry)
                                              |
                                  application main(argc, argv)
```

| Layer | Choice | Reason |
|---|---|---|
| Monitor, kernel driver, domain CRT | C with assembly only for machine state transitions | These are the existing ABI/toolchain boundaries; capability-aware code must use the established compiler |
| Linux guest launcher and execution library | C, compiled by the ordinary RISC-V Linux toolchain | Direct reuse of libcapstone, host services and signal handling; no interpreter needed in the base guest |
| Host VM management | One typed Python package and one CLI entry point | Existing host tooling already uses Python; process, file, SSH and QMP orchestration can share a tested implementation |
| Builds | Existing CMake, upstream builds and Buildroot packaging | A launch command consumes installed/built artifacts; it does not become another build system |
| Shell | Interactive use and minimal compatibility wrappers | No new shell state machines, per-application boot loops or log-based process protocols |

Rust is a reasonable language for a new conventional Linux service, but adopting
it here would add a build/toolchain integration while the existing C boundary
still needs repair. It does not make the monitor's ownership or cancellation
semantics correct automatically. We have not established a Rust purecap runtime
path in this proposal. Start with the two existing implementation languages;
keep the C interface narrow enough that a future Linux component could use it
without changing domain binaries. C parsers at the boundary require explicit
length/overflow checks, bounded allocation and malformed-input tests.

Do not start with a guest daemon or a framework of port plugins. One Linux
process per application already provides an owner, file descriptors, a working
directory, a wait status and a unit to terminate. A small VM supervisor may be
needed on the host to own a detached QEMU, drain serial output and hold its lock;
it must not become the guest application scheduler.

## The process contract

### Startup and executable identity

`capstone-exec PROGRAM [ARG...]` receives ordinary argv, environment, cwd and
standard descriptors. Its shared CRT calls the application's normal main and
performs the existing TLS, constructor, atexit and stdio-exit work in the correct
order. Argument and environment storage must exist before constructors run.
Return from main and explicit exit use the same normal termination path;
fault termination does not invoke application destructors.

Define a versioned startup block with fixed-width lengths, offsets and bytes,
passed in an explicitly granted region. Preserve empty arguments, whitespace,
newlines and argv[0]; enforce an explicit total size limit. Do not send raw host
pointers or reconstruct domain capabilities from integer addresses. The CRT
derives pointers from its granted capability after validating bounds. Pointer
arrays are created on the domain side with the required capability alignment.

Extend executable resource metadata from the existing domain requirement
declaration: startup ABI, runtime service requirements, stack/data needs and
declared additional grants. Keep allocation policy at launch time and enforce
platform limits. Match the host services, CRT, ioctl and monitor versions before
creating resources. Preserve the current image geometry until a separately
validated change, including capability representability, permits changing it.

The command consumes any compatible executable, without a port manifest. Legacy
entry protocols retain their old explicit loaders until migrated. Avoid a chain
of guessed ABIs. Direct execution via binfmt_misc or interpreter wrappers can
follow only after executable identification is unambiguous; explicit launch is
the first interface.

### I/O and operating-system services

Use Linux descriptors as the backing objects for supported domain descriptors.
The launcher grants access through a per-execution handle table, never an
unchecked domain-supplied Linux fd. Import stdin, stdout and stderr independently,
including closed descriptors. Keep application bytes separate from diagnostics
and control records: no LT-RESULT lines on application stdout.

Add versioned stream read/write operations alongside existing positioned file
operations. A pipe or terminal needs read/write, not pread/pwrite. Define short
I/O, EOF, EINTR, nonblocking behavior, duplication/closure and shared file offsets;
the existing domain-local offset model must not silently approximate dup or
inherited open-file descriptions. Preserve the bounce buffer for mappings that
Linux cannot pin. Retain snapshot-and-validate handling of each HostCall request.

Delegate cwd, path resolution and supported file operations to the launcher's
Linux context. A normal developer launch has that process's file authority;
capability memory isolation is not a filesystem sandbox. A restricted-service
policy, if requested later, must enforce grants explicitly rather than claiming
that a shared directory confines arbitrary paths.

Move production HostCall definitions out of probe directories into a versioned
runtime interface; initially leave forwarding includes for existing probes.
Preserve old wire semantics rather than changing opcodes underneath old binaries.
Unsupported services fail explicitly. Existing no-op timers or restricted pipe
behavior must be advertised as such and must not satisfy a requirement for full
timer or pipe semantics.

### Faults, exit and cancellation

Keep three separate channels: application exit status, execution failure and
platform/transport failure. A normal exit 139 is not a SIGSEGV. A failed create
is not an application fault. The interactive launcher propagates normal exit
status and, for a delivered capability fault, terminates with the chosen signal
after runtime cleanup. SIGSEGV is the existing policy; record original cause and
PC separately. Reporting must not be able to block required fault termination.
Bind a fault record to the executed image hash and load address and retain its
matching debug image for symbolization. A Linux launcher core dump is not a
domain backtrace. Keep this diagnostic path common to every application.

Adapt recovery deliberately to the musl yield path. Save the unique caller
continuation across every entry/resume; cover faults before the first HostCall,
after many yields, during initialization and during region delivery. A faulted
domain never resumes its application body. Do not apply a C longjmp using the
faulting program's possibly damaged stack. The existing standalone recovery is
the starting evidence, not proof that start-musl.S is already protected.

The long-term monitor API returns a typed execution event: yielded request,
normal completion, fault or cancellation. Fault provenance must come from trusted
state; an application-written result or the cooperative sentinel alone cannot
prove a hardware exception. Model faulted/quarantined separately from destroyed.

Interactive Ctrl-C, timeout and SIGKILL require a trusted path out of a domain
that never yields. A signal pending in Linux cannot repair an ioctl that never
returns. Establish monitor timer/preemption behavior and cancellation races before
promising bounded interruption. Resume-to-service and cancel must serialize, and
a cancelled execution must not be re-entered by a late completion.

Cooperative recovery can deliver a useful first shell demonstration. Reliable
termination independent of application state requires a monitor-owned escape and
saved caller state, and may expose QEMU/ISA prerequisites. The existing CIH is an
interrupt scheduler, not automatically a synchronous-fault caller continuation.
Audit that boundary before selecting a trap-routing change. Kernel/monitor failure
remains a platform failure, and QEMU validation is not FPGA validation.

## Ownership and destruction

The proposed lifecycle is:

```text
NEW -> PREPARED -> RUNNING <-> YIELDED
                      |
                  EXITED / FAULTED / CANCELLED
                      |
                   DESTROYING -> DEAD
```

Creation failure unwinds everything acquired so far. A destroy operation may
request cancellation of a running domain, but must wait for quiescence before
reusing memory. Destruction is idempotent at the public interface.

Create an execution context owned by an open driver object, rather than relying
on global arrays and a process-global library fd. The context tracks domain
storage, region grants, ownership relationships, mappings and pending calls.
File duplication, fork/exec, mappings that outlive an fd and close-on-exec need
explicit lifetime rules; closing one descriptor is not proof that no reference
remains. The kernel owns forced cleanup when a launcher dies. A userspace cleanup
callback improves normal error handling but cannot cover SIGKILL.

The monitor must first prevent re-entry, unwind the current caller safely and
invalidate access derived from the execution's owned resources. Then reclaim
domain slots, region/CPMP state and backing pages, respecting live kernel mappings
and borrowers. Linux must not free pages while a domain can still address them.

Track authority transfer separately from backing-storage ownership. A
REV_TRANSFERRED grant has no ordinary retaining revoke handle today; a launcher
cannot reclaim it simply by calling the existing release_region. Define the
monitor's destruction authority over the execution's resource tree. Initially
reject grants escaping that tree unless an explicit lifetime transfer is
implemented. Test children, transferred heaps and retained sibling grants.

The stack-shaped region table cannot be the long-term reclamation interface for
independently ending processes. Repair M-7/M-9 and support stable handles with
generation/lifetime validation, or an equivalent model that does not require
unrelated later allocations to disappear first. Distinguish domain/region handle
generations from capability revocation-node reclamation: one does not fix the
other. Stale handles and stale capabilities must not authorize a reused allocation.

Expose accounting sufficient to test allocated pages, live domains, live regions,
mapping references and revocation metadata. Fixing CMA leaks alone leaves finite
tables and revocation nodes as possible reboot limits. Where the platform retires
identities permanently, report the bound and exhaustion explicitly. A bounded
soak run is evidence of reuse, not proof of unlimited lifetime.

## Persistent VM and host tooling

The first implementation is `capstone-vm` under `capstone/runtime/host`, with
`up`, `shell`, `run`, `exec`, `status` and `down`. The guest launcher and shared
CRT are described in [applications.md](../../runtime/applications.md).
The requirements below informed the now-implemented session transport,
destruction and trusted cancellation paths.

The host interface is a small set of commands such as `capstone vm start dev`,
`capstone vm shell dev`, `capstone vm status dev` and `capstone vm stop dev`.
An optional `capstone exec --vm dev -- PROGRAM ...` invokes the same guest
launcher. There is no application-specific test subcommand or build dispatch.

Buildroot installs the matching driver, launcher and service library once. Its
init setup loads the module and establishes the development share. Initially the
ordinary serial shell is enough. Add a standard SSH server/virtio network setup
for automated execution after the guest configuration is verified; use QMP for
VM lifecycle and serial for console/debugging. Avoid application output as a
machine-control protocol or shell-prompt scraping as an execution result.

Use direct argv APIs locally. SSH command construction must have one tested
POSIX-shell quoting implementation; do not concatenate port-supplied fragments.
An optional per-invocation C waitpid collector records actual child signal/exit
status and writes a separate atomic result file. SSH's numeric exit status alone
cannot distinguish ordinary exit 139 from signal death. This collector belongs
to the launcher package and need not be a permanent guest agent.

A session records QEMU, firmware, kernel, module and runtime identities, CPU
options, boot/CMA settings and sharing configuration. Reuse only a compatible
session. A mismatch in a named interactive session is an error with an explicit
restart action, not permission to kill its other jobs. Preserve the repository's
single QEMU lock for the session lifetime initially; expose its owner and stop
operation. Concurrent guests require a separate resource/lock policy.

Keep the root image immutable with per-session writable state and private
per-invocation artifact directories under CAPSTONE_TMP_ROOT. Do not overwrite a
running application's staged binary when rebuilding. Record completed runs and
failed attempts separately. Do not silently rerun a started application after a
disconnect: its file writes may already have happened. Restart a broken VM
explicitly or under a caller-selected policy; report the interrupted run first.

## Delivery sequence and acceptance

1. **Shared application entry and bounded shell recovery.** Build one C launcher
   and shared musl startup/recovery path; boot manually to Linux. Run a small
   main-based contract program and at least two existing real application ports,
   including one interpreter and a different application. Include a fault after
   HostCall yields and a healthy application afterward. Use actual waitpid signal
   evidence and a live shell. This milestone explicitly retains current resource
   limits and does not claim infinite-loop cancellation.
2. **Execution ownership, trusted stop and destruction.** Implement versioned
   monitor/driver/library operations together in isolated component worktrees.
   Test normal exit, initialization failure, fault, timeout and launcher SIGKILL;
   test out-of-order lifetimes and allocation-failure rollback. Repair M-7/M-9.
   A proposed 1,000-launch mixed sequence must exceed the old resource limits and
   show stable live-resource accounting after warmup, with a healthy final run.
   Test stale authority after reuse and preserve an unaffected sibling. Finite
   retired-node growth is accounted separately and cannot be labeled leak-free.
3. **Normal shell I/O.** Validate empty/newline-containing arguments, environment
   in constructors, cwd, independent stdout/stderr, stdin EOF, closed descriptors,
   short I/O, large 9p I/O and blocked pipes. Prove bounded Ctrl-C/timeout both
   while servicing I/O and in a no-yield loop. Gate each promised behavior rather
   than advertising a complete POSIX implementation.
4. **Installed guest tools and persistent host sessions.** Package the runtime
   with Buildroot, implement the Python CLI and optional SSH execution, then
   verify reuse, configuration mismatch, host-CLI disconnect and explicit restart.
   Guest shell execution remains usable independently. Keep fresh-boot mode for
   platform regressions and measurements requiring it.
5. **Port migration and upstream tests.** Convert real applications to the shared
   CRT and launcher. Let a native guest harness invoke the target interpreter via
   a standard executable wrapper where supported. Keep test fixtures and expected
   outputs upstream. Add further OS services only with their actual semantics;
   external launch alone does not implement fork inside the target application.

Ordinary Linux services should supply future descriptor, process and networking
functionality where possible. Spawning a fresh domain through a service may be a
useful first process facility; it must not masquerade as fork, which also involves
memory, capability, descriptor and continuation duplication. Threads, dynamic
loading and target signal handlers each need their own ABI work when required.

For each migrated application, compare a representative existing workload's
output and status against its previous verified invocation. Move common boot,
staging and result mechanics into the package, replace old entry points with thin
compatibility wrappers, and delete wrappers only after their callers migrate.
Keep standalone bug reproducers and historical evidence intact. Do not rewrite
all ports or remove their gates before the first shared path works.

## Code placement and review boundaries

Extend `capstone/runtime/` with the common startup contract and Linux execution
library/launcher. Extract reusable musl host services there incrementally, while
musl-specific syscall translation stays in the musl port. Put the single host CLI
package under `capstone/runtime/host` and reuse or move the existing common host helpers
with compatibility imports. Buildroot owns installation; monitor and driver
changes remain in their owning repositories. Avoid embedding application names
in these packages.

Review startup/I/O ABI, monitor escape/destruction and host session management as
separate changes with end-to-end gates. Before implementation, enumerate the
exact monitor continuation and revocation primitives available in the selected
component revisions; do not infer them from the cooperative demo. Once a new
workflow is verified, update the canonical state, testing matrix and command
cookbook together. The verified implementation now supersedes the original bounded demo. See
[applications.md](../../runtime/applications.md) for the installed workflow and
[the current state](../state/current-state.md) for the checked acceptance scope.
