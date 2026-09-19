# PoisonCap FFmpeg allocator pilot

The extracted FFmpeg 9.0.1 AVBufferPool/AVRefStructPool port builds and runs on
the reconstructed published PoisonCap platform. The full suite passes all
29 processes when the guest's automatic libc-revocation default is disabled
before starting SSH. The adapter's explicit PoisonCap revocation remains
active. Three short native-recording replays match all 2,379 observed events
and produce byte-identical output reports.

With the guest default preserved, longer suites encounter a kernel
`share->excl` panic during VM-map locking, including a spatial-only arm.
Disabling automatic revocation for helper processes is a validated experiment
configuration, not a fix for that kernel failure or whole-process protection.

## Scope and platform

The [workflow](../../../host/cheribsd/poisoncap/README.md) provides preparation,
build, direct-link example and run commands. Its
[manifest](../../../host/cheribsd/poisoncap/platform.json) pins the published
artifact, complete source bases and compressed-capability dependency.
85,453 published regular source files were verified after building the
compiler and emulator. The artifact's separately supplied version-aware
capability header replaces the older dependency header. No additional change
to PoisonCap's compiler, emulator or kernel semantics was made.

The kernel is `CHERI-PURECAP-QEMU-POISON`; userspace and compiler are built
from the matching sources. Firmware is an external, fingerprinted input.
Each JSON report includes hashes of the emulator, firmware, kernel, libc,
image and programs. Libc automatic revocation is disabled for the tested
programs. The adapter explicitly invokes synchronous kernel revocation.
The accepted full suite additionally sets and verifies
`security.cheri.runtime_revocation_default=0` before starting SSH, using
`--disable-default-revocation`. Earlier attempts preserve the kernel default
of 1. The same kernel, libc, image and program binaries are used in both
configurations; the successful configuration does not modify PoisonCap code.
Initialisation safety is not enabled. The trusted manager obtains its arena
from `mmap()` and removes poison/VM permissions from client capabilities.

## Functional checks

`platform-1.json` records seven passing processes: purecap ABI, ordinary
bounds fault, live access, poisoned read/write, successful sweep/reuse and a
retained old alias that faults after reuse. The sweep control observes
`old_tag=0` and `root_tag=1`; its unaffected sibling remains readable.

`pool-isolated-1.json` records two ABI/bounds controls and ten passing
PoisonCap pool cases. Every expected fault requires a preceding setup marker
and exit status 162 (CheriBSD SIGPROT). The two valid cases must finish normally.

`replay-default-off-1.json` records the accepted full suite: seven platform
controls, one directly linked allocator example, all ten pool cases in both
spatial mode 0 and protected mode 2, and the recording. All spatial cases
complete; protected outcomes follow the table below. This is ten paired
pool cases, not twenty distinct fixtures.

| Case | Operation checked | PoisonCap mode 2 |
|---|---|---|
| 0 | Retained references, deferred destruction, RefStruct persistent state | Completes |
| 1 / 2 | Buffer read / write after last return | Faults |
| 3 / 4 | Old buffer read / write after same-address reuse | Faults |
| 5 | RefStruct read after last return | Faults |
| 6 | Old RefStruct write after same-address reuse | Faults |
| 8 | Nested reset returns a child buffer; old child read | Faults |
| 9 | Returned buffer followed by pool close; old read | Faults |
| 13 | Eight reuse rounds, new access and unaffected sibling | Completes |

The standalone `allocator-example` also passes in both interrupted combined
runs. A separately supplied client source links through `FFmpeg::BufferPool`;
that additional binary was built but not separately executed.
Native regression tests pass (four CTest entries), as do 23 shared Python tests,
including a console-draining test that exceeds the PTY buffer size.

## First replay and adapter counters

The recording is the existing `short` workload: 30 frames, 320x180, one second.
Two isolated fresh-guest replays produce byte-identical output reports and
the same counters (`recording-isolated-1.json` and `recording-isolated-2.json`).
The accepted full-suite replay produces the same report and counters again.
The native oracle was regenerated from this worktree and matches the previous
reference byte for byte. `measurements.json` records input/oracle hashes.
It was recorded during native decoding; this guest runs allocator operations,
not video decoding. The output event bytes match the native oracle exactly,
including allocation/reuse identities and callback effects. Platform-dependent
header accounting is compared separately, not included in that byte equality.

| Quantity | Bytes or count |
|---|---:|
| Observed events | 2,379 |
| Synchronous sweeps | 63 |
| Payload arena carved | 372,544 B |
| Pool metadata carved | 12,288 B |
| Retained payload snapshots | 372,352 B |
| Bytes poisoned, cumulative | 4,071,488 B |
| Bytes detoxed, cumulative | 3,699,136 B |
| Bytes copied, cumulative | 7,770,624 B |

Snapshots preserve persistent RefStruct fields across returns. This first
implementation snapshots both pool types and retains snapshots for the
process lifetime, including unused backing records. The almost one-for-one
extra storage is a cost of this adapter policy, not an inherent lower bound
for PoisonCap. Selective preservation and quarantine policies remain open.
These counters exclude allocator bookkeeping for snapshots, fixed arrays,
kernel storage, tags and page tables. They are not total memory overhead.

## Failed combined runs

Both `replay-1.json` and `replay-2.json` contain 13 completed passing cases and
an interrupted transfer before `pool-0-5`. Neither reaches the protected pool
matrix or recording. The first run did not continuously drain the console;
its transfer error cannot independently establish a kernel-panic cause.
The second run captures the following diagnostic after console draining was
added:

```text
exclusive lock of (sx) vm map (user) @ vm_map.c:6103
while share locked from vm_map.c:5978
panic: share->excl
_vm_map_lock_upgrade -> vm_map_lookup -> vm_fault -> vm_fault_trap
do_trap_supervisor -> do_trap_supervisor
exception 13, tval = 0x40236570
```

The caller sees an SSH banner timeout. This is a captured prototype/platform
failure, not a failed stale-pointer oracle or an architectural comparison.
The same kernel panic also occurs in `spatial-isolated-1.json`, after ten
accepted processes while downloading the next control's output. That run
contains no poison-on-return or explicit revocation calls from the adapter;
its program counters are zero. The kernel's automatic libc-revocation default
is enabled for guest helper processes, independently of the explicit disabled
policy passed to test programs.
The shared runner now drains serial output while using SSH and includes SCP
error details. Raw consoles and SSH keys remain outside the repository.

To diagnose a case independently, use `run.py --case NAME`; each invocation
starts a fresh guest and still checks ABI/bounds. `selection.json` explicitly
marks such a run as a subset. For example, `--stage pool --case pool-2-0`
checks the valid protected pool lifecycle, while `--stage replay --case recording`
requires the same recording and native oracle arguments as the full replay.
The default suite remains intact so the combined failure is reproducible.

To run the complete successful configuration, add
`--disable-default-revocation` to `--stage replay` without `--case` filters.
It passes the same controls and recording, rather than skipping failed tests.
The changed guest policy is recorded separately in the summary. The failed
default-policy attempts remain part of this bundle.

Before treating the default platform as ready for a larger comparison, isolate
the VM locking failure. Repeat and extend the explicitly configured full suite
before drawing scaling conclusions. This pilot establishes trusted
per-lease adaptation for the tested cases. It does not establish hostile
nested-manager isolation, full application protection, hardware performance
or a measured Capstone/PoisonCap ranking.
