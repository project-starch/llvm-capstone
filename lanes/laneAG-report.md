# Lane AG: module binfmt experiment

Branch: `experiment/module-binfmt`, based on `origin/dev` at `a799be44704b`.
Worktree: `~/llvm-capstone-laneAG`. Remote artifacts: `p12:~/lanes/laneAG`
and `p13:~/lanes/laneAG`. Experimental only; not a landing candidate.

## Design, written before implementation

Choose B: `load_binary` completes exec and returns to a tiny scalar RV64
trampoline. It calls a private step ioctl on an anonymous context descriptor;
the module owns the `cv_step` record, services, initial capability frame and
lifetime namespace. No libc, helper process, dummy thread, or new seccomp
filter is required. The domain receives no capability for the trampoline,
native stack, descriptor, or kernel state. Existing inherited seccomp still
applies to exec and the trampoline's syscalls.

A kernel loop inside `load_binary` (A) would retain exec's unfinished state,
credential/exec serialization and executable-format reference throughout the
program. It would also bypass the normal syscall return that handles pending
signals. B lets `finalize_exec` and the outer exec path finish before the first
CSRUNV, and keeps ordinary Linux PID, parent/wait and exit behavior. Linux
ptrace and signal registers describe the trampoline, not the domain. Caught
signals and capability register editing are UNRESOLVED; day one supports only
ordinary default/ignored dispositions and termination/stop/continue. The
existing 5 ms step budget and supervisor quantum remain. Each step returns
through Linux so pending work can run; no busy loop in unfinished exec.

Recognition: ELF64 little-endian ET_EXEC, machine 259, the v7 marker
`CPONVVM7`, application descriptor v2 and exactly the two load segments the
SDK linker script emits. Reject malformed or unsupported images before
`begin_new_exec`. Keep overflow-safe bounds checks on every offset, size and
count. Accept only stable local regular files under exec's write denial;
unlinked or remote-filesystem images are refused in this first prototype
rather than silently trusting bytes that may change. Setuid/setgid execution
is outside this root-only experiment and must be refused.

Load text as a private RX file mapping; data's complete pages as private RW
file mappings and its partial tail/BSS as anonymous pages. Reserve a
representable aligned image region. Reuse `add`, `resolve`, retirement and the
namespace allocator rather than inventing new capability semantics. Initial
PCC/gp name the image, sp the domain stack, a0 META, a1 exchange and a2 the
packed startup block, matching `exec.c`. Preserve argv/environment and task
identity in the launch block. The trampoline gets a separate native stack.
The context descriptor closes on a later exec and holds the module through
exit; normal file release discards the supervisor context and clears tags
before unpinning. No Linux or ISA patch and no huge pages.

Native services: MAP/UNMAP/NODES and thread-self use module helpers; only
whole grants may retire, before `vm_munmap`. Read/write use the existing
exchange-offset wire ABI and exported kernel I/O, with one snapshotted request
and bounded copies. Keep the SDK's capability-checked caller-to-exchange and
exchange-to-caller copies. This deliberately retains the exchange region and
its current pin policy so a startup improvement cannot be mistaken for a
capability-checked direct-copy implementation. There is one domain thread;
its initial transport cannot be retired. Drop the global VM lock around
blocking I/O. Exit/exit_group return an encoded exit status to the trampoline,
which executes Linux exit_group. Unsupported calls return -ENOSYS. The SDK's
mmap/munmap already use VM services; raw Linux brk cannot return usable
capability authority, and compatibility beyond the SDK path is UNRESOLVED.

Safety boundary: this is trusted Linux, single hart and pinned anonymous
backing, as on the baseline. No raw-domain-address read/write fast path:
range membership cannot establish an object's capability liveness (lane AB).
Fork, threads, signal handlers, capability ptrace/core dumps, partial unmap,
file-backed application mmap, swap and migration remain unsupported.

Implementation steps: (1) isolate the existing step operation without changing
launcher behavior; (2) opt-in binfmt, bounded loader, private trampoline and
limited services; (3) contract and measurement harness, then fixes in separate
commits. Run the staged precommit scan before every commit and update this
report with actual outputs after each step.

Validation: p12 builds only. A real-file copy of platform-dev4 carries the
module, with platform hashes refreshed. Compare default-off launcher behavior
and direct exec of hello, a compute loop and malloc/write; test malformed
images, unsupported calls, signals, wait status and module teardown. Run lane
N's unmodified plain micro parent with `exec 10` and `exec 100` against matched
plain, launcher and native children, repeat pairs, require bad=0 and one valid
profile window, report `(W100-W10)/90` split by layer. Using a plain parent
avoids requiring native fork/spawn inside the domain; the old 3.50 M reference
used a Capstone parent and is not a matched denominator. p13 gates must wait
until other gate arms finish and use only this lane's output directories.

## Preflight observations

`git worktree add` output: `HEAD is now at a799be44704b`.
`ssh p12 'df -h ~; uptime'`: available `46G`, load averages
`18.86, 19.51, 20.35`. p13: available `48G`, load averages
`14.90, 12.71, 11.89`. Other p13 gate arms were active; none were disturbed.

The build paths in the brief are absent on p12 (`No such file or directory`).
The matching prepared kernel and toolchain exist on the local host; read-only
copies are being transferred into this lane's p12 directory. Local `du -sh`
reported `1.8G` for the prepared kernel and `249M` for the toolchain.
The named micro README is absent in both local and remote branch trees;
`run-micro.sh` has the complete usage header and was read with `lqmicro.c` and
`micro-table.py`. No performance measurement has run yet.

## Results

Step 1 extracted the existing CV_STEP body into `cv_step_locked`, preserving
request/result copies at the public ioctl boundary. A slot wait now reacquires
`vm_lock` before returning to its caller. No native loader is enabled yet.
The prepared kernel and toolchain copies completed inside the lane directory.
p12 build command: `bash ~/lanes/laneAG/build.sh`; exit 0. Module SHA-256:
`6ea9346f95db4862cb720854b56b33df07d3f20d2c5e59f062610bba6efd88e0`.
The launcher is byte-identical to platform-dev4. The partial build tree emits
a toolchain-freshness warning because it has no compiler test suite; the SDK
and compiler are unchanged shared baseline artifacts.

Smoke command: `bash ~/lanes/laneAG/step1-smoke.sh`, using a real platform copy
and the unchanged Capstone micro program. Output:

```
0000 rc=0 window 0 new 0:tsv records=11 exec 10 /tmp/lqmicro
LQ capstone exec n=10 processes=10 bad=0
window 0 (invocation 0000, lqmicro): 36,621,716 attempts
```

This is a functionality smoke, not a paired per-process measurement.
Artifacts: `p12:~/lanes/laneAG/step1-smoke`. Latest prebuild disk check:
`39G` available, load averages `20.29, 20.07, 19.76`.

UNRESOLVED: native implementation, contracts, gates and paired per-process
cost are pending. No speedup or production safety qualification is claimed.

## Step 2 design amendment, before the protection bridge

The first native smoke failed, so its attempt window is INVALID as a performance
result. Command `bash ~/lanes/laneAG/native-smoke.sh` produced:

```
LQ plain exec n=10 processes=10 bad=10
capstone_dom: fault rc=-14 kind=2 cause=24 pc=0x3ff7f81de0 address=0xfffffffffffffedc
```

Disassembly places that PC in `mprotect` writing errno before TLS initialization.
The SDK's mallocng startup needs to open an initially PROT_NONE reservation.
The matching kernel's `Module.symvers` exports `vm_mmap` and `vm_brk`, but no
mprotect helper. Returning ENOSYS here cannot start an unchanged SDK program.

Extend B with one bounded syscall bridge: the module validates a page-aligned
span inside a dynamic arena, excluding all initial grants, and supplies scalar
mprotect arguments in the trampoline registers. A distinct positive return
selects that operation; its result returns via a private completion ioctl before
any domain re-entry. The module retains the pending service and a single task
owns the transport. No capability or domain-supplied syscall number reaches the
trampoline. This uses Linux's ordinary protection implementation and TLB work
without an unexported call, kernel patch, or recreated mm internals. Other
unsupported services still return ENOSYS. The trampoline becomes slightly larger
than the original ten-instruction loop. This is a kernel export gap discovered
by the experiment, not a reason to suppress mallocng's protection checks.

## Step 2 implementation and contracts

`dom_binfmt=1` inserts the loader ahead of binfmt_misc; it is absent from the
binary-format list by default. The module owns an anonymous context fd, parses
bounded ELF metadata, reuses the existing arena/step/retirement machinery and
runs the scalar trampoline. MAP/UNMAP/NODES, self, read/write and exit work;
PROTECT uses the validated bridge described above. Read-only attribution
parameters expose the last image/trampoline base for measurement outside a
window. The image's declared 15 extra transport slots are accepted as metadata;
creating threads remains unsupported. Inherited seccomp still applies.

Build on p12: `bash ~/lanes/laneAG/build-native.sh`, exit 0, no diagnostic output.
Module SHA-256:
`5d2dd8a7679d2eea04b63c90c86420e6d6e20c3eee151c05e10ed90df8c1e0f6`.
The RV64I trampoline was assembled/disassembled on p12 to verify its branch
words; it is 92 bytes. Its assembly source is retained beside the implementation.

`bash ~/lanes/laneAG/native-smoke.sh` after the protection bridge produced:

```
0000 rc=0 window 0 new 0:tsv records=0 exec 10 /tmp/child-cap
LQ plain exec n=10 processes=10 bad=0
window 0 (invocation 0000, lqmicro): 11,692,165 attempts
```

This smoke used a plain parent, so it is not a comparison with the earlier
Capstone-parent smoke. Attribution and paired runs remain pending.

`bash ~/lanes/laneAG/contracts.sh`, with binfmt_misc disabled to prevent fallback,
produced the following on p12 (exit 0):

```
PASS hello
PASS loop
PASS malloc
PASS mmap
PASS read
PASS unsupported
PASS startup
PASS exit
PASS no-helper-children
PASS one-linux-thread
PASS stop
PASS continue
PASS terminate-wait
PASS truncated
PASS marker
PASS overflow
PASS wx
PASS setuid
native contract failures=0
unload_rc=0
```

The driver verifies every output byte of the 1 MiB allocation/write, argv/env
startup and cached task IDs, exit status 37, one Linux thread/no child helpers,
and wait's signal status. `getcwd` remains ENOSYS; the first draft of the test
incorrectly expected it to succeed and was corrected to match the stated scope.
Negative images test truncated tables, a bad marker, overflowed file offset,
write+execute text and setuid refusal. Module unload after exit succeeded.
Artifacts: `p12:~/lanes/laneAG/contracts-final` and `native-smoke2`.

UNRESOLVED: full signals, capability ptrace, fork, raw brk authority and direct
capability-checked kernel copies. The SDK mmap path works; raw scalar brk is
not implemented because it cannot mint usable capability authority. No full
syscall compatibility or production safety qualification is claimed.
