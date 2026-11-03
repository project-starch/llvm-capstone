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

UNRESOLVED: implementation, build, contracts, gates and per-process cost are
pending. No speedup or production safety qualification is claimed.
