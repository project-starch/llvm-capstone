# Lane AG: module binfmt experiment

Branch: `experiment/module-binfmt`, based on `origin/dev` at `a799be44704b`.
Worktree: `~/llvm-capstone-laneAG`. Remote artifacts: `p12:~/lanes/laneAG`
and `p13:~/lanes/laneAG`. Experimental only; not a landing candidate.

Verified at `346af62c417d`: native exec 1.159 M attempts/process versus 2.554 M
launcher and 0.199 M plain; 18 native contract checks passed. The final errno
correction and two added checks remain UNRESOLVED because SSH authentication
expired. p13 qualification is also UNRESOLVED; the overlapping control run was
stopped and invalidated. No full compatibility or landing approval is claimed.

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
`micro-table.py`. At this preflight point no performance measurement had run.

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

At step 1, native implementation, contracts, gates and paired per-process
cost were pending. The later sections record the completed experiments.

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

## Matched exec measurement (commit 346af62c417d)

Command on p12: `bash ~/lanes/laneAG/contracts/measure.sh` (exit 0), followed by
`python3 ~/lanes/laneAG/contracts/summarize.py ~/lanes/laneAG` (exit 0).
The unchanged micro source is from `perf/linux-question-micro` at
`390a4043851a9995be74b726fc72ceec1cd6a7b8`. All three arms use the same plain
parent and experimental module; plain/launcher use dom_binfmt=0, native uses 1.
The launcher binary is baseline-identical, and the Capstone child is identical
between launcher and native. SLOTS=2, 8G, no-pcrel, icount=0, ASLR disabled,
no huge pages. Every one of the 18 invocations had rc=0, bad=0 and exactly one
complete window. No kernel-idle attempts were classified in these windows.

Quoted summary output (attempts, `(W100-W10)/90`):

```
plain pair=1 W10=2007312 W100=19896582 per_process=198769.67
plain pair=2 W10=1957930 W100=19882335 per_process=199160.06
plain pair=3 W10=1954060 W100=19837155 per_process=198701.06
plain mean=198876.93 range=198701.06..199160.06 busy=198876.93
launcher pair=1 W10=25661690 W100=255372746 per_process=2552345.07
launcher pair=2 W10=25419364 W100=255353978 per_process=2554829.04
launcher pair=3 W10=25417956 W100=255280726 per_process=2554030.78
launcher mean=2553734.96 range=2552345.07..2554829.04 busy=2553734.96
native pair=1 W10=11639130 W100=115987800 per_process=1159429.67
native pair=2 W10=11585489 W100=115844820 per_process=1158437.01
native pair=3 W10=11590695 W100=115905783 per_process=1159056.53
native mean=1158974.40 range=1158437.01..1159429.67 busy=1158974.40
native reduction=54.616% speedup=2.203x
```

This is an instruction-attempt ratio, not measured wall-clock acceleration.
The historical 3.50 M launcher result used a Capstone parent; the matched
plain-parent launcher control here is 2.554 M. It would be incorrect to divide
3.50 M by this experiment's 1.159 M and call that a measured speedup.

Mean attempts per process by layer, quoted from the same command:

```
layer plain launcher native native-minus-launcher
domain: 0.00 16305.00 16305.00 +0.00
firmware: 17.03 649.25 285.76 -363.49
kernel: 196223.90 2318866.24 1062521.30 -1256344.94
launcher: 0.00 143170.38 0.00 -143170.38
module: 0.00 72745.06 78015.32 +5270.26
other: 0.00 245.04 0.00 -245.04
program: 2636.00 1754.00 1754.00 +0.00
trampoline: 0.00 0.00 93.03 +93.03
```

The added module work is small beside the eliminated Linux and launcher work.
The domain's own instruction count is identical. The largest function deltas
(native minus launcher, attempts per process) from that command are:

```
kernel do_seccomp -163113.00
kernel __memset -80481.59
kernel bpf_convert_filter -57262.00
launcher memcpy -49850.00
kernel __memcpy -41586.50
kernel __mod_memcg_lruvec_state -41159.05
kernel unmap_page_range -40983.37
kernel __mod_lruvec_page_state -31988.00
launcher msort_with_tmp -24207.00
kernel folio_flags.constprop.0 -23948.77
```

Module sections were recorded after reload. Six unmeasured launcher probes
agreed on image base `0x3ff7f00000`; all six native samples agreed on image
`0x3ff7f80000` and trampoline `0x3ff7f6b000`. Attribution checks instruction bytes
at those placements, preserves any ambiguous matches, and uses the exact image
hashes recorded by the harness. No ambiguous bucket appeared in these results.
`other` remains explicit. Native read-only parameter reads and launcher probes
occur after the measured windows. Artifacts:
`p12:~/lanes/laneAG/measure-{plain,launcher,native}` and
`p12:~/lanes/laneAG/measurement-summary.txt`.

The 1 MiB malloc/write program passed byte-for-byte functionality; its attempt
cost was not separately profiled.

## Gate status and review follow-up

UNRESOLVED: no valid sequential p13 gate comparison. The prepared plain/hooks
modules use supervisor_slots=32, with SHA-256 respectively:
`0d7ac5070347e2270862e4eb009dbf86b0934138500c6396f15912d685182819` and
`6b2c8998123f082e89511f60c634332ee9f02bd09c83c91dc2ccc292789919a8`.
Both platforms were real-file copies of platform-landing inside this lane.
The control's preflight lacked an empty llvm directory; that was corrected.
More materially, another lane started an arm while these copies were prepared.
The final ps check printed its active jobs but the shell did not stop on them.
I detected the overlap, verified every member of this lane's process group
660555, and terminated only that group. The experiment arm never started.
The incomplete `p13:~/lanes/laneAG/out-ctl` is invalid for qualification, even
where individual checks passed. The other lane was still active on recheck.
No other lane's process or artifact was changed.

A final source review found that negative read/write fds returned EFAULT rather
than EBADF. The final branch includes the small errno correction and tests for invalid fds
plus protection of invalid/initial ranges. UNRESOLVED: their build and runtime
verification were blocked when p12 SSH demanded a new Tailscale authentication
check. p13 later required reauthentication too. Only this lane's waiting local
SSH/rsync process groups were stopped; no remote verification command from that
blocked upload was started. The measurements and 18-pass contract result above describe the tested
commit 346af62c417d, not the unverified correction. The final driver now contains 20 checks; the
two new checks have not been run. The last successful p12 disk check reported
28G free; the last successful p13 check reported 38G free.

## Recommendation

GO for continued experimentation; NO-GO for landing this prototype. Native exec
works for the stated single-thread contracts and removes over half the matched
process-start attempts, but most remaining attempts are still in Linux. Keep
the trampoline approach: it preserves ordinary exec completion and signal/wait
behavior while requiring only 92 bytes of scalar code. A supported kernel
mprotect export could remove its sole syscall bridge.

Next: obtain a clean default-off gate comparison, validate the errno correction,
then investigate the remaining kernel startup/pin/teardown work with the same
matched harness. Keep the fixed transport until a capability-checked copy
primitive can validate object liveness; arena membership alone cannot do that.
Full signal delivery needs a capability register/frame ABI, ptrace needs a
capability regset, fork needs a context/namespace lifetime design, and raw brk
needs an authority-return ABI. These are UNRESOLVED, not silently emulated.

Commits so far: `613da802c07b` (design), `613acaa0482e` (step extraction),
`346af62c417d` (opt-in loader and validated protection bridge). Every commit's
staged precommit scan exited 0 and reported CLEAN. The scanner warned that its
external name denylist was absent; the FPGA URL check was explicitly disabled
as required by the lane brief. New committed content was also reviewed for
personal names and home-directory paths.

The final validation commit adds the reproducible contract, malformed-image,
mixed-image profiler and matched-pair summary scripts under
`capstone/runtime/virtual/tests/native-dom`. The module's fd correction changes
only its invalid-descriptor errno; it is kept explicit rather than claiming a
build or test that authentication prevented. `git diff --check` passed. The
assembly-byte check on p12 also printed:

```
PASS trampoline assembled words match; bytes=92
```
