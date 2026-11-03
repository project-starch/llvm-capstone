# M4 round 2: native domain execution

Branch: `runtime/module-binfmt`. Worktree: `~/llvm-capstone-laneAG2`.
Base fetched from `origin/dev`: `114b90b7be2e`.
Remote writes are confined to `p12:~/lanes/laneAG2` and
`p13:~/lanes/laneAG2`. Status: design recorded; implementation and
qualification pending. No performance result is claimed yet.

## Design recorded before implementation

Replay the previous lane's four commits onto current dev in small scanned
commits, resolving conflicts against today's step and mapping machinery.
Keep the native binfmt behind `dom_binfmt=1`, default off. Preserve exec
completion into a private scalar user trampoline, a private context fd,
separate native stack and domain capability grants. There is no glibc
launcher, spawner, dummy thread or new seccomp filter on this path.

Extend the trampoline through the existing delegated wire protocol. The
module must snapshot and validate each request before exposing a syscall;
never load a syscall number or native pointer directly from untrusted domain
memory. Reuse the existing shape validation, fd policy and argument
marshalling rules. Raw pointer arguments may name only bounded transport or
private bounce memory. Domain objects stay accessible through their own
capability-checked copies. Hold transport for the entire operation and admit
writable transport with CV_TOUCH semantics before writing or calling Linux.
Services already handled by the module remain there. A raw syscall result
must complete the outstanding request before any domain re-entry.

Review signal handover, disposition and restart rules against the existing
service loop. Do not silently drop caught signals or treat domain handler
addresses as native function pointers. Thread creation, fork, spawner job
control and any operation whose policy or lifetime cannot be preserved return
ENOSYS. Record the exact supported syscall set against the launcher seccomp
allowlist; a generic six-argument ecall alone is not a safe delegate service.
The domain cannot execute Linux syscalls directly or mint capabilities for
trampoline code, stack or context state. The native task retains the domain's
Linux credentials; inherited seccomp restrictions continue to apply.

Validation proceeds from remote module build and native contract tests to
hosted/runtime gates, then matched plain/launcher/native lqmicro exec and
SPEC test gcc, imagick, lbm and nab. Use existing compilers, kernel build and
SPEC trees. Make real-file platform variants and refresh hashes. Run p13
arms only when no other gate arm is active. Mark unsupported gates and blocked
steps UNRESOLVED rather than treating missing coverage as success. Produce
north-star and top-function deltas only for completed PASS runs.

Before every build/copy check free space; do not start below the brief's
threshold. At completion remove only this lane's remote build/platform/scratch
trees and retain cited logs and results. Each implementation commit requires
a staged precommit scan. Push only `runtime/module-binfmt`.

## Preflight

Commands run: `git fetch` with recursive submodule fetch and commit-graph
writes disabled; `git worktree add -b runtime/module-binfmt ... origin/dev`.
Worktree output: `HEAD is now at 114b90b7be2e`.
Remote `df -h ~` outputs showed `30G` available on p12 and `40G` on p13.
p13 had another lane's gate arm running; no gate was started or disturbed.
The task's prepared kernel path is absent on p12; locate a reusable build
before copying or building. This is not yet a build blocker.

## Bridge implementation plan

The scalar trampoline will forward a module-prepared syscall number and six
register arguments, then return its result through a completion ioctl. Only
the module constructs native pointers. Ordinary buffer arguments use a private
untagged bounce mapping; the module copies bounded results into held transport
through `touch()` (the implementation of CV_TOUCH). The domain receives no
capability for bounce memory. The shared delegate validator and result-length
rules are compiled into the module, avoiding a second shape table. Vector
arguments require a separately rebuilt native iovec array.

Initial implementation excludes caught-signal registration, waits requiring
signal restart handover, nested message arguments, process creation and job
control until their existing semantics can be preserved. Refusal is ENOSYS,
never raw forwarding. Inherited signal mask/ignored dispositions must be
published at startup even in this restricted mode. Full caught-signal parity
is UNRESOLVED at this step. SPEC coverage must be established by execution;
absence of threads alone does not prove compatibility.

The prior four changes replayed without conflicts as `cca762d6cd17`,
`c939237c9671`, `d2afe5da16c0`, `e84bba350f61`. Each staged scan exited zero.
The prepared kernel and toolchain from the previous lane are available on
p12 and reused read-only; no duplicate kernel/compiler build is needed.

## First bridge validation

p12 commands `bash ~/lanes/laneAG2/build.sh` (logs/build-bridge2.log) and
`bash ~/lanes/laneAG2/contracts/build-contracts.sh` completed successfully.
The assembled scalar trampoline is 152 bytes. No local build was run.
`bash ~/lanes/laneAG2/contracts/run-contract.sh` completed with output:

```
PASS files
PASS private-fd
PASS signal-mask
native contract failures=0
unload_rc=0
```

The original exec/negative-image contracts also passed. The new cases exercise
open/stat/seek/vector and positioned I/O, protect the anonymous context fd from
close/dup/overwrite/fcntl, and check masks plus default/ignored dispositions.
The previously unverified invalid-fd and invalid-protection cases now ran.
The bridge shares the wire validator, result-byte calculation, fd positions
and fcntl/ioctl command policy with the launcher. Writes into transport use
`touch()` before copying. The trampoline touches neither transport nor domain
objects. Private bounce and vector mappings have no domain grants.

A review before wider qualification found an aliasing difference: strings in
the launcher point into transport while output buffers use bounce memory.
Putting both at their exchange offsets in one native bounce could overwrite a
pathname when a forged wire request overlaps it with output. Reserve separate
private slices for strings. Ordinary SDK requests use disjoint wire spans;
this correction is necessary for arbitrary valid wire requests too.

The new raw-wire contract reproduced this bug before the fix:
`status=7936 output_size=0`, `FAIL wire`, `native contract failures=1`
(`contracts-wire-before/contract.log`). After rebuilding the module and running
the same binary against it, `contracts-wire-final/contract.log` contains all
24 `PASS` lines, including `PASS wire`, and `native contract failures=0`;
`after.log` ends `unload_rc=0`. The wire test also rejects malformed flags,
overflowing lengths, hidden-pointer ioctl, clone, raw sigaction and malformed
exit requests. Final module SHA256:
`b66ff16ca270d402dc4f78254e44c47112be52d5b2383438cb201f4b10d804ca`.
