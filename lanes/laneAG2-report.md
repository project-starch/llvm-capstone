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
