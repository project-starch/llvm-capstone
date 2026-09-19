# Port and runtime integration

Status: pending integration. Checked 2026-09-19 against LLVM `dev`
`dc40bc1df43f`. This is the remaining integration plan, not a new runtime or
silicon validation. The [port catalog](../../ports/README.md) describes scope
and build entry points independently of the PR queue.

## Prerequisites and order

| Change | Base and prerequisites | Integration action |
|---|---|---|
| [LLVM #50](https://github.com/project-starch/llvm-capstone/pull/50): Whisper contexts | `dev` | Independent allocator port; preserve borrowed-buffer ownership semantics |
| [LLVM #51](https://github.com/project-starch/llvm-capstone/pull/51): PostgreSQL managers | `dev` | Land the four-manager implementation and consistent 17.0 pin; retain 17.5 result provenance |
| [LLVM #53](https://github.com/project-starch/llvm-capstone/pull/53): generic recovery | `dev`; QEMU #6 for delivered faults | Canonical allocator-independent runtime implementation |
| [LLVM #52](https://github.com/project-starch/llvm-capstone/pull/52): PostgreSQL recovery | Based on #51; also needs #53 and QEMU #6 | After #51 and #53, reconcile onto their common base so the remaining change is PostgreSQL adoption |
| [LLVM #54](https://github.com/project-starch/llvm-capstone/pull/54): PostgreSQL corpus | #51 | Eight reduced consumer-defect sequences with real allocators; halting fault arms |
| [LLVM #55](https://github.com/project-starch/llvm-capstone/pull/55): delivered corpus | #52 and #54; QEMU #6 | Carries cherry-picked corpus commits; reconcile after both parents so they are not reviewed twice |
| [LLVM #56](https://github.com/project-starch/llvm-capstone/pull/56): pymalloc corpus | `dev` with the shared Sublet header | Twenty reduced consumer sequences, forty paired QEMU arms; not a full interpreter |
| [LLVM #57](https://github.com/project-starch/llvm-capstone/pull/57): ggml survey | Stacked on #56 | Survey evidence; does not require the Whisper port #50 |
| [QEMU #5](https://github.com/project-starch/capstone-qemu/pull/5): capability atomics | `c128-qemu-merge` | Pair with the LLVM atomic lowering already in `dev`; precedes QEMU #6 |
| [QEMU #6](https://github.com/project-starch/capstone-qemu/pull/6): local fault delivery | QEMU #5 | Required for the recovery runtime and delivered corpus, separate from firmware |

The shared Sublet header restoration is byte-identical to the header on #50,
#51 and #56. It can land independently without selecting an allocator port.
Do not replace it with the older `capstone/sublet/sublet.h`: that is a different
implementation and is not the public runtime include path.

## Resolve the shared files deliberately

A synthetic merge of #50 into the audited `dev` succeeds. Merging #51 or #53
after that exposes overlapping additions to `state/current-state.md`; #53
also conflicts in `capstone/runtime/CMakeLists.txt`. Individual GitHub
`MERGEABLE` statuses do not establish that the combined stack merges cleanly.

For the runtime CMake conflict, retain both the header-presence checks from
the port branches and the generic domain configuration/Linux fault policy
from #53. The resulting combination is already present in #52's runtime
CMake file. Do not select either conflict side wholesale. For state documents,
retain each component's scope and evidence, and distinguish pending branch
results from changes already on `dev`.

Validate the resolved stack with the generic runtime's native policy tests
and standalone QEMU arms, the PostgreSQL native and QEMU client tests, and
both corpus variants. Select QEMU `77d69353b7` or a verified descendant via
`CAPSTONE_QEMU_BINARY` and build both domain and guest with
`CAPSTONE_DOMAIN_FAULT_RECOVERY=ON`. OFF remains the default. An old emulator
can produce an expected capability fault while still terminating the VM;
that is not delivered-fault evidence. Recovery remains cooperative, with no
claim of hostile-domain containment or new FPGA validation.

## Other repositories and existing branches

[Buildroot #3](https://github.com/project-starch/caplifive-buildroot/pull/3)
is already included by ancestry in `capstone-bootstrap` at `d04bd83b13cd`,
together with #2. It is not a pending code dependency despite its stale open
PR state at audit time. Loader, module and host binaries must still be rebuilt
together because the domain-create ioctl structure changed.

[Paper #1](https://github.com/project-starch/nested-allocators-paper/pull/1)
indexes the historical FFmpeg evidence. Its code reference, LLVM #44, is now
included in `dev`; its pinned historical evidence should not be relabeled as
results from later source or runtime revisions.

The compiler atomics, musl survey, FFmpeg, original PostgreSQL replay/profiles,
shared template and CPython port branch tips are already ancestors of the
audited `dev`, even where GitHub labels their PRs CLOSED rather than MERGED.
Keep branch ancestry separate from PR status when identifying cleanup
candidates. Remove a branch/worktree only after checking its dirty state and
unique commits; older divergent `c128/*` branches are not automatically stale.

Follow the existing `<topic>/<sequence>-<purpose>` naming for new work. Keep
one reviewable concern per PR, state both Git base and cross-repository runtime
dependencies, and preserve the validation revision and known failed attempts.
No experimental application CI is required; skipped metadata workflows supply
no build or runtime verdict.
