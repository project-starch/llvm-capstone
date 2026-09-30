# Capstone project documentation

Memory design direction on `memory-trusted-linux`:
[trusted Linux, ordinary virtual memory and Capstone object lifetimes](design/trusted-linux-memory.md).
The proposal separates translation from pointer authority and defines dynamic
malloc/free and the requirements for independent fork lifetimes. It branches
from `dev`; the design is not an implemented kernel or hardware extension.

Application ports now require the [shared delegated ABI-v2 SDK](../ports/common/application/README.md).
Seven application recipes are migrated, with new QEMU functional and safety
qualification, explicit compiler/resource requirements and preserved negative
results. The launcher rejects old images; historical measurement archives
retain their original ABI and binary identities.

Inner-reuse expansion: [CPython](../experiments/study/results/cpython-reuse-four-arm-20260928/README.md)
and [PostgreSQL](../experiments/study/results/postgres-reuse-four-arm-20260928/README.md)
now each validate 12/12 complete processes across four arms. The
[first reuse figure](../experiments/study/results/reuse-five-applications-20260928/README.md)
contains SQLite, mruby, FFmpeg, PostgreSQL and CPython. CPython required a
VM-object poison-probe repair and deferred pymalloc publication at the
published quarantine thresholds. The [Perl CheriBSD recipe](../ports/perl/cheribsd/README.md)
now builds 5.36.3 and passes the 17-section smoke with both libc policy switches;
its inner SV adapters and four-arm reuse measurements remain to be implemented.

Published-threshold memory comparison: the
[fresh mruby/FFmpeg campaign](../experiments/study/results/published-policy-20260928/README.md)
validates 48 complete application processes with the audited PoisonCap
thresholds transferred from SQLite and outer libc defaults enabled. Its
reuse and selected-memory plots replace the historical custom-policy
comparisons for those workloads, with runtime repairs and countercosts
explicitly reported. Six new outer-default SQLite runs plus six archived
Capstone controls complete the 60-process three-application figure set.
The wider six-application study remains unfinished.

PostgreSQL complete-backend memory work: the
[17.5 single-user port](../ports/postgres/app/README.md) now completes
the same native-matched SQL qualification and inner reuse checks in all four
arms. Its metadata-capacity and queue-policy repair is documented in the
[four-arm campaign](../experiments/study/results/postgres-reuse-four-arm-20260928/README.md).
Standard pgbench workloads and total-memory ledgers remain later work.

Application benchmark study: the [Sublet/PoisonCap design](plans/sublet-poisoncap-memory-study.md)
uses two matched pairs for the nested boundary; default CheriBSD on/off remains
separate reference data. The [SQLite 3.22.0 pilot](../experiments/study/results/sqlite-322-memory-20260927/README.md)
now has 32/32 native-matched SQL phases in all four nested arms, selected
application-visible backing measurements, a policy-path audit and three memory
figures. These are exploratory: the SQLite source forks and build options are
not normalized across platforms, as the pilot's build audit explains. The
published PoisonCap path has unrevoked full-queue drains; the
protected comparator uses a separately identified correction. The
[normalized SQLite campaign](../experiments/study/results/sqlite-normalized-memory-20260927/README.md)
adds a matched build gate, 12 complete repeated-work processes, paired
protection-cost plots and explicit negative results. Its
[reuse-gap follow-up](../experiments/study/results/sqlite-reuse-gaps-20260927/README.md)
measures logical memsys5 address reuse in four arms. The
[whole FFmpeg decoder pool pilot](../experiments/study/results/ffmpeg-pool-memory-20260927/README.md)
adds 24 exact-output attempts over three workload sizes and four nested arms,
with separate memory ledgers and two further figures. The
[memory follow-up](../experiments/study/results/memory-followup-20260927/README.md)
adds SQLite budget attempts, a size-2 scaling attempt, and a selective FFmpeg
adapter. It leaves the protected PoisonCap minimum unresolved after kernel
panics and removes the original FFmpeg snapshot advantage as an adapter
artifact. The
[FFmpeg lease-gap campaign](../experiments/study/results/ffmpeg-reuse-gaps-20260927/README.md)
repeats the four full-decoder arms at three workload sizes; all 36 runs pass
frame oracles and all four arms have equal observed pool reuse bins. The
[mruby GC-slot campaign](../experiments/study/results/mruby-gc-memory-20260927/README.md)
adds a complete interpreter benchmark: 24/24 AO-render processes at two widths
match native PPM oracles, and Sublet preserves prompt slot reuse where the explicit
PoisonCap temporal adapter delays it. The selected GC-page metadata and
post-render retention countercosts are reported alongside the advantage. The
[full-application campaign contract](plans/application-memory-campaign.md)
fixes the same four memory experiments for each admitted benchmark. The
[planner](../experiments/study/README.md) now admits a pinned mruby four-arm
binding for subsequent runs; other PoisonCap application boundaries still
need registered adapters. The measured AO campaign used the shared guest
runners directly.

Application memory behavior: [twelve paired workload configurations](../experiments/applications/results/20260927-reuse/README.md)
pass 72/72 attempts. Every recorded Capstone memory phase matches older-QEMU
controls without in-process collection. The results quantify prompt address
reuse and post-release retention, include the large-retained-graph counterexample,
and make no timing or total-RSS claim.

Application memory: [Capstone ports versus default CheriBSD](../experiments/applications/comparison.md) now covers
FFmpeg and mruby with common allocation counters. The original matrix recorded
six Capstone node-capacity failures. The [QEMU node-reuse follow-up](../runtime/tests/application/results/20260927-node-reuse/README.md)
passes all 27 Capstone repeats at the same 65,536-node capacity, including all six
previous failures, using unchanged application binaries. Keep the original data
and larger-node controls separate. These are memory observations, not timings.

Application execution: [shared launcher, persistent Linux shell, build commands and limits](../runtime/applications.md).

Allocator trace tooling: [formats, CLI, validation scope and adapter tests](../ports/common/host/port_trace/README.md).

Everything durable the project knows about itself: architecture, the issue registry, the test
matrix, root-cause trails, and the plans in flight. Written for human developers and for any AI
coding assistant.

> **Renamed 2026-09-04.** This directory was `capstone/agent-handoff/` until it stopped being a
> handoff channel and became the documentation. `$CAPSTONE_HANDOFF_DIR` still resolves, as an
> alias for `$CAPSTONE_DOCS_DIR`.

| Path | Default |
|------|---------|
| `$CAPSTONE_DOCS_DIR` | `capstone/docs` |
| `$CAPSTONE_TMP_ROOT` (scratch) | `/tmp/capstone` |
| Shared env file | `capstone/tests/capstone-test-env.sh` |

**New here?** Start with `ONBOARDING.md` (it has a short callout for non-Claude coding agents
like Codex/Cursor, which do not auto-read `CLAUDE.md`).

## Minimal startup reading set

For application and allocator work, start with the
[port catalog](../ports/README.md) and the
[cross-repository integration plan](plans/port-stack-integration.md).

For a normal fresh session, read only these files first:

1. `README.md`
2. `state/current-state.md`
3. `state/current-next-step.md`

Everything else should be loaded only when the task needs it.

## Who works where (lanes)

UPDATED 2026-08-18: **there is no lane B.** The two-lane coordination set (`COORDINATION.md`, `MULTI-AGENT-WORKFLOW.md`, `AGENT-B-SETUP.md`, and lane B's own state files) is archived under `history/18-08-2026_ARCHIVED_*`. Anything below describing a second lane is historical.

Work is split across peer agent lanes plus, when needed, an external collaborator. **The
lane structure below is historical in its two-lane form** — see the 2026-08-18 note above. As of
2026-09-04 the live lanes are: this one (compiler/board/docs), an RTL lane, a synthesis lane on a
separate machine, and a paper lane. Read the roles, not the lane count:

- **Lane A** → commits to `dev` (called `capstone-bootstrap` until 2026-09-04).
  **Lane B** → committed to `capstone-bootstrap-b`, now `lane-b/capstone-bootstrap-b`. Both branch off the shared mainline; sync via
  `git merge origin/dev`. The A↔B split, hand-off rules, and the permanent
  repository rules are in **`CLAUDE.md`**; subagent roster and rules in
  **`ref/SUBAGENTS.md`** (read it before delegating). The old A↔B peer-lane guide is
  archived at `history/29-07-2026_ARCHIVED_DELEGATION-lane-a-b.md`.
- **External collaborators** using their own coding agent get a **self-contained,
  stock-toolchain** task doc under `plans/` (e.g. `plans/archived/xlang-repro-task.md`, the
  cross-language reproduction task) that does *not* depend on our in-flux compiler/ABI. The
  ONBOARDING callout covers pasting `CLAUDE.md` as context for non-Claude agents.

## Directory layout

This index exists because the two `state/` files went two weeks stale without anyone
noticing: nothing mapped the tree. **File counts are deliberately not kept here** — they
were, and every one of them drifted (`ref/` said 31 against 38 files, `plans/` 29 live
against 69, `history/` 196 against 225, and the header claimed 319 markdown files against
414). Run `find capstone/docs/<dir> -maxdepth 1 -type f | wc -l` for a number; what this
table is for is the **role** of each directory.

| directory | what belongs here | how to read it |
|---|---|---|
| `state/` | what is true **right now** | the first thing a new session reads. If it disagrees with `ref/ISSUES.md`, ISSUES.md wins. |
| `ref/` | durable quick-reference that rarely changes | `ISSUES.md` is the registry of every OPEN defect and is the authoritative status for them; `ISSUES-ARCHIVE.md` holds the resolved ones verbatim (split 2026-09-09). `SUBAGENTS.md` before delegating. |
| `design/` | architecture and **design decisions only** | a bug-fix, root-cause trail or audit is *not* a design decision — those go to `history/`. |
| `plans/` | work in flight | check the status line, then check `plans/archived/README.md` — a plan's own status line does not know it was archived. |
| `history/` | dated investigation notes, root-cause trails, superseded coordination docs | append-only. Do not retro-edit a finding; add a dated correction under it. |
| `patches/` | out-of-tree patches | |

### The files worth knowing by name

- **`ref/ISSUES.md`** — every defect, its status and its evidence. The single most load-bearing
  document in the repo. Silicon defects are `S-nn`, compiler defects `C-nn`, QEMU `Q-nn`,
  RTL-vs-spec `R-nn`.
- **`ref/testing-matrix.md`** — what is expected to pass where.
- **`ref/HOW-TO-LAUNCH-ON-FPGA.md`** — the long form behind the `board-run` skill.
- **`ref/fpga-silicon-measurements-for-paper.md`** — where measured numbers land. Results can be
  recorded here without touching the paper, which is deliberate.
- **`ONBOARDING.md`** — fast-track setup, including the callout for non-Claude agents that do not
  auto-read `CLAUDE.md`.

### Related trees outside this directory

- `capstone/bug-corpora/INDEX.md` — **generated**: every piece of bug material in the repo, in
  one table. Third-party defects as cases (here and in `xlang/`), our own silicon defects, our
  own compiler and runtime defects, and the protection fixtures that are none of those. Counts
  come from each corpus's `corpus.json` and each port's `port.json`, never from typing.
- `capstone/tests/fpga-repros/` — one folder per silicon defect, each a **self-contained report**
  that may already be a live link held by the hardware side. See its own `README.md`. These are
  evidence and are never pruned.
- `.claude/skills/` — procedures that auto-load when a task matches. `board-run` and `rtl-sim`.
- `CLAUDE.md` (repo root) — the permanent rules. Read it before the docs, not after.

## Current verified baseline

The `domain-process-runtime` application stack supports a persistent Linux guest,
ordinary application arguments/streams, trusted fault/preemption return and owned
resource reuse. The installed QEMU guest passes exhaustion/recovery followed by
1,008 mixed starts in the same boot, with stable retained resources. Perl uses
the shared SDK; Perl and mruby execute through the common launcher. The delegated
runtime (application ABI v2) is the only application runtime: the musl runtime's
HostCall v0 mode and the probes that ran on it were removed on 2026-09-30, and the
runtime probes that remain run as delegated applications
(`tests/runtime-qemu/run-delegated-probes.py`). See
[applications](../runtime/applications.md), the
[checked acceptance](../runtime/tests/application/results/20260926-qemu-rebased.json)
and [current state](state/current-state.md) for scope and remaining failures.
This is a one-hart QEMU platform extension, not a new FPGA result.
The current [Perl `t/base` result](../ports/perl/musl/results/2026-09-26/base-tests-rebased-qemu.txt)
is 8/9 files passing; the remaining case requires target subprocess creation.
Upstream Perl coverage remains incomplete.


Opt-in [generic client-fault recovery](../runtime/domain-faults.md) and its
[standalone tests](../runtime/tests/fault-recovery/README.md) are independent of
the allocator ports. The runtime requires the matching QEMU trap-delivery change;
it does not change the default fault behavior or claim FPGA recovery.
PostgreSQL's CMake component supports AllocSet, Generation, Slab and Bump under
Sublet at the shared 17.0 pin. Its native and QEMU test entry points and scope
are in [the component README](../ports/postgres/memory-contexts/README.md).
This is allocator-level coverage, not a protected server or consumer-defect suite.


> **Historical baseline list (scope updated 2026-09-26).** The list below records earlier
> QEMU/runtime validation. The available legacy snapshot currently fails null_blk and
> the borrowed-region file-open-close proof on both old and new platforms; see current state.
> It accumulated before the silicon work and says nothing about it. For
> what is verified **on the board** — the resident bitstream, S-06/S-07/S-08/S-12, and SQLite's
> logic tests running in a capability domain — read `state/current-state.md`. For the status of
> any individual defect, `ref/ISSUES.md` outranks both.

- working sample-domain build + runtime validation
- working OpenSBI/runtime path via `capstone/caplifive-buildroot/build/local.mk`
- validated shared-region runtime proof
- validated HostCall stdout, filewrite, fileread proofs
- validated HostCall file open/close handle-lifecycle proof
- validated HostCall handle-based FILE_WRITE, FILE_READ, FILE_SYNC, FILE_STAT_BASIC, FILE_TRUNCATE proofs
- validated HostCall SQLite-facing PATH_ACCESS and PATH_DELETE proofs
- validated combined reusable file-object proof
- working baseline and split `null_blk` regressions
- validated CoreMark profile-run on Capstone PureCap ("Correct operation validated.")
  using compiled C `domain_main` rather than `coremark_domain_entry.S`
- 78 validated BEEBS benchmarks on the split host/domain runtime path; the
  newest are `matmult-float` and `whetstone` (added `atan` to the shared libm),
  completing the soft-float/libm-only FP class
- aggregate regression wrappers are available for HostCall proofs, `null_blk`,
  and the full validated BEEBS set; BEEBS is serial by default and supports
  opt-in isolated parallel runs with `RUN_ALL_BEEBS_JOBS=N`

See `state/current-state.md` for the canonical snapshot.

## Contributing rules

- treat these as local workflow overlays on top of normal LLVM/Buildroot/Linux/QEMU conventions,
- do not mark a step complete until it has been tested at the affected layer,
- keep non-trivial code documented with concise comments, especially around state transitions and ownership rules,
- after a coherent validated change set, prefer a multi-line commit message with a short subject plus a detailed body,
- for capstone-local commits, do not add a redundant `capstone` prefix to the commit subject unless the broader monorepo context requires it,
- keep manager-facing summaries as local artifacts under `$CAPSTONE_TMP_ROOT/`, not as committed repository files,
- active plans go in `plans/` (committed here); do not store project plans outside this repository.

## Read on demand

Use these only when the task actually needs them:

- **`ref/RATE-RULE.md` — why a single wedge is NOT a result on silicon, with the measured k/n.
  Read before recording, citing or acting on any board outcome.** Rescued 2026-08-18 from deep
  inside `SILICON-BLOCKER.md`, where it silently invalidated most of that document.
- `ref/known-good-controls.md` — **partly refreshed.** Its three load-bearing rows (`k800`,
  `k1200`, `r14lp`) were re-verified 2026-09-05 on `caplifive_s12fix_5097eb166`; every other row
  still reads `last verified 2026-08-06`, against bitstreams replaced at least three times since.
  A preflight gate depends on this file: re-verify a row before relying on it.
- `ref/SILICON-BLOCKER.md` — **SUPERSEDED**, the 2026-08-01..06 investigation. The defect it
  chased is S-06, fixed in silicon. Kept because its line numbers are cited from live repro
  folders; do not renumber or trim it.
- **`ref/ISSUES.md` — the open-issues registry (RTL/FPGA + compiler), each with a runnable repro. Read before re-investigating anything; update whenever an issue is found, characterised or closed.**
- `ref/HOW-TO-MEASURE-OVERHEAD.md` — **how overhead is measured** (bare-metal baseline, gates, traps). Read before producing or citing a ratio.
- `ref/testing-matrix.md` — compact map of test layers and entry points
- `ref/capstone-agent-test-instructions.md` — practical command cookbook
- `ref/capstone-coding-conventions.md` — local coding conventions
- `history/09-09-2026_ARCHIVED_delegation-guidance.md` — bounded executor rules for split agent
  work; archived 2026-09-09, superseded in practice by `ref/SUBAGENTS.md`
- `ref/beebs-benchmark-bringup-manual.md` — exact workflow for adding one or
  more BEEBS benchmark wrappers
- `ref/capstone-purecap-pointer-model.md` — pointer/capability authority model
- `ref/project-structure-overview.md` — workspace guide
- `ref/runtime-terms-glossary.md` — terminology reference
- `design/sqlite-minimal-vfs-path.md` — concrete SQLite-facing next step and minimal VFS mapping
- `design/hostcall-file-service-v0-wire-spec.md` — wire-format and state-machine spec for the HostCall file service
- `design/stable-file-service-subset.md` — reusable HostCall file-service proposal
- `design/split-host-enclave-strategy.md` — source-backed architectural detail
- `design/hosted-libc-os-analysis.md` — hosted Linux blockers and sysroot mismatch analysis
- `design/research-decisions-log.md` — paper-worthy implementation decisions and tradeoffs, cited by commit hash
- `plans/backend-compiler-fixes.md` — known backend bugs and workarounds (from CoreMark bring-up)
- [Domain applications as Linux commands](plans/domain-process-runtime.md) — implemented
  shared launcher, owned lifecycle, shell I/O, common SDK and persistent development VM;
  architecture, verified acceptance and platform limits
- `history/README.md` — historical index and note selection guide

## History rules

- write history notes in English,
- use `DD-MM-YYYY_HH-MM-SS` in filenames,
- avoid proper names or direct references to specific people in filenames/titles,
- keep durable current guidance in `state/`, `ref/`, or `design/`, not in `history/`,
- if two history notes become near-duplicates, keep one full primary source and reduce the other to a short pointer.

## Maintenance rule

If the validated baseline or recommended workflow changes, update at least:

- `README.md`
- `new-chat-prompt.md`
- `state/current-state.md`
- `state/current-next-step.md`
- `ref/testing-matrix.md`
- `ref/capstone-agent-test-instructions.md`

Update deeper `design/` files only when their subject actually changed.

## What this does not yet mean

This does **not** yet mean that a full hosted `capstone64-unknown-linux-gnu` user-space is ready.
The current validated path is still the split host/domain runtime path.
