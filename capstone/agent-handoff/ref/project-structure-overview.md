# Project structure overview

This note is the quick orientation guide for both future agents and human developers.
It explains how this workspace is organized, with special attention to the
`capstone/` subtree where the Capstone-specific work lives.

## 1. Top-level repository shape

This workspace is an LLVM monorepo with a Capstone bring-up layered on top.

At a high level:

- `llvm/` — backend, MC layer, target parser, code generation, target-specific tests.
- `clang/` — frontend target support, builtins, driver logic.
- `lld/` — linker emulation / ELF behavior.
- `libc/`, `compiler-rt/`, `libcxx/`, etc. — standard LLVM subprojects.
- `capstone/` — Capstone-specific runtime, helper scripts, nested component repositories, tests, and handoff documents.
- `build/` — local build tree for the monorepo.

## 2. What `capstone/` contains

`capstone/` is not one single codebase. It is a workspace umbrella for several roles.

### A. `capstone/agent-handoff/`
Persistent context for continuing the work in a new chat or session.

- `state/` — volatile: `current-state.md`, `current-next-step.md` (update after each milestone).
- `ref/` — durable quick reference (testing matrix, test cookbook, conventions, glossary, project layout).
- `design/` — deep architecture and design docs (HostCall wire spec, CoreMark PureCap, SQLite VFS plan, ...).
- `plans/` — active WIP plans committed to git (currently: backend compiler fixes).
- `history/` — timestamped archival notes.
- `new-chat-prompt.md` — session prompt template (non-Claude agents: see the ONBOARDING callout).

Read `README.md` then `state/current-state.md` and `state/current-next-step.md` first
when re-entering the project.

### B. `capstone/tests/`
Shared test harnesses and reproducible runtime probes.

Important paths:

- `capstone/tests/capstone-test-env.sh` — common path defaults used by tests and docs.
- `capstone/tests/runtime-qemu/` — QEMU-based runtime wrappers and probes.

This is the safest place to add small validated experiments before touching child repositories.

### C. `capstone/utils/`
Top-level helper scripts that are useful across the Capstone workspace.

Current examples:

- `run-qemu.sh` — manual QEMU launcher.
- `pack-capstone-files.sh` — helper for packaging context files.

### D. `capstone/my_first_domain/`
Minimal native Capstone sample domain.

Important files:

- `build.sh` — builds the sample using the in-tree LLVM toolchain.
- `main.c`, `start.S`, `link.ld` — the sample program and link layout.

Use this directory when validating the native `EM_CAPSTONE` sample flow.

### E. `capstone/benchmarks/`
Benchmark programs ported to the Capstone PureCap execution model.

- `capstone/benchmarks/coremark/` — CoreMark 1.01 PureCap bring-up (complete). All three
  algorithms (list, matrix, state machine) run with validated CRCs. Backend workarounds and
  their root causes are documented in `capstone/agent-handoff/plans/backend-compiler-fixes.md`.
  Architecture-specific design decisions (CRC derivation, node count differences) are in
  `capstone/agent-handoff/design/coremark-purecap.md`.

Next planned benchmarks (pending prologue frame lowering fix):
- BEEBS (https://github.com/mageec/beebs)
- RV8 (https://github.com/larkmjc/rv8-bench)

### F. Nested component repositories inside `capstone/`
These are separate repositories used by the local workspace and now exposed as
explicit submodules in the top-level repo.

- `capstone/caplifive-buildroot/` — Buildroot-based guest runtime, Linux/OpenSBI integration, userspace helpers, kernel modules, packaging.
- `capstone/capstone-qemu/` — Capstone-enabled QEMU tree.
- `capstone/capstone-c/` — companion Capstone compiler/runtime repository used by some Buildroot flows.
- `capstone/capstone-spec/` — SPEC-related or benchmark-related auxiliary repository.

These child repos should remain visible in the IDE and should be changed only when
there is a clear justification.

## 3. Where the main classes of work happen

### LLVM backend / codegen work
Usually under:

- `llvm/lib/Target/Capstone/`
- `llvm/test/CodeGen/Capstone/`
- `llvm/lib/TargetParser/`

### Clang frontend / target / builtin work
Usually under:

- `clang/lib/Basic/Targets/`
- `clang/lib/CodeGen/TargetBuiltins/`
- `clang/test/CodeGen/`
- `clang/test/Driver/`

### Linker work
Usually under:

- `lld/ELF/`
- `lld/test/ELF/`

### Guest runtime / firmware / kernel-module work
Usually under:

- `capstone/caplifive-buildroot/components/opensbi/`
- `capstone/caplifive-buildroot/components/linux/`
- `capstone/caplifive-buildroot/package/modcapstone/`
- `capstone/caplifive-buildroot/package/capstone-null-blk/`
- `capstone/caplifive-buildroot/package/capstone-sbi-domain/`

### Runtime probes and reproducible experiments
Usually under:

- `capstone/tests/runtime-qemu/`

## 4. Important execution-model terminology

### Host
In the current split-runtime notes, “host” usually means the ordinary Linux userspace
helper running inside the QEMU guest image.

Examples:

- `/capstone-test.user`
- `sbi-dom.user`
- the new HostCall probe helper built in `capstone/tests/runtime-qemu/`

This is **not** the same thing as the developer's physical workstation.

### Guest
The whole virtual machine booted by QEMU:

- OpenSBI
- Linux kernel
- Buildroot root filesystem
- ordinary RISC-V userspace helpers
- Capstone runtime module/device path

### Domain
An isolated Capstone payload executed through the Capstone runtime path, rather than
as a normal Linux process ABI.

### `.smode`
An S-mode payload used inside the split-domain runtime path.

## 5. Current practical rule of thumb

When deciding where to implement the next step:

- if it is a compiler/linker bug, prefer `llvm/`, `clang/`, or `lld/` plus lit tests;
- if it is a reproducible runtime probe, prefer `capstone/tests/runtime-qemu/` first;
- if it is a real firmware/kernel/module/runtime change, then touch the relevant child repo under `capstone/`;
- if it is project memory or process guidance, update `capstone/agent-handoff/`.

