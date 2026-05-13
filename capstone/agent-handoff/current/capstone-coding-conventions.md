# Coding conventions for the `capstone/` workspace layer

This note defines the recommended coding/style policy before the `capstone/`
workspace grows much further.

## 1. Primary rule: follow the subtree you are touching

There is **not** one universal code style for the whole workspace.

The repository contains multiple codebases with different native conventions.
The correct default is:

> preserve the surrounding subtree/file style and avoid large style-only rewrites

That means:

- `llvm/`, `clang/`, `lld/` -> follow normal LLVM coding conventions and review habits.
- `capstone/caplifive-buildroot/package/modcapstone/module/` -> follow Linux kernel style.
- Buildroot package files / Kconfig / make fragments -> follow existing Buildroot-style conventions in that subtree.
- top-level `capstone/` paths that are **not** child repositories (`capstone/tests/`, `capstone/utils/`, `capstone/agent-handoff/`, `capstone/my_first_domain/`) -> use the explicit workspace-layer convention defined below.

## 2. Explicit convention for top-level `capstone/` code

For top-level `capstone/` code that is maintained directly in this monorepo layer,
use the following default unless a file already has a stronger local pattern.

### A. C / C++ test helpers and probes

Applies especially to `capstone/tests/runtime-qemu/**/*.c`.

- indentation: **2 spaces**, no tabs,
- braces: K&R style (`if (...) {`),
- helper functions: prefer `static` for file-local helpers,
- names: descriptive snake_case for functions/locals, `UPPER_CASE` for protocol constants,
- protocol structs: prefer fixed-width integer types or clearly named typedefs when the layout is part of an ABI,
- control flow: keep it linear and easy to audit; prefer explicit early error returns over deep nesting,
- logging: print stable, grep-friendly markers for the QEMU wrappers,
- comments: explain protocol layout, ownership transfer, round transitions, and any non-obvious ordering assumptions.

### B. Shell wrappers

Applies especially to `capstone/tests/runtime-qemu/*.sh` and `capstone/utils/*.sh`.

- start with `#!/usr/bin/env bash` and `set -euo pipefail`,
- uppercase names for environment/configuration variables,
- quote variable expansions unless deliberate word splitting is required,
- keep wrappers one-purpose and reproducible,
- prefer explicit build/run steps over hidden shell metaprogramming,
- use short comments for why a step exists, especially when it selects a non-default compiler, image, or runtime path.

### C. Markdown handoff / planning notes

Applies to `capstone/agent-handoff/**/*.md`.

- use ATX headings (`#`, `##`, `###`),
- keep long-lived facts in `current/` and session-specific facts in `history/`,
- prefer source-backed statements over speculation,
- distinguish clearly between:
  - validated current state,
  - recommendation / next step,
  - open question,
  - longer-term roadmap,
- when introducing project-specific terms, either define them locally or add them to `runtime-terms-glossary.md`.

## 3. Practical policy for new code under top-level `capstone/`

For new code in the top-level workspace layer (especially `capstone/tests/runtime-qemu/`
and `capstone/utils/`):

- keep functions small and purpose-specific,
- prefer descriptive names over short names,
- use simple, conservative control flow,
- avoid unrelated refactors,
- do not reformat large surrounding regions just to impose a new style.

## 4. Comment policy

Comments are required for non-trivial logic, especially:

- protocol layouts,
- shared-memory ownership rules,
- state-machine transitions,
- non-obvious control flow,
- assumptions about guest/runtime ordering,
- why a non-default compiler or execution path is being used.

Comments should explain **why** and **how the protocol works**, not restate trivial syntax.

## 5. Runtime probe / wrapper conventions

For files under `capstone/tests/runtime-qemu/`:

- keep wrappers one-purpose and reproducible,
- emit stable success markers,
- keep host/guest commands explicit,
- write logs to `$CAPSTONE_TMP_ROOT/`,
- prefer the guest/helper term consistently when the code runs inside the QEMU guest, so it is not confused with the developer workstation,
- prefer adding a focused wrapper over embedding opaque ad hoc command sequences in docs only.

## 6. Naming conventions

Recommended defaults for the top-level `capstone/` workspace layer:

- shell wrappers: `run-*.sh`, `build-*.sh`
- runtime probes: descriptive lowercase names with hyphenated directories or underscore-separated source files, matching the current local pattern
- handoff notes: `DD-MM-YYYY_HH-MM-SS_description.md`

## 7. What to avoid

- broad formatting-only commits,
- mixing style cleanup with semantic runtime/toolchain changes,
- inventing a new style that conflicts with the touched subtree,
- adding complex undocumented protocol code.

## 8. Review rule of thumb

Before changing a file, ask:

1. Which codebase/subtree does this belong to?
2. What style does that subtree already use?
3. Is this change semantic, or am I accidentally doing a style rewrite too?
4. Have I commented the non-obvious parts?

That is the recommended convention policy for the `capstone/` workspace unless a
subtree later adopts a stricter documented local style of its own.

