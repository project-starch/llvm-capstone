# Capstone bring-up onboarding

Fast-track for anyone (developer or AI assistant) starting from scratch.

> **Using a non-Claude coding agent** (Codex, Cursor, …)? Those agents do **not** auto-read
> the repo-root `CLAUDE.md`, so paste it in explicitly as context, together with `README.md`,
> `state/current-state.md`, and `state/current-next-step.md`. Everything those agents need is
> in the tracked repo files — there is no separate agent-specific onboarding file.
>
> **Handed a self-contained task doc** under `plans/` (e.g. the cross-language reproduction
> task, `plans/xlang-repro-task.md`)? That doc is authoritative and stock-toolchain only — you
> can skip the repo-build steps below unless it says otherwise.
>
> **Contributing to the core repo?** Work happens in two peer lanes — Lane A on
> `capstone-bootstrap`, Lane B on `capstone-bootstrap-b` — under the rules in `CLAUDE.md`
> (peer-lane guide archived at `history/29-07-2026_ARCHIVED_DELEGATION-lane-a-b.md`)
> and the hard constraints in the repo-root `CLAUDE.md`. Read both before committing.

---

## 0. Clone the repository

```bash
git clone --recurse-submodules -b capstone-bootstrap https://github.com/project-starch/llvm-capstone
cd llvm-capstone
```

`--recurse-submodules` is **required, not optional**: the submodules
(`caplifive-buildroot`, `caplifive-system`, `capstone-qemu`, `capstone-c`,
`capstone-spec`, `capstone-ariane`, `paper`) themselves contain **nested** submodules
(QEMU roms/edk2, buildroot/opensbi components, …). If you already cloned without it, run:

```bash
git submodule update --init --recursive
```

Plain `git submodule update --init` (non-recursive) leaves the nested submodules empty and
the build will fail. After the recursive populate, the workspace is ready for the steps below.

---

## 1. Prerequisites

| Component | What you need |
|-----------|--------------|
| LLVM build | Pre-built at `$CAPSTONE_LLVM_BUILD_DIR` (default: `llvm/cmake-build-debug`). If absent, build LLVM with the Capstone target enabled. |
| QEMU | Pre-built Capstone-enabled QEMU at `capstone/capstone-qemu/build/qemu-system-riscv64`. |
| Buildroot image | Pre-built at `capstone/caplifive-buildroot/`. Provides the guest Linux + OpenSBI + kernel module. |
| Cross-toolchain | `riscv64-buildroot-linux-gnu-gcc` from the Buildroot SDK (for guest-side `.user` binaries). |
| Host tools | `bash`, `python3`, standard POSIX utils. |

If the pre-built artifacts are missing, consult the sub-repo READMEs:
- `capstone/capstone-qemu/` — QEMU build instructions
- `capstone/caplifive-buildroot/` — Buildroot image rebuild

---

## 2. Environment setup

Source the shared environment file from the repo root (or any subdirectory):

```bash
source capstone/tests/capstone-test-env.sh
```

Key variables it sets (all overridable via env before sourcing):

| Variable | Default | Purpose |
|----------|---------|---------|
| `CAPSTONE_REPO_ROOT` | repo root | Absolute path to repository |
| `CAPSTONE_TMP_ROOT` | `/tmp/capstone` | Scratch output (logs, built artifacts) |
| `CAPSTONE_LLVM_BUILD_DIR` | `llvm/cmake-build-debug` | LLVM build tree |
| `CAPSTONE_CLANG` | `…/bin/clang` | Capstone-target clang |
| `CAPSTONE_LD_LLD` | `…/bin/ld.lld` | Capstone-target lld |
| `CAPSTONE_QEMU_BINARY` | `capstone-qemu/build/qemu-system-riscv64` | QEMU binary |
| `CAPSTONE_HANDOFF_DIR` | `capstone/agent-handoff` | This directory |

---

## 3. Verify the toolchain

```bash
source capstone/tests/capstone-test-env.sh
"$CAPSTONE_CLANG" --version        # should print Capstone-enabled clang
"$CAPSTONE_LD_LLD" --version       # should print lld
"$CAPSTONE_QEMU_BINARY" --version  # should print QEMU version
```

If any binary is missing, the pre-built artifacts need rebuilding (see sub-repo READMEs).

---

## 4. Build and run the sample domain

The sample domain is the minimal "hello world" for the Capstone runtime.

```bash
source capstone/tests/capstone-test-env.sh
bash capstone/tests/runtime-qemu/run-smoke.sh
```

Expected output contains:
```
Created domain ID = 0
Called dom (1-th time) retval = 42
```

---

## 5. Run the full runtime probe suite

Each probe is self-contained: it builds the domain + host binary, runs QEMU, and checks
for a success marker. Run them in order:

```bash
source capstone/tests/capstone-test-env.sh

bash capstone/tests/runtime-qemu/run-shared-region-probe.sh
bash capstone/tests/runtime-qemu/run-hostcall-stdout-probe.sh
bash capstone/tests/runtime-qemu/run-hostcall-filewrite-probe.sh
bash capstone/tests/runtime-qemu/run-hostcall-fileread-probe.sh
bash capstone/tests/runtime-qemu/run-hostcall-file-open-close-probe.sh
bash capstone/tests/runtime-qemu/run-hostcall-file-handle-write-probe.sh
bash capstone/tests/runtime-qemu/run-hostcall-file-handle-read-probe.sh
bash capstone/tests/runtime-qemu/run-hostcall-file-handle-sync-probe.sh
bash capstone/tests/runtime-qemu/run-hostcall-file-handle-stat-probe.sh
bash capstone/tests/runtime-qemu/run-hostcall-file-handle-truncate-probe.sh
bash capstone/tests/runtime-qemu/run-hostcall-path-access-probe.sh
bash capstone/tests/runtime-qemu/run-hostcall-path-delete-probe.sh
bash capstone/tests/runtime-qemu/run-hostcall-combined-file-object-probe.sh
bash capstone/tests/runtime-qemu/run-coremark.sh
```

All should exit 0 and print a `PASSED` or `validated` marker.

---

## 6. Null-block kernel module tests

These require a rebuilt kernel module against the active kernel:

```bash
bash capstone/tests/runtime-qemu/run-nullblk-baseline.sh
bash capstone/tests/runtime-qemu/run-nullblk-split-io.sh
bash capstone/tests/runtime-qemu/run-nullblk-split-rmmod.sh
```

See `ref/testing-matrix.md` for when these matter.

---

## 7. What to read next

| File | Purpose |
|------|---------|
| `README.md` | Project rules, directory map, baseline summary |
| `state/current-state.md` | Authoritative minimal state snapshot |
| `state/current-next-step.md` | Current recommended next milestone |
| `ref/capstone-agent-test-instructions.md` | Command cookbook for common tasks |
| `ref/testing-matrix.md` | When to run which test layer |
| `ref/project-structure-overview.md` | Repository layout guide |
