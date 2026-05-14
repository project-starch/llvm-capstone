# Capstone test/run instructions for future agent sessions

This file is the practical command cookbook for the current tree.
It intentionally keeps only the most reused commands and points the reader at the test wrappers instead of re-explaining every old path in detail.

## 0. Setup once per shell

Common defaults live in:
- `capstone/tests/capstone-test-env.sh`

Use:

```bash
cd "$(git rev-parse --show-toplevel)"
source capstone/tests/capstone-test-env.sh
mkdir -p "$CAPSTONE_TMP_ROOT"
```

Important variables after sourcing:
- `$CAPSTONE_REPO_ROOT`
- `$CAPSTONE_TMP_ROOT`
- `$CAPSTONE_LLVM_BUILD_DIR`
- `$CAPSTONE_LLVM_BIN`
- `$CAPSTONE_LLVM_LIT`
- `$CAPSTONE_CLANG`
- `$CAPSTONE_LD_LLD`
- `$CAPSTONE_LLVM_READOBJ`
- `$CAPSTONE_BUILDROOT_DIR`
- `$CAPSTONE_QEMU_BINARY`
- `$CAPSTONE_HANDOFF_DIR`

## 1. General rule

Prefer redirecting command output to `$CAPSTONE_TMP_ROOT/...` and then inspecting the log.
Do not mark a step complete until it has been re-tested at the affected layer.

Also preserve the user's local IDE data:
- do **not** delete `$CAPSTONE_REPO_ROOT/.idea/`

## 2. Fast compiler/linker checks

### Backend / SelectionDAG

```bash
cd "$CAPSTONE_REPO_ROOT" && \
"$CAPSTONE_LLVM_LIT" -sv \
  "$CAPSTONE_REPO_ROOT/llvm/test/CodeGen/Capstone" \
  > "$CAPSTONE_TMP_ROOT/capstone-lit-codegen.txt" 2>&1

sed -n '1,260p' "$CAPSTONE_TMP_ROOT/capstone-lit-codegen.txt"
```

### Clang builtins

```bash
cd "$CAPSTONE_REPO_ROOT" && \
"$CAPSTONE_LLVM_LIT" -sv \
  "$CAPSTONE_REPO_ROOT/clang/test/CodeGen/capstone-builtins.c" \
  "$CAPSTONE_REPO_ROOT/clang/test/CodeGen/builtins-capstone.c" \
  > "$CAPSTONE_TMP_ROOT/capstone-clang-builtins.txt" 2>&1

sed -n '1,220p' "$CAPSTONE_TMP_ROOT/capstone-clang-builtins.txt"
```

### LLD / ELF emulation

```bash
cd "$CAPSTONE_REPO_ROOT" && \
"$CAPSTONE_LLVM_LIT" -sv \
  "$CAPSTONE_REPO_ROOT/lld/test/ELF/emulation-capstone.s" \
  > "$CAPSTONE_TMP_ROOT/capstone-lld.txt" 2>&1

sed -n '1,220p' "$CAPSTONE_TMP_ROOT/capstone-lld.txt"
```

### Linux driver regression

```bash
cd "$CAPSTONE_REPO_ROOT" && \
"$CAPSTONE_LLVM_LIT" -sv \
  "$CAPSTONE_REPO_ROOT/clang/test/Driver/capstone-linux-toolchain.c" \
  > "$CAPSTONE_TMP_ROOT/capstone-driver.txt" 2>&1

sed -n '1,220p' "$CAPSTONE_TMP_ROOT/capstone-driver.txt"
```

## 3. Rebuild LLVM/Clang when needed

```bash
cd "$CAPSTONE_REPO_ROOT" && \
cmake --build "$CAPSTONE_LLVM_BUILD_DIR" -j"$(nproc)" \
  > "$CAPSTONE_TMP_ROOT/capstone-llvm-build.txt" 2>&1

tail -n 200 "$CAPSTONE_TMP_ROOT/capstone-llvm-build.txt"
```

## 4. Build and inspect the sample domain

### Build

```bash
cd "$CAPSTONE_REPO_ROOT/capstone/my_first_domain" && \
LLVM_BIN="$CAPSTONE_LLVM_BIN" ./build.sh \
  > "$CAPSTONE_TMP_ROOT/capstone-my-domain-build.txt" 2>&1

sed -n '1,220p' "$CAPSTONE_TMP_ROOT/capstone-my-domain-build.txt"
```

### Inspect ELF header

```bash
"$CAPSTONE_LLVM_READOBJ" -h \
  "$CAPSTONE_REPO_ROOT/capstone/my_first_domain/my_domain.dom" \
  > "$CAPSTONE_TMP_ROOT/capstone-my-domain-readobj.txt" 2>&1

sed -n '1,220p' "$CAPSTONE_TMP_ROOT/capstone-my-domain-readobj.txt"
```

## 5. Current runtime entry points

### Shared-region proof

```bash
cd "$CAPSTONE_REPO_ROOT" && \
bash capstone/tests/runtime-qemu/run-shared-region-probe.sh \
  > "$CAPSTONE_TMP_ROOT/run-shared-region-probe.txt" 2>&1

sed -n '1,260p' "$CAPSTONE_TMP_ROOT/run-shared-region-probe.txt"
```

### HostCall stdout proof

```bash
cd "$CAPSTONE_REPO_ROOT" && \
bash capstone/tests/runtime-qemu/run-hostcall-stdout-probe.sh \
  > "$CAPSTONE_TMP_ROOT/run-hostcall-stdout-probe.txt" 2>&1

sed -n '1,260p' "$CAPSTONE_TMP_ROOT/run-hostcall-stdout-probe.txt"
```

### HostCall filewrite proof

```bash
cd "$CAPSTONE_REPO_ROOT" && \
bash capstone/tests/runtime-qemu/run-hostcall-filewrite-probe.sh \
  > "$CAPSTONE_TMP_ROOT/run-hostcall-filewrite-probe.txt" 2>&1

sed -n '1,260p' "$CAPSTONE_TMP_ROOT/run-hostcall-filewrite-probe.txt"
```

### HostCall fileread proof

```bash
cd "$CAPSTONE_REPO_ROOT" && \
bash capstone/tests/runtime-qemu/run-hostcall-fileread-probe.sh \
  > "$CAPSTONE_TMP_ROOT/run-hostcall-fileread-probe.txt" 2>&1

sed -n '1,260p' "$CAPSTONE_TMP_ROOT/run-hostcall-fileread-probe.txt"
```

### `null_blk` baseline and split regressions

```bash
cd "$CAPSTONE_REPO_ROOT" && \
bash capstone/tests/runtime-qemu/run-nullblk-baseline.sh \
  > "$CAPSTONE_TMP_ROOT/run-nullblk-baseline.txt" 2>&1

cd "$CAPSTONE_REPO_ROOT" && \
bash capstone/tests/runtime-qemu/run-nullblk-split-io.sh \
  > "$CAPSTONE_TMP_ROOT/run-nullblk-split-io.txt" 2>&1

cd "$CAPSTONE_REPO_ROOT" && \
bash capstone/tests/runtime-qemu/run-nullblk-split-rmmod.sh \
  > "$CAPSTONE_TMP_ROOT/run-nullblk-split-rmmod.txt" 2>&1
```

Inspect whichever log matches the path you touched.

## 6. Important caveats

- The current validated path is still the split host/domain runtime path, not a full hosted Capstone Linux user-space.
- `run-smoke.sh` remains useful as a quick probe, but the dedicated HostCall wrappers are the stronger runtime checks.
- After OpenSBI/kernel changes, rebuild dependent modules/packages if their `vermagic` must match the active kernel.

## 7. When to load deeper docs

Read these only if the task needs them:

- `current/testing-matrix.md` — which test layer proves what
- `current/stable-file-service-subset.md` — the next reusable HostCall file-service target
- `current/split-host-enclave-strategy.md` — architectural rationale
- `current/hosted-libc-os-analysis.md` — hosted Linux blockers
