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

### HostCall file open/close proof

```bash
cd "$CAPSTONE_REPO_ROOT" && \
bash capstone/tests/runtime-qemu/run-hostcall-file-open-close-probe.sh \
  > "$CAPSTONE_TMP_ROOT/run-hostcall-file-open-close-probe.txt" 2>&1

sed -n '1,260p' "$CAPSTONE_TMP_ROOT/run-hostcall-file-open-close-probe.txt"
```

### HostCall file handle write proof

```bash
cd "$CAPSTONE_REPO_ROOT" && \
bash capstone/tests/runtime-qemu/run-hostcall-file-handle-write-probe.sh \
  > "$CAPSTONE_TMP_ROOT/run-hostcall-file-handle-write-probe.txt" 2>&1

sed -n '1,260p' "$CAPSTONE_TMP_ROOT/run-hostcall-file-handle-write-probe.txt"
```

### HostCall file handle read proof

```bash
cd "$CAPSTONE_REPO_ROOT" && \
bash capstone/tests/runtime-qemu/run-hostcall-file-handle-read-probe.sh \
  > "$CAPSTONE_TMP_ROOT/run-hostcall-file-handle-read-probe.txt" 2>&1

sed -n '1,260p' "$CAPSTONE_TMP_ROOT/run-hostcall-file-handle-read-probe.txt"
```

### HostCall file handle sync proof

```bash
cd "$CAPSTONE_REPO_ROOT" && \
bash capstone/tests/runtime-qemu/run-hostcall-file-handle-sync-probe.sh \
  > "$CAPSTONE_TMP_ROOT/run-hostcall-file-handle-sync-probe.txt" 2>&1

sed -n '1,260p' "$CAPSTONE_TMP_ROOT/run-hostcall-file-handle-sync-probe.txt"
```

### HostCall file handle stat proof

```bash
cd "$CAPSTONE_REPO_ROOT" && \
bash capstone/tests/runtime-qemu/run-hostcall-file-handle-stat-probe.sh \
  > "$CAPSTONE_TMP_ROOT/run-hostcall-file-handle-stat-probe.txt" 2>&1

sed -n '1,260p' "$CAPSTONE_TMP_ROOT/run-hostcall-file-handle-stat-probe.txt"
```

### HostCall file handle truncate proof

```bash
cd "$CAPSTONE_REPO_ROOT" && \
bash capstone/tests/runtime-qemu/run-hostcall-file-handle-truncate-probe.sh \
  > "$CAPSTONE_TMP_ROOT/run-hostcall-file-handle-truncate-probe.txt" 2>&1

sed -n '1,260p' "$CAPSTONE_TMP_ROOT/run-hostcall-file-handle-truncate-probe.txt"
```

### HostCall path access proof

```bash
cd "$CAPSTONE_REPO_ROOT" && \
bash capstone/tests/runtime-qemu/run-hostcall-path-access-probe.sh \
  > "$CAPSTONE_TMP_ROOT/run-hostcall-path-access-probe.txt" 2>&1

sed -n '1,260p' "$CAPSTONE_TMP_ROOT/run-hostcall-path-access-probe.txt"
```

### HostCall path delete proof

```bash
cd "$CAPSTONE_REPO_ROOT" && \
bash capstone/tests/runtime-qemu/run-hostcall-path-delete-probe.sh \
  > "$CAPSTONE_TMP_ROOT/run-hostcall-path-delete-probe.txt" 2>&1

sed -n '1,260p' "$CAPSTONE_TMP_ROOT/run-hostcall-path-delete-probe.txt"
```

### HostCall combined file-object proof

```bash
cd "$CAPSTONE_REPO_ROOT" && \
bash capstone/tests/runtime-qemu/run-hostcall-combined-file-object-probe.sh \
  > "$CAPSTONE_TMP_ROOT/run-hostcall-combined-file-object-probe.txt" 2>&1

sed -n '1,260p' "$CAPSTONE_TMP_ROOT/run-hostcall-combined-file-object-probe.txt"
```

### Optional: metadata-only second-`PENDING` diagnostic

```bash
cd "$CAPSTONE_REPO_ROOT" && \
bash capstone/tests/runtime-qemu/run-hostcall-second-pending-probe.sh \
  > "$CAPSTONE_TMP_ROOT/run-hostcall-second-pending-probe.txt" 2>&1

sed -n '1,260p' "$CAPSTONE_TMP_ROOT/run-hostcall-second-pending-probe.txt"
```

### Optional: second-`PENDING` payload-reuse diagnostic

```bash
cd "$CAPSTONE_REPO_ROOT" && \
bash capstone/tests/runtime-qemu/run-hostcall-second-pending-payload-probe.sh \
  > "$CAPSTONE_TMP_ROOT/run-hostcall-second-pending-payload-probe.txt" 2>&1

sed -n '1,260p' "$CAPSTONE_TMP_ROOT/run-hostcall-second-pending-payload-probe.txt"
```

### Optional: second-`PENDING` payload-reuse revoke diagnostic

```bash
cd "$CAPSTONE_REPO_ROOT" && \
bash capstone/tests/runtime-qemu/run-hostcall-second-pending-payload-revoke-probe.sh \
  > "$CAPSTONE_TMP_ROOT/run-hostcall-second-pending-payload-revoke-probe.txt" 2>&1

sed -n '1,260p' "$CAPSTONE_TMP_ROOT/run-hostcall-second-pending-payload-revoke-probe.txt"
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

Expected: all three wrappers print `QEMU smoke passed.`. Inspect whichever log
matches the path you touched.

### CoreMark correctness run

```bash
cd "$CAPSTONE_REPO_ROOT" && \
bash capstone/tests/runtime-qemu/run-coremark.sh \
  > "$CAPSTONE_TMP_ROOT/run-coremark.txt" 2>&1

grep -E "(Correct|Errors|crc|ERROR|PASSED)" "$CAPSTONE_TMP_ROOT/run-coremark.txt"
```

Expected: `Correct operation validated.` and `__COREMARK_PASSED__` in the output.
This also verifies that CoreMark runs through the compiled C `domain_main` wrapper
instead of the old per-domain assembly entry.
Run when touching anything in `capstone/benchmarks/coremark/` or backend codegen
(instruction selection, frame lowering — see `plans/backend-compiler-fixes.md`).

### BEEBS `fac` correctness run

```bash
cd "$CAPSTONE_REPO_ROOT" && \
bash capstone/benchmarks/beebs/run-beebs-fac.sh \
  > "$CAPSTONE_TMP_ROOT/run-beebs-fac.txt" 2>&1

grep -E "(BEEBS|beebs-fac|PASSED|ERROR)" "$CAPSTONE_TMP_ROOT/run-beebs-fac.txt"
```

Expected: `beebs-fac-host: correctness marker validated` and
`__BEEBS_FAC_PASSED__` in the output. Run when touching
`capstone/benchmarks/beebs/` or the benchmark split host/domain wrapper path.

### BEEBS `insertsort` correctness run

```bash
cd "$CAPSTONE_REPO_ROOT" && \
bash capstone/benchmarks/beebs/run-beebs-insertsort.sh \
  > "$CAPSTONE_TMP_ROOT/run-beebs-insertsort.txt" 2>&1

grep -E "(BEEBS|beebs-insertsort|PASSED|ERROR)" "$CAPSTONE_TMP_ROOT/run-beebs-insertsort.txt"
```

Expected: `beebs-insertsort-host: correctness marker validated` and
`__BEEBS_INSERTSORT_PASSED__` in the output. Run when touching
`capstone/benchmarks/beebs/`, the benchmark split host/domain wrapper path, or
backend codegen used by BEEBS scalar integer benchmarks.

### BEEBS `fibcall` correctness run

```bash
cd "$CAPSTONE_REPO_ROOT" && \
bash capstone/benchmarks/beebs/run-beebs-fibcall.sh \
  > "$CAPSTONE_TMP_ROOT/run-beebs-fibcall.txt" 2>&1

grep -E "(BEEBS|beebs-fibcall|PASSED|ERROR)" "$CAPSTONE_TMP_ROOT/run-beebs-fibcall.txt"
```

Expected: `beebs-fibcall-host: correctness marker validated` and
`__BEEBS_FIBCALL_PASSED__` in the output. Run when touching
`capstone/benchmarks/beebs/` or the benchmark split host/domain wrapper path.

### BEEBS `cnt` correctness run

```bash
cd "$CAPSTONE_REPO_ROOT" && \
bash capstone/benchmarks/beebs/run-beebs-cnt.sh \
  > "$CAPSTONE_TMP_ROOT/run-beebs-cnt.txt" 2>&1

grep -E "(BEEBS|beebs-cnt|PASSED|ERROR)" "$CAPSTONE_TMP_ROOT/run-beebs-cnt.txt"
```

Expected: `beebs-cnt-host: correctness marker validated` and
`__BEEBS_CNT_PASSED__` in the output. Run when touching
`capstone/benchmarks/beebs/`, the benchmark split host/domain wrapper path, or
backend codegen used by BEEBS global-state integer benchmarks.

### BEEBS `bubblesort` correctness run

```bash
cd "$CAPSTONE_REPO_ROOT" && \
bash capstone/benchmarks/beebs/run-beebs-bubblesort.sh \
  > "$CAPSTONE_TMP_ROOT/run-beebs-bubblesort.txt" 2>&1

grep -E "(BEEBS|beebs-bubblesort|PASSED|ERROR)" "$CAPSTONE_TMP_ROOT/run-beebs-bubblesort.txt"
```

Expected: `beebs-bubblesort-host: correctness marker validated` and
`__BEEBS_BUBBLESORT_PASSED__` in the output. Run when touching
`capstone/benchmarks/beebs/`, the benchmark split host/domain wrapper path, or
backend codegen used by BEEBS global-state integer benchmarks.

### BEEBS `prime` correctness run

```bash
cd "$CAPSTONE_REPO_ROOT" && \
bash capstone/benchmarks/beebs/run-beebs-prime.sh \
  > "$CAPSTONE_TMP_ROOT/run-beebs-prime.txt" 2>&1

grep -E "(BEEBS|beebs-prime|PASSED|ERROR)" "$CAPSTONE_TMP_ROOT/run-beebs-prime.txt"
```

Expected: `beebs-prime-host: correctness marker validated` and
`__BEEBS_PRIME_PASSED__` in the output. Run when touching
`capstone/benchmarks/beebs/`, the benchmark split host/domain wrapper path, or
backend codegen used by BEEBS scalar global-state and modulo/division paths.

### BEEBS `recursion` correctness run

```bash
cd "$CAPSTONE_REPO_ROOT" && \
bash capstone/benchmarks/beebs/run-beebs-recursion.sh \
  > "$CAPSTONE_TMP_ROOT/run-beebs-recursion.txt" 2>&1

grep -E "(BEEBS|beebs-recursion|PASSED|ERROR)" "$CAPSTONE_TMP_ROOT/run-beebs-recursion.txt"
```

Expected: `beebs-recursion-host: correctness marker validated` and
`__BEEBS_RECURSION_PASSED__` in the output. Run when touching
`capstone/benchmarks/beebs/`, the benchmark split host/domain wrapper path, or
backend codegen used by BEEBS recursive-call and scalar global-state paths.

### BEEBS `janne_complex` correctness run

```bash
cd "$CAPSTONE_REPO_ROOT" && \
bash capstone/benchmarks/beebs/run-beebs-janne-complex.sh \
  > "$CAPSTONE_TMP_ROOT/run-beebs-janne-complex.txt" 2>&1

grep -E "(BEEBS|beebs-janne-complex|PASSED|ERROR)" "$CAPSTONE_TMP_ROOT/run-beebs-janne-complex.txt"
```

Expected: `beebs-janne-complex-host: correctness marker validated` and
`__BEEBS_JANNE_COMPLEX_PASSED__` in the output. Run when touching
`capstone/benchmarks/beebs/`, the benchmark split host/domain wrapper path, or
backend codegen used by BEEBS scalar global-state integer benchmarks.

## 6. Important caveats

- The current validated path is still the split host/domain runtime path, not a full hosted Capstone Linux user-space.
- The QEMU smoke harness uses snapshot mode, so runtime tests should not mutate `rootfs.ext2`.
- Buildroot getty is pinned to `ttyS0`, matching the active QEMU serial console.
- The QEMU smoke harness forces `-smp 1` for deterministic boot progress.
- `run-smoke.sh` remains useful as a quick probe, but the dedicated HostCall wrappers are the stronger runtime checks.
- After OpenSBI/kernel changes, rebuild dependent modules/packages if their `vermagic` must match the active kernel.

## 7. When to load deeper docs

Read these only if the task needs them:

- `ref/testing-matrix.md` — which test layer proves what
- `design/stable-file-service-subset.md` — the next reusable HostCall file-service target
- `design/split-host-enclave-strategy.md` — architectural rationale
- `design/hosted-libc-os-analysis.md` — hosted Linux blockers
