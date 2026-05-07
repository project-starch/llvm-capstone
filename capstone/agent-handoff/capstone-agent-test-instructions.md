# Capstone test/run instructions for future agent sessions

This file is a practical handoff note for future agent sessions working on the Capstone backend/toolchain in `$CAPSTONE_REPO_ROOT`.

The user explicitly prefers that terminal output be redirected to files in `$CAPSTONE_TMP_ROOT/` (default: `/tmp/capstone/`) and then inspected from there, rather than reading command output directly.

---

## 0. Repositories / important paths

Common path defaults live in:
- `capstone/tests/capstone-test-env.sh`

After sourcing it, the important variables are:
- `$CAPSTONE_REPO_ROOT`
- `$CAPSTONE_TMP_ROOT` (default: `/tmp/capstone`)
- `$CAPSTONE_LLVM_BUILD_DIR` (default: `$CAPSTONE_REPO_ROOT/llvm/cmake-build-debug`)
- `$CAPSTONE_LLVM_BIN`
- `$CAPSTONE_LLVM_LIT`
- `$CAPSTONE_CLANG`
- `$CAPSTONE_LD_LLD`
- `$CAPSTONE_LLVM_READOBJ`
- `$CAPSTONE_BUILDROOT_DIR`
- `$CAPSTONE_QEMU_BINARY`
- `$CAPSTONE_HANDOFF_DIR`

Before running commands that produce logs, start from the repository root and source the shared defaults:

```bash
cd "$(git rev-parse --show-toplevel)"
source capstone/tests/capstone-test-env.sh
```

---

## 1. General rule for running commands

Always redirect output to `$CAPSTONE_TMP_ROOT/...` and inspect the file afterwards.

Examples:

```bash
cd "$CAPSTONE_REPO_ROOT" && \
cmake --build "$CAPSTONE_LLVM_BUILD_DIR" --target check-llvm > "$CAPSTONE_TMP_ROOT/capstone-check-llvm.txt" 2>&1
```

```bash
sed -n '1,220p' "$CAPSTONE_TMP_ROOT/capstone-check-llvm.txt"
```

For shell scripts, if you want command tracing, prefer:

```bash
bash -x ./build.sh > "$CAPSTONE_TMP_ROOT/my-domain-build.txt" 2>&1
```

or, if the script already has `set -x`, just redirect its output:

```bash
./build.sh > "$CAPSTONE_TMP_ROOT/my-domain-build.txt" 2>&1
```

---

## 2. Fast backend regression checks

### 2.1 Run focused Capstone llc tests

Use this when you changed backend lowering, instruction selection, frame lowering, memory lowering, etc.

```bash
cd "$CAPSTONE_REPO_ROOT" && \
"$CAPSTONE_LLVM_LIT" -sv \
  "$CAPSTONE_REPO_ROOT/llvm/test/CodeGen/Capstone" \
  > "$CAPSTONE_TMP_ROOT/capstone-lit-codegen.txt" 2>&1
```

Inspect:

```bash
sed -n '1,260p' "$CAPSTONE_TMP_ROOT/capstone-lit-codegen.txt"
```

Important tests currently present in `llvm/test/CodeGen/Capstone/`:
- `intrinsics.ll`
- `cap-control-flow.ll`
- `load-store.ll`
- `ptr-arith.ll`
- `select-cap.ll`
- `branch_cmp.ll`
- `calling-conv.ll`
- `globals.ll`
- `external-calls.ll`
- `frame-lowering.ll`
- `frame-realign.ll`
- `dynamic-alloca.ll`
- `dynamic-alloca-realign-fail.ll`
- `mem-intrinsics.ll`
- `aggregate-copy.ll`
- `cap-constants.ll`
- `cap-constants-invalid.ll`
- `legalize-trunc-i128.ll`

### 2.2 Run only a subset of Capstone backend tests

Useful after a focused patch.

```bash
cd "$CAPSTONE_REPO_ROOT" && \
"$CAPSTONE_LLVM_LIT" -sv \
  "$CAPSTONE_REPO_ROOT/llvm/test/CodeGen/Capstone/cap-control-flow.ll" \
  "$CAPSTONE_REPO_ROOT/llvm/test/CodeGen/Capstone/load-store.ll" \
  "$CAPSTONE_REPO_ROOT/llvm/test/CodeGen/Capstone/frame-lowering.ll" \
  > "$CAPSTONE_TMP_ROOT/capstone-lit-focused.txt" 2>&1
```

Inspect:

```bash
sed -n '1,220p' "$CAPSTONE_TMP_ROOT/capstone-lit-focused.txt"
```

---

## 3. Clang frontend builtin checks

Use this after touching `BuiltinsCapstone.td` or `clang/lib/CodeGen/TargetBuiltins/Capstone.cpp`.

```bash
cd "$CAPSTONE_REPO_ROOT" && \
"$CAPSTONE_LLVM_LIT" -sv \
  "$CAPSTONE_REPO_ROOT/clang/test/CodeGen/capstone-builtins.c" \
  > "$CAPSTONE_TMP_ROOT/capstone-clang-builtins.txt" 2>&1
```

Inspect:

```bash
sed -n '1,220p' "$CAPSTONE_TMP_ROOT/capstone-clang-builtins.txt"
```

---

## 4. Rebuild LLVM/Clang after source changes

If backend, clang, or lld sources changed, rebuild before running tests.

```bash
cd "$CAPSTONE_REPO_ROOT" && \
cmake --build "$CAPSTONE_LLVM_BUILD_DIR" -j$(nproc) \
  > "$CAPSTONE_TMP_ROOT/capstone-llvm-build.txt" 2>&1
```

Inspect tail first:

```bash
tail -n 200 "$CAPSTONE_TMP_ROOT/capstone-llvm-build.txt"
```

If needed, inspect head/middle too:

```bash
sed -n '1,260p' "$CAPSTONE_TMP_ROOT/capstone-llvm-build.txt"
```

---

## 5. Rebuild and run the `my_first_domain` runtime sample

This is the current VM-validated sample flow.

### 5.1 Build the sample domain

`capstone/my_first_domain/build.sh` already uses `set -euxo pipefail`, so command tracing is printed automatically.

```bash
cd "$CAPSTONE_REPO_ROOT/capstone/my_first_domain" && \
LLVM_BIN="$CAPSTONE_LLVM_BIN" ./build.sh > "$CAPSTONE_TMP_ROOT/capstone-my-domain-build.txt" 2>&1
```

Inspect:

```bash
sed -n '1,220p' "$CAPSTONE_TMP_ROOT/capstone-my-domain-build.txt"
```

Expected current behavior:
- `build.sh` defaults to the in-tree `ld.lld`
- the produced ELF is native `EM_CAPSTONE`
- the old header rewrite shim is only used when `HOST_LD` is overridden to a non-`ld.lld` linker

### 5.2 Inspect the resulting ELF header

```bash
"$CAPSTONE_LLVM_READOBJ" -h \
  "$CAPSTONE_REPO_ROOT/capstone/my_first_domain/my_domain.dom" \
  > "$CAPSTONE_TMP_ROOT/capstone-my-domain-readobj.txt" 2>&1
```

```bash
sed -n '1,220p' "$CAPSTONE_TMP_ROOT/capstone-my-domain-readobj.txt"
```

### 5.3 Optional: disassemble the sample

Expect some `<unknown>` output for custom instructions unless disassembler support has been extended.

```bash
"$CAPSTONE_LLVM_BIN/llvm-objdump" -d \
  "$CAPSTONE_REPO_ROOT/capstone/my_first_domain/my_domain.dom" \
  > "$CAPSTONE_TMP_ROOT/capstone-my-domain-objdump.txt" 2>&1
```

```bash
sed -n '1,260p' "$CAPSTONE_TMP_ROOT/capstone-my-domain-objdump.txt"
```

### 5.4 Rebuild the userspace loader/module package if you changed loader-side source

Needed after edits under:
- `capstone/caplifive-buildroot/package/modcapstone/...`

```bash
cd "$CAPSTONE_BUILDROOT_DIR/build" && \
make modcapstone-rebuild > "$CAPSTONE_TMP_ROOT/capstone-modcapstone-rebuild.txt" 2>&1
```

```bash
sed -n '1,220p' "$CAPSTONE_TMP_ROOT/capstone-modcapstone-rebuild.txt"
```

### 5.5 Copy the sample into the Buildroot test-domains directory

```bash
cp "$CAPSTONE_REPO_ROOT/capstone/my_first_domain/my_domain.dom" \
  "$CAPSTONE_BUILDROOT_DIR/build/target/test-domains/my_domain.dom"
```

### 5.6 Rebuild the rootfs image so the new domain lands in the VM image

```bash
cd "$CAPSTONE_BUILDROOT_DIR/build" && \
make > "$CAPSTONE_TMP_ROOT/capstone-buildroot-make.txt" 2>&1
```

Inspect:

```bash
tail -n 200 "$CAPSTONE_TMP_ROOT/capstone-buildroot-make.txt"
```

### 5.7 Run QEMU

`run-qemu.sh` itself does not print shell tracing, so use `bash -x`.

```bash
cd "$CAPSTONE_BUILDROOT_DIR" && \
bash -x ./run-qemu.sh > "$CAPSTONE_TMP_ROOT/capstone-qemu-run.txt" 2>&1
```

Because QEMU is interactive, this command will continue running until the VM exits. In practice, for manual interactive testing it is often easier to run it in a terminal you watch directly. If using file capture, you may need another shell/session to inspect the growing log:

```bash
tail -n 300 "$CAPSTONE_TMP_ROOT/capstone-qemu-run.txt"
```

Inside the VM, the validated manual sequence is:

```text
root
insmod /capstone.ko
/capstone-test.user /test-domains/my_domain.dom
/capstone-test.user /test-domains/fib.dom
```

Expected success markers for `my_domain.dom` include:
- `Ok, good file.`
- `Found 2 segments`
- `Loadable executable segment found.`
- `Created domain ID = 0`
- `Called dom (1-th time) retval = 0`

Expected success marker compared to the old failure mode:
- the old QEMU assert `env->priv < PRV_C failed` should be absent

---

## 6. Important caveat about the current sample flow

The current `my_first_domain` path now has a native Capstone sample flow for the validated domain example:
- in-tree LLVM `clang` compiles the code,
- in-tree `ld.lld` links the final domain ELF natively,
- the output ELF carries `EM_CAPSTONE` (`0x103`),
- the Buildroot userspace loader accepts both `EM_RISCV` and `EM_CAPSTONE`,
- the VM path has been revalidated with `/capstone-test.user /test-domains/my_domain.dom`.

This does **not** yet imply that the whole broader hosted toolchain/runtime stack is ready for large programs. It means the minimal sample domain flow is known-good and should be treated as the baseline runtime regression.

---

## 6. Fast QEMU runtime smoke without rebuilding the rootfs on every iteration

This is the preferred runtime revalidation path when you want to check that the
current domain baseline still works, but you do **not** want to rebuild
`rootfs.ext2` for each small test-domain change.

It uses:
- `capstone/tests/runtime-qemu/build-domain.sh`
- `capstone/tests/runtime-qemu/run-domain-smoke.py`
- `capstone/tests/runtime-qemu/run-smoke.sh`

The key trick is that QEMU exports a host directory into the guest via `9p`, and
the guest mounts that shared directory before running `/capstone-test.user`.

### 6.1 Run the one-command smoke test

```bash
cd "$CAPSTONE_REPO_ROOT" && \
bash capstone/tests/runtime-qemu/run-smoke.sh \
  > "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-smoke-wrapper.txt" 2>&1
```

Inspect the wrapper output:

```bash
sed -n '1,220p' "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-smoke-wrapper.txt"
```

Inspect the full QEMU serial log:

```bash
sed -n '1,260p' "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-smoke.log"
```

Expected markers include:
- `Ok, good file.`
- `Loadable executable segment found.`
- `Created domain ID = 0`
- `Called dom (1-th time) retval = 0`

### 6.2 Build a different tiny domain and run it through the same harness

```bash
mkdir -p "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-share"
cd "$CAPSTONE_REPO_ROOT" && \
bash capstone/tests/runtime-qemu/build-domain.sh \
  capstone/tests/runtime-qemu/domains/write_42.c \
  "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-share/write_42.dom" \
  > "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-build-domain.txt" 2>&1
```

Then run it:

```bash
cd "$CAPSTONE_REPO_ROOT" && \
python3 capstone/tests/runtime-qemu/run-domain-smoke.py \
  --share-dir "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-share" \
  --log-file "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-direct.log" \
  "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-share/write_42.dom" \
  > "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-direct-wrapper.txt" 2>&1
```

---

## 7. Hosted smoke tests without QEMU: current recommendation

Do **not** add a separate large Capstone-specific hosted smoke suite yet unless a
newly fixed hosted blocker needs a dedicated regression test.

Why:
- the tree already has existing driver/sysroot patterns for other targets, e.g.
  `clang/test/Driver/linux-cross.cpp` and `clang/test/Driver/baremetal-sysroot.cpp`
- Capstone already has a focused hosted driver regression in
  `clang/test/Driver/capstone-linux-toolchain.c`
- the current real hosted blocker is still earlier and very concrete:
  the current Buildroot glibc sysroot rejects Capstone when ordinary hosted
  headers are included (`bits/wordsize.h: unsupported ABI`)

So for now:
1. keep the driver regression test,
2. keep probing the real hosted compile path against the real sysroot,
3. once a hosted blocker is fixed, add the smallest exact regression for that case.

---

## 7. What to inspect when something fails

### Build failure in LLVM/Clang/LLD
Inspect:
- `$CAPSTONE_TMP_ROOT/capstone-llvm-build.txt`

### Backend regression test failure
Inspect:
- `$CAPSTONE_TMP_ROOT/capstone-lit-codegen.txt`
- `$CAPSTONE_TMP_ROOT/capstone-lit-focused.txt`

### Clang builtin regression failure
Inspect:
- `$CAPSTONE_TMP_ROOT/capstone-clang-builtins.txt`

### Sample domain build failure
Inspect:
- `$CAPSTONE_TMP_ROOT/capstone-my-domain-build.txt`

### Rootfs rebuild failure
Inspect:
- `$CAPSTONE_TMP_ROOT/capstone-buildroot-make.txt`

### Runtime/QEMU failure
Inspect:
- `$CAPSTONE_TMP_ROOT/capstone-qemu-run.txt`
- and compare against the latest proof logs saved in `capstone/agent-handoff/`

---

## 8. Maintenance rule for future sessions

If a session changes the validated flow or achieves a new milestone, it should update the persistent handoff bundle under:
- `$CAPSTONE_HANDOFF_DIR`

At minimum, keep these files accurate:
- `README.md`
- `new-chat-prompt.md`
- `capstone-agent-test-instructions.md`
- `capstone-backend-status-for-llm.md`

And replace or refresh the proof logs if the known-good baseline changes.
