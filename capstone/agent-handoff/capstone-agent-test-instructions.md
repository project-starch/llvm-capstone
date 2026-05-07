# Capstone test/run instructions for future agent sessions

This file is a practical handoff note for future agent sessions working on the Capstone backend/toolchain in `/home/alexey/dev/llvm-capstone`.

The user explicitly prefers that terminal output be redirected to files in `/tmp` and then inspected from there, rather than reading command output directly.

---

## 0. Repositories / important paths

Workspace root:
- `/home/alexey/dev/llvm-capstone`

LLVM tree:
- `/home/alexey/dev/llvm-capstone/llvm`

LLVM build dir used in this work:
- `/home/alexey/dev/llvm-capstone/llvm/build`
- sometimes also `/home/alexey/dev/llvm-capstone/llvm/cmake-build-debug`

Capstone-related runtime repos in-tree:
- `/home/alexey/dev/llvm-capstone/capstone/caplifive-buildroot`
- `/home/alexey/dev/llvm-capstone/capstone/capstone-qemu`
- `/home/alexey/dev/llvm-capstone/capstone/my_first_domain`

---

## 1. General rule for running commands

Always redirect output to `/tmp/...` and inspect the file afterwards.

Examples:

```bash
cd /home/alexey/dev/llvm-capstone && \
cmake --build /home/alexey/dev/llvm-capstone/llvm/build --target check-llvm > /tmp/capstone-check-llvm.txt 2>&1
```

```bash
sed -n '1,220p' /tmp/capstone-check-llvm.txt
```

For shell scripts, if you want command tracing, prefer:

```bash
bash -x ./build.sh > /tmp/my-domain-build.txt 2>&1
```

or, if the script already has `set -x`, just redirect its output:

```bash
./build.sh > /tmp/my-domain-build.txt 2>&1
```

---

## 2. Fast backend regression checks

### 2.1 Run focused Capstone llc tests

Use this when you changed backend lowering, instruction selection, frame lowering, memory lowering, etc.

```bash
cd /home/alexey/dev/llvm-capstone/llvm && \
/home/alexey/dev/llvm-capstone/llvm/build/bin/llvm-lit -sv \
  /home/alexey/dev/llvm-capstone/llvm/test/CodeGen/Capstone \
  > /tmp/capstone-lit-codegen.txt 2>&1
```

Inspect:

```bash
sed -n '1,260p' /tmp/capstone-lit-codegen.txt
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
cd /home/alexey/dev/llvm-capstone/llvm && \
/home/alexey/dev/llvm-capstone/llvm/build/bin/llvm-lit -sv \
  /home/alexey/dev/llvm-capstone/llvm/test/CodeGen/Capstone/cap-control-flow.ll \
  /home/alexey/dev/llvm-capstone/llvm/test/CodeGen/Capstone/load-store.ll \
  /home/alexey/dev/llvm-capstone/llvm/test/CodeGen/Capstone/frame-lowering.ll \
  > /tmp/capstone-lit-focused.txt 2>&1
```

Inspect:

```bash
sed -n '1,220p' /tmp/capstone-lit-focused.txt
```

---

## 3. Clang frontend builtin checks

Use this after touching `BuiltinsCapstone.td` or `clang/lib/CodeGen/TargetBuiltins/Capstone.cpp`.

```bash
cd /home/alexey/dev/llvm-capstone/llvm && \
/home/alexey/dev/llvm-capstone/llvm/build/bin/llvm-lit -sv \
  /home/alexey/dev/llvm-capstone/clang/test/CodeGen/capstone-builtins.c \
  > /tmp/capstone-clang-builtins.txt 2>&1
```

Inspect:

```bash
sed -n '1,220p' /tmp/capstone-clang-builtins.txt
```

---

## 4. Rebuild LLVM/Clang after source changes

If backend or clang sources changed, rebuild before running tests.

```bash
cd /home/alexey/dev/llvm-capstone && \
cmake --build /home/alexey/dev/llvm-capstone/llvm/build -j$(nproc) \
  > /tmp/capstone-llvm-build.txt 2>&1
```

Inspect tail first:

```bash
tail -n 200 /tmp/capstone-llvm-build.txt
```

If needed, inspect head/middle too:

```bash
sed -n '1,260p' /tmp/capstone-llvm-build.txt
```

---

## 5. Rebuild and run the `my_first_domain` runtime sample

This is the current VM-validated sample flow.

### 5.1 Build the sample domain

`capstone/my_first_domain/build.sh` already uses `set -euxo pipefail`, so command tracing is printed automatically.

```bash
cd /home/alexey/dev/llvm-capstone/capstone/my_first_domain && \
./build.sh > /tmp/capstone-my-domain-build.txt 2>&1
```

Inspect:

```bash
sed -n '1,220p' /tmp/capstone-my-domain-build.txt
```

Expected current behavior:
- `build.sh` now defaults to the in-tree `ld.lld`
- the produced ELF should be native `EM_CAPSTONE`
- the old header rewrite shim is only used when `HOST_LD` is overridden to a non-`ld.lld` linker

### 5.2 Inspect the resulting ELF header

```bash
/home/alexey/dev/llvm-capstone/llvm/build/bin/llvm-readobj -h \
  /home/alexey/dev/llvm-capstone/capstone/my_first_domain/my_domain.dom \
  > /tmp/capstone-my-domain-readobj.txt 2>&1
```

```bash
sed -n '1,220p' /tmp/capstone-my-domain-readobj.txt
```

### 5.3 Disassemble the sample (expect some `<unknown>` for custom instructions)

```bash
/home/alexey/dev/llvm-capstone/llvm/build/bin/llvm-objdump -d \
  /home/alexey/dev/llvm-capstone/capstone/my_first_domain/my_domain.dom \
  > /tmp/capstone-my-domain-objdump.txt 2>&1
```

```bash
sed -n '1,260p' /tmp/capstone-my-domain-objdump.txt
```

### 5.4 Rebuild the userspace loader/module package if you changed loader-side source

Needed after edits under:
- `capstone/caplifive-buildroot/package/modcapstone/...`

```bash
cd /home/alexey/dev/llvm-capstone/capstone/caplifive-buildroot/build && \
make modcapstone-rebuild > /tmp/capstone-modcapstone-rebuild.txt 2>&1
```

```bash
sed -n '1,220p' /tmp/capstone-modcapstone-rebuild.txt
```

### 5.5 Copy the sample into the Buildroot test-domains directory

```bash
cp /home/alexey/dev/llvm-capstone/capstone/my_first_domain/my_domain.dom \
  /home/alexey/dev/llvm-capstone/capstone/caplifive-buildroot/build/target/test-domains/my_domain.dom
```

### 5.6 Rebuild the rootfs image so the new domain lands in the VM image

```bash
cd /home/alexey/dev/llvm-capstone/capstone/caplifive-buildroot/build && \
make > /tmp/capstone-buildroot-make.txt 2>&1
```

Inspect:

```bash
tail -n 200 /tmp/capstone-buildroot-make.txt
```

### 5.7 Run QEMU

`run-qemu.sh` itself does not print shell tracing, so use `bash -x`.

```bash
cd /home/alexey/dev/llvm-capstone/capstone/caplifive-buildroot && \
bash -x ./run-qemu.sh > /tmp/capstone-qemu-run.txt 2>&1
```

Because QEMU is interactive, this command will continue running until the VM exits. In practice, for manual interactive testing it is often easier to run it in a terminal you watch directly. If using file capture, you may need another shell/session to inspect the growing log:

```bash
tail -n 300 /tmp/capstone-qemu-run.txt
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

The current `my_first_domain` path now has a **native Capstone sample flow** for the validated domain example:

- in-tree LLVM `clang` compiles the code,
- in-tree `ld.lld` links the final domain ELF natively,
- the output ELF carries `EM_CAPSTONE` (`0x103`),
- the Buildroot userspace loader source has been updated to accept both `EM_RISCV` and `EM_CAPSTONE`,
- the VM path has been revalidated with `/capstone-test.user /test-domains/my_domain.dom`.

However, this does **not** mean the whole broader toolchain/runtime stack is fully native yet for all programs. It means the minimal sample domain flow now works natively without the old `dd`/header-rewrite hack when using the in-tree `ld.lld` and rebuilt loader.

---

## 7. What to inspect when something fails

### Build failure in LLVM/Clang
Inspect:
- `/tmp/capstone-llvm-build.txt`

### Backend regression test failure
Inspect:
- `/tmp/capstone-lit-codegen.txt`
- `/tmp/capstone-lit-focused.txt`

### Clang builtin regression failure
Inspect:
- `/tmp/capstone-clang-builtins.txt`

### Sample domain build failure
Inspect:
- `/tmp/capstone-my-domain-build.txt`

### Rootfs rebuild failure
Inspect:
- `/tmp/capstone-buildroot-make.txt`

### Runtime/QEMU failure
Inspect:
- `/tmp/capstone-qemu-run.txt`
- and compare against the known-good analysis in `/tmp/capstone-runtime-analysis-for-llm.md`

---

## 8. Current known-good interpretation

As of the current validated run:
- the backend-generated sample domain builds,
- the sample now links natively with in-tree `ld.lld`,
- the resulting domain ELF is `EM_CAPSTONE`,
- the domain ELF is accepted by the loader,
- the kernel module creates the domain,
- the domain executes under Capstone QEMU,
- the previous QEMU assert is gone,
- the currently visible harness result is `retval = 0`.

That means future sessions should treat the present `my_first_domain` flow as the baseline known-good runtime test.

