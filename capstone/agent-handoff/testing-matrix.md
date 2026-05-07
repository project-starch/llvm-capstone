# Capstone testing matrix and current recommendations

This file is the compact, current map of how the Capstone work should be tested.
It is intended to stay up to date as the bring-up moves from backend work to hosted user-space work.

All example commands below follow the user's preferred workflow:
- source the shared path defaults from `capstone/tests/capstone-test-env.sh`,
- write logs to `$CAPSTONE_TMP_ROOT/...` (default: `/tmp/capstone/...`),
- then inspect those log files.

## Quick cheat sheet

First, set up the common defaults once per shell:

```bash
cd "$(git rev-parse --show-toplevel)"
source capstone/tests/capstone-test-env.sh
```

Then the most useful test entry points are:

```bash
# Backend / SelectionDAG regressions
"$CAPSTONE_LLVM_LIT" -sv \
  "$CAPSTONE_REPO_ROOT/llvm/test/CodeGen/Capstone"

# Clang builtins
"$CAPSTONE_LLVM_LIT" -sv \
  "$CAPSTONE_REPO_ROOT/clang/test/CodeGen/capstone-builtins.c" \
  "$CAPSTONE_REPO_ROOT/clang/test/CodeGen/builtins-capstone.c"

# LLD / ELF emulation
"$CAPSTONE_LLVM_LIT" -sv \
  "$CAPSTONE_REPO_ROOT/lld/test/ELF/emulation-capstone.s"

# Hosted Linux driver regression
"$CAPSTONE_LLVM_LIT" -sv \
  "$CAPSTONE_REPO_ROOT/clang/test/Driver/capstone-linux-toolchain.c"

# Current runtime smoke in QEMU (single boot, shared 9p directory)
bash "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-smoke.sh"
```

---

## 1. Why multiple test layers are needed

The project has several distinct bring-up layers, and each one can fail independently:

1. **LLVM backend / SelectionDAG lowering**
   - instruction selection,
   - frame lowering,
   - capability loads/stores,
   - control flow lowering.
2. **Clang frontend / builtins**
   - builtin lowering to IR intrinsics.
3. **LLD / ELF flavor**
   - `EM_CAPSTONE`,
   - ELF emulation names,
   - native object/executable identity.
4. **Clang driver / Linux hosted plumbing**
   - sysroot search,
   - startup files,
   - link line construction,
   - dynamic linker path.
5. **Current domain runtime path in QEMU**
   - boot,
   - loader acceptance,
   - domain execution.
6. **Future hosted user-space path**
   - headers,
   - crt objects,
   - libc/sysroot ABI,
   - ordinary Linux executables.

A single test layer cannot cover all of this. Fast tests localize compiler/linker bugs; slower QEMU tests prove that the current runtime baseline still works.

---

## 2. Current test layers

### A. Backend regression tests (`llvm-lit`, very fast)

**What they prove**
- SelectionDAG codegen still works for the implemented Capstone subset.

**Where**
- `llvm/test/CodeGen/Capstone/`

**Run**

```bash
cd "$CAPSTONE_REPO_ROOT" && \
"$CAPSTONE_LLVM_LIT" -sv \
  "$CAPSTONE_REPO_ROOT/llvm/test/CodeGen/Capstone" \
  > "$CAPSTONE_TMP_ROOT/capstone-testing-codegen.txt" 2>&1
```

Inspect:

```bash
sed -n '1,260p' "$CAPSTONE_TMP_ROOT/capstone-testing-codegen.txt"
```

**When to run**
- after backend lowering / isel / frame / memory changes.

---

### B. Clang builtin / frontend checks (`llvm-lit`, very fast)

**What they prove**
- Clang builtins still lower to the expected Capstone IR intrinsics.

**Where**
- `clang/test/CodeGen/capstone-builtins.c`
- `clang/test/CodeGen/builtins-capstone.c`

**Run**

```bash
cd "$CAPSTONE_REPO_ROOT" && \
"$CAPSTONE_LLVM_LIT" -sv \
  "$CAPSTONE_REPO_ROOT/clang/test/CodeGen/capstone-builtins.c" \
  "$CAPSTONE_REPO_ROOT/clang/test/CodeGen/builtins-capstone.c" \
  > "$CAPSTONE_TMP_ROOT/capstone-testing-clang-builtins.txt" 2>&1
```

Inspect:

```bash
sed -n '1,220p' "$CAPSTONE_TMP_ROOT/capstone-testing-clang-builtins.txt"
```

---

### C. LLD / ELF emulation checks (`llvm-lit`, fast)

**What they prove**
- `ld.lld` still accepts Capstone ELF emulation names,
- produced ELFs remain native `EM_CAPSTONE`.

**Where**
- `lld/test/ELF/emulation-capstone.s`

**Run**

```bash
cd "$CAPSTONE_REPO_ROOT" && \
"$CAPSTONE_LLVM_LIT" -sv \
  "$CAPSTONE_REPO_ROOT/lld/test/ELF/emulation-capstone.s" \
  > "$CAPSTONE_TMP_ROOT/capstone-testing-lld.txt" 2>&1
```

Inspect:

```bash
sed -n '1,220p' "$CAPSTONE_TMP_ROOT/capstone-testing-lld.txt"
```

---

### D. Clang Linux driver checks (`llvm-lit`, fast)

**What they prove**
- the Linux driver builds the correct link line for hosted Capstone triples,
- current tested behavior includes:
  - `-m elf64lcapstone`
  - `-dynamic-linker /lib/ld-linux-capstone64-lp64d.so.1`

**Where**
- `clang/test/Driver/capstone-linux-toolchain.c`

**Run**

```bash
cd "$CAPSTONE_REPO_ROOT" && \
"$CAPSTONE_LLVM_LIT" -sv \
  "$CAPSTONE_REPO_ROOT/clang/test/Driver/capstone-linux-toolchain.c" \
  > "$CAPSTONE_TMP_ROOT/capstone-testing-driver.txt" 2>&1
```

Inspect:

```bash
sed -n '1,220p' "$CAPSTONE_TMP_ROOT/capstone-testing-driver.txt"
```

**Important limitation**
- this is a **driver command-line regression test**, not a guest runtime test.
- it does **not** prove that a normal hosted Linux executable already builds or runs.

---

### E. Current native sample-domain baseline (build + runtime)

**What it proves**
- the current validated domain ABI path still works end to end.

**Where**
- `capstone/my_first_domain/`

**Build**

```bash
cd "$CAPSTONE_REPO_ROOT/capstone/my_first_domain" && \
LLVM_BIN="$CAPSTONE_LLVM_BIN" ./build.sh \
  > "$CAPSTONE_TMP_ROOT/capstone-testing-my-domain-build.txt" 2>&1
```

Inspect:

```bash
sed -n '1,220p' "$CAPSTONE_TMP_ROOT/capstone-testing-my-domain-build.txt"
```

Optional ELF inspection:

```bash
"$CAPSTONE_LLVM_READOBJ" -h \
  "$CAPSTONE_REPO_ROOT/capstone/my_first_domain/my_domain.dom" \
  > "$CAPSTONE_TMP_ROOT/capstone-testing-my-domain-readobj.txt" 2>&1
```

---

### F. QEMU runtime smoke with shared directory (single boot, no rootfs rebuild per iteration)

**What it proves**
- QEMU boots,
- guest `9p` mount works,
- `capstone.ko` loads,
- `/capstone-test.user` can execute a Capstone domain directly from a host-shared directory,
- current runtime baseline can be revalidated without rebuilding `rootfs.ext2` every time.

**Where**
- `capstone/tests/runtime-qemu/`

**Run**

```bash
cd "$CAPSTONE_REPO_ROOT" && \
bash capstone/tests/runtime-qemu/run-smoke.sh \
  > "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-smoke-wrapper.txt" 2>&1
```

Inspect the driver-side wrapper output:

```bash
sed -n '1,220p' "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-smoke-wrapper.txt"
```

Inspect the full serial/QEMU log:

```bash
sed -n '1,260p' "$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-smoke.log"
```

**Why this layer matters**
- It is slower than `lit`, but it catches regressions that `lit` cannot see:
  - guest boot/runtime issues,
  - loader/runtime acceptance,
  - QEMU/device interactions,
  - module/runtime integration.

**Why it is implemented with `9p`**
- The guest kernel already has `CONFIG_NET_9P`, `CONFIG_NET_9P_VIRTIO`, and `CONFIG_9P_FS`.
- Exporting a host directory into the guest allows fast iteration on test domains.
- This avoids rebuilding the rootfs image for every small smoke-case change.

---

## 3. Hosted smoke tests without QEMU: should we add them now?

### Short answer
Not as a separate Capstone-specific suite **yet**.

### Why not yet
There are already existing patterns elsewhere in the tree for this style of testing, especially in:
- `clang/test/Driver/linux-cross.cpp`
- `clang/test/Driver/baremetal-sysroot.cpp`
- the general Clang driver/sysroot tests

Those tests are good for checking:
- sysroot include discovery,
- library search paths,
- startup file selection,
- `-dynamic-linker` behavior,
- GCC toolchain detection.

For Capstone specifically, we already have the first dedicated hosted-driver regression:
- `clang/test/Driver/capstone-linux-toolchain.c`

At the moment, a broader hosted smoke suite would mostly duplicate current driver coverage while the **first real hosted blocker** is still earlier and very concrete:
- including glibc headers from the current Buildroot sysroot already fails in `bits/wordsize.h` with `unsupported ABI`.

So the recommended approach is:
1. keep the current driver regression,
2. keep probing the real hosted compile path against the real sysroot,
3. once the first sysroot/libc compatibility blocker is fixed, add the smallest focused regression for that exact case.

This avoids inventing a parallel suite that mostly re-states a currently known failure.

---

## 4. Current validated status summary

As of the latest checked state:
- backend `llvm-lit` Capstone tests pass,
- Clang builtin tests pass,
- LLD Capstone emulation test passes,
- the hosted Linux driver regression for `capstone64-unknown-linux-gnu` passes,
- the native sample-domain path is still valid,
- the new `9p`-based QEMU smoke path works and avoids per-iteration rootfs rebuilds.

What is **not** validated yet:
- general hosted Linux user-space build + run,
- libc/sysroot ABI compatibility for normal user-space sources,
- larger application builds.

---

## 5. Recommended default test bundle after a non-trivial change

If a change touches the backend/toolchain/runtime in a way that is broader than a one-line tweak, the default focused bundle should be:

1. `llvm/test/CodeGen/Capstone`
2. `clang/test/CodeGen/capstone-builtins.c` and `builtins-capstone.c`
3. `lld/test/ELF/emulation-capstone.s`
4. `clang/test/Driver/capstone-linux-toolchain.c`
5. `capstone/tests/runtime-qemu/run-smoke.sh`

That combination gives one fast compiler/linker pass plus one runtime sanity check.

