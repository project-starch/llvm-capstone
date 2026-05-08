# Hosted libc / OS / syscall analysis for Capstone

This note is intended as a handoff artifact for another LLM or future session.
It captures the current diagnostic state of the hosted Capstone bring-up **without implementing any fixes**.

Repository root in the current setup is available as `$CAPSTONE_REPO_ROOT`.
Temporary logs are written under `$CAPSTONE_TMP_ROOT` (default: `/tmp/capstone`).

---

## 1. Executive summary

### Current verified state
- The current validated runtime baseline is the **domain path**, not a general hosted Linux user-space path.
- The sample domain path already works end-to-end:
  - in-tree LLVM `clang` compiles the domain,
  - in-tree `ld.lld` links it as `EM_CAPSTONE`,
  - the userspace loader accepts `EM_CAPSTONE`,
  - the domain executes inside the current QEMU/Buildroot runtime.
- There is also a fast `9p`-based QEMU smoke path:
  - `capstone/tests/runtime-qemu/run-smoke.sh`
  - it revalidates the domain runtime baseline without rebuilding `rootfs.ext2` for every small test iteration.

### Current hosted blocker
The current minimal hosted blocker is **not** linker emulation anymore.
The first real blocker is earlier:
- including a normal libc header (e.g. `#include <stdio.h>`) against the current Buildroot sysroot fails in:
  - `bits/wordsize.h: unsupported ABI`

### Why it fails
The current Capstone target surface does **not** match the ABI assumptions baked into the current Buildroot sysroot:
- the sysroot is a normal `riscv64 + glibc + lp64d` Linux sysroot,
- the current Capstone target exposes a capability/purecap-like pointer model by default,
- glibc headers expect the normal RISC-V Linux ABI macros and pointer size model.

### Important separate blocker
Even if header parsing worked, there is a later mismatch:
- the Capstone Linux driver now requests
  - `/lib/ld-linux-capstone64-lp64d.so.1`
- but the current sysroot contains
  - `/lib/ld-linux-riscv64-lp64d.so.1`

So there are **at least two** distinct hosted blockers:
1. **header / ABI model mismatch**
2. **dynamic loader naming mismatch**

---

## 2. Key diagnostics that were run

### Macro dumps
- `$CAPSTONE_TMP_ROOT/capstone64-macros.txt`
- `$CAPSTONE_TMP_ROOT/clang-riscv64-macros.txt`
- `$CAPSTONE_TMP_ROOT/gcc-riscv64-buildroot-macros.txt`
- compact comparison:
  - `$CAPSTONE_TMP_ROOT/macro-key-diff.txt`

### Hosted header probe
Source used:
- `$CAPSTONE_TMP_ROOT/hosted-stdio-smoke.c`

Logs:
- Capstone target against real sysroot:
  - `$CAPSTONE_TMP_ROOT/capstone64-stdio-Ev.txt`
- Clang RISC-V baseline against same sysroot:
  - `$CAPSTONE_TMP_ROOT/clang-riscv64-stdio-Ev.txt`
- GCC Buildroot baseline against same sysroot:
  - `$CAPSTONE_TMP_ROOT/gcc-riscv64-stdio-Ev.txt`

### Loader mismatch snapshot
- `$CAPSTONE_TMP_ROOT/loader-mismatch.txt`

### Current userspace loader binary format
- `$CAPSTONE_TMP_ROOT/capstone-test-user-file.txt`
- `$CAPSTONE_TMP_ROOT/capstone-test-user-readobj.txt`

---

## 3. What the current Buildroot sysroot actually is

From:
- `capstone/caplifive-buildroot/build/.config`
- `capstone/caplifive-buildroot/buildroot/toolchain/toolchain-buildroot/Config.in`
- `capstone/caplifive-buildroot/buildroot/package/glibc/glibc.mk`

Current toolchain/sysroot facts:
- `BR2_riscv=y`
- `BR2_ARCH="riscv64"`
- `BR2_GCC_TARGET_ABI="lp64d"`
- `BR2_TOOLCHAIN_USES_GLIBC=y`
- `BR2_TOOLCHAIN_BUILDROOT_GLIBC=y`

So the current Linux sysroot is:

> **riscv64 + glibc + lp64d**

This is **not** a Capstone-native hosted libc/sysroot.
It is a normal RISC-V Linux user-space environment.

---

## 4. The current userspace loader/runtime is still ordinary RISC-V Linux user-space

The current guest userspace helper `capstone-test.user` is not a Capstone-native hosted ELF.
It is currently a normal RISC-V Linux user program.

Evidence:
- `file` output in `$CAPSTONE_TMP_ROOT/capstone-test-user-file.txt`
- `llvm-readobj` output in `$CAPSTONE_TMP_ROOT/capstone-test-user-readobj.txt`

Observed facts:
- `Format: elf64-littleriscv`
- `Arch: riscv64`
- `Machine: EM_RISCV`
- interpreter:
  - `/lib/ld-linux-riscv64-lp64d.so.1`

So the currently working runtime baseline is:
- Linux kernel + OpenSBI + QEMU machine,
- ordinary RISC-V glibc user-space,
- plus a Capstone-specific kernel module and SBI extension,
- plus Capstone domains loaded via `/dev/capstone`.

This is very different from saying that a normal `capstone64-unknown-linux-gnu` user-space executable already works.

---

## 5. Why the current hosted header probe fails

### Relevant glibc header
From:
- `capstone/caplifive-buildroot/build/host/riscv64-buildroot-linux-gnu/sysroot/usr/include/bits/wordsize.h`

The critical check is:

```c
#if __riscv_xlen == (__SIZEOF_POINTER__ * 8)
# define __WORDSIZE __riscv_xlen
#else
# error unsupported ABI
#endif
```

### Current Capstone macro surface
From `$CAPSTONE_TMP_ROOT/macro-key-diff.txt`, the Capstone target currently exposes at least:

```text
#define __capstone 1
#define __capstone_xlen 64
#define __capstone_float_abi_soft 1
#define __linux__ 1
#define __ELF__ 1
#define __POINTER_WIDTH__ 128
#define __SIZEOF_POINTER__ 16
#define __INTPTR_WIDTH__ 64
#define __SIZE_WIDTH__ 64
```

### RISC-V baseline macro surface
For the normal RISC-V Linux ABI the baseline is:

```text
#define __riscv 1
#define __riscv_xlen 64
#define __riscv_float_abi_double 1
#define __LP64__ 1
#define _LP64 1
#define __POINTER_WIDTH__ 64
#define __SIZEOF_POINTER__ 8
```

### Immediate conclusions
There are multiple mismatches.

#### Mismatch A: macro namespace
The current glibc headers expect RISC-V macros such as:
- `__riscv`
- `__riscv_xlen`
- `__riscv_float_abi_double`

The Capstone target currently defines:
- `__capstone_xlen`
- `__capstone_float_abi_soft`

#### Mismatch B: pointer model
The current Capstone target uses a capability/purecap-like default pointer model:
- `__SIZEOF_POINTER__ = 16`
- `__POINTER_WIDTH__ = 128`

This is consistent with:
- `clang/lib/Basic/Targets/Capstone.h`
- `Capstone64TargetInfo` sets:
  - `PointerWidth = PointerAlign = 128`
  - a capability-oriented data layout
  - default address space remapped to AS200

But the current RISC-V glibc headers expect the ordinary LP64 world:
- `__SIZEOF_POINTER__ = 8`
- `__riscv_xlen = 64`

So even if `__riscv_xlen` were magically present, the glibc check would still fail as long as the pointer size remained 16 bytes.

#### Mismatch C: float ABI expectations
The current Linux driver test for Capstone expects:
- `-dynamic-linker /lib/ld-linux-capstone64-lp64d.so.1`

That implies a hosted Linux ABI story shaped like `lp64d`.
But the current frontend macro surface shows:
- `__capstone_float_abi_soft`

So the current frontend default ISA/ABI surface is not yet aligned with the hosted Linux driver assumptions.

---

## 6. Why `__capstone_float_abi_soft` appears right now

From:
- `clang/lib/Basic/Targets/Capstone.cpp`
- `llvm/lib/TargetParser/CapstoneISAInfo.cpp`

`CapstoneTargetInfo::handleTargetFeatures()` sets the default ABI from:
- `ISAInfo->computeDefaultABI()`

And `computeDefaultABI()` returns:
- `lp64d` if the ISA has `d`
- `lp64f` if the ISA has `f`
- otherwise `lp64`

But the default negative-feature population path uses:
- `rv64i`

So unless the target feature set grows beyond the minimal base, the frontend can easily end up without `f/d` and therefore expose:
- soft-float oriented macros

This is a separate frontend/target-default diagnostic issue.

---

## 7. Current hosted Linux driver facts

From:
- `clang/lib/Driver/ToolChains/CommonArgs.cpp`
- `clang/lib/Driver/ToolChains/Linux.cpp`
- `clang/test/Driver/capstone-linux-toolchain.c`

The Linux driver currently does the following for Capstone hosted triples:
- `capstone32` -> `elf32lcapstone`
- `capstone64` -> `elf64lcapstone`
- dynamic loader name for `capstone64` defaults to:
  - `/lib/ld-linux-capstone64-lp64d.so.1`

This is a valid and useful driver-side regression step, but it does **not** imply that a normal hosted Capstone Linux executable already builds or runs.

---

## 8. Current Linux / OS / syscall situation

This section answers the question: if a future `sqlite` or libc issues system calls, what would they actually talk to?

### The short version
Right now there **is** an OS in the working runtime path:
- Linux 6.1.26 is booted inside QEMU.

But it is **not** yet a Linux userspace ABI that has been brought up specifically for `capstone64-unknown-linux-gnu`.

The currently validated runtime arrangement is:
- OpenSBI 1.2
- Linux 6.1.26
- Buildroot rootfs
- ordinary `riscv64` glibc user-space
- Capstone kernel module `/dev/capstone`
- Capstone SBI extension used by that kernel module
- Capstone domains loaded by a user-space helper

### Evidence for the Linux baseline
From:
- `capstone/caplifive-buildroot/configs/qemu_capstone_defconfig`
- `capstone/caplifive-buildroot/configs/kernel.config`
- `capstone/caplifive-buildroot/run-qemu.sh`

Observed facts:
- Buildroot target arch is RISC-V 64-bit
- Linux kernel version is `6.1.26`
- OpenSBI version is `1.2`
- QEMU machine is:
  - `virt-capstone`
- QEMU CPU is:
  - `rv64,sstc=false,h=false`

### How the currently working domain path interacts with the OS
The currently working domain flow is **not** a normal Linux process ABI demonstration for Capstone code.
Instead it works like this:

1. `capstone-test.user` is an ordinary **RISC-V Linux** userspace program.
2. It uses ordinary libc/Linux services such as:
   - `open`
   - `ioctl`
   - `mmap`
   - `close`
   - file I/O and ELF parsing.
3. It talks to `/dev/capstone`.
4. The kernel module `capstone.ko` handles those `ioctl`s and `mmap`s.
5. The kernel module then issues SBI calls via `sbi_ecall(...)` into the Capstone runtime/firmware layer.
6. The Capstone domain is then created and invoked through that module/SBI path.

So the currently working path is:

> **ordinary RISC-V Linux userspace** -> **Linux syscalls** -> **kernel module `/dev/capstone`** -> **SBI extension** -> **Capstone domain runtime**

That is very different from:

> **Capstone-native hosted Linux process** -> **native Capstone Linux syscall ABI** -> **kernel**

### Source evidence for the current user/kernel contract
Userspace library / loader side:
- `capstone/caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.c`

It uses ordinary Linux interfaces such as:
- `open(CAPSTONE_DEV_PATH, ...)`
- `ioctl(...)`
- `mmap(...)`
- `close(...)`

Kernel module side:
- `capstone/caplifive-buildroot/package/modcapstone/module/capstone.c`

It provides:
- `device_ioctl`
- `device_mmap`
- and forwards work through `sbi_ecall(...)`

Shared user/kernel ABI for the device is in:
- `capstone/caplifive-buildroot/package/modcapstone/include/capstone.h`

This is a `/dev/capstone` userspace-kernel interface, not a normal generic Linux system-call ABI for Capstone processes.

---

## 9. How libc normally fits into a hosted Linux stack

For a normal hosted Linux program such as `sqlite`:
- the program is linked against a libc,
- libc provides wrappers such as `open`, `read`, `write`, `mmap`, `poll`, etc.,
- those wrappers eventually issue Linux syscalls using the architecture's user/kernel syscall ABI,
- the kernel returns results using that same ABI.

So yes:
- `sqlite` and libc do rely on syscalls, directly or indirectly.

That means a real hosted Capstone Linux user-space story needs **all** of the following to agree:
1. compiler target ABI
2. C data model (pointer size, struct layout, varargs, etc.)
3. startup files / crt
4. libc headers and libc implementation
5. syscall calling convention and syscall numbers
6. kernel userspace ABI handling
7. signal frame / TLS / auxiliary vector / dynamic loader details

If any of those are still RISC-V-only while the user program is now truly `capstone64-unknown-linux-gnu`, the hosted stack is incomplete.

---

## 10. What Linux syscall support exists in the source tree today

There is Linux syscall support in `llvm-libc`, but only for standard architectures currently wired in there.

From:
- `libc/src/__support/OSUtil/linux/syscall.h`
- `libc/src/__support/OSUtil/linux/riscv/syscall.h`

Observed facts:
- `llvm-libc` has a Linux syscall layer for:
  - x86_32
  - x86_64
  - aarch64
  - arm
  - riscv
- the RISC-V syscall implementation uses:
  - registers `a0`..`a5` and `a7`
  - instruction `ecall`

So `llvm-libc` today knows how to issue **RISC-V Linux syscalls**, not a Capstone-specific Linux syscall ABI.

Also, there are no direct `Capstone` references under `libc/` today.
So there is no in-tree evidence yet of a dedicated Capstone port of `llvm-libc`.

---

## 11. libc options and what they imply

### Option A: Keep using the current glibc-based Buildroot stack as the hosted target

#### What this means
Try to make `capstone64-unknown-linux-gnu` compatible enough with the current Linux userspace ABI expectations, or introduce a hosted mode compatible with the current `riscv64 + glibc + lp64d` stack.

#### Pros
- Closest to the real goal of running serious Linux software.
- Reuses the existing Buildroot rootfs, loader, crt objects, dynamic linking, and package ecosystem.
- Least distance to `sqlite` / `libpng` / `ffmpeg` as Linux applications.

#### Cons
- Current Capstone default pointer/data model is not compatible with the present RISC-V glibc sysroot.
- If the goal is a genuinely native capability-default Linux userspace ABI, then glibc porting becomes a large architecture/ABI project.

#### Practical conclusion
This is the most natural **end-state** if the goal is serious Linux software, but it may or may not be the easiest first native ABI target.

---

### Option B: Use `musl` instead of glibc for the first native Linux libc port

Buildroot already supports `musl` as an internal toolchain C library option.

Evidence:
- `capstone/caplifive-buildroot/buildroot/toolchain/toolchain-buildroot/Config.in`
- `capstone/caplifive-buildroot/buildroot/package/musl/musl.mk`

#### Pros
- Lighter and simpler than glibc.
- Usually a more realistic first libc for a new Linux architecture/ABI bring-up.
- Still a real Linux userspace libc, unlike `newlib`.

#### Cons
- Still requires a coherent Linux userspace ABI.
- Does not magically remove the need to define pointer model, syscall ABI, loader naming, crt, TLS, signals, etc.

#### Practical conclusion
If a truly native Capstone Linux userspace ABI is required, `musl` is a very plausible candidate for the **first serious libc port**.

---

### Option C: Use `uClibc-ng`

Buildroot supports it too.

Evidence:
- `capstone/caplifive-buildroot/buildroot/toolchain/toolchain-buildroot/Config.in`
- `capstone/caplifive-buildroot/buildroot/package/uclibc/uclibc.mk`

#### Practical conclusion
Possible in theory, but today it looks less attractive than `musl` as a first-class path for a new hosted Linux ABI.

---

### Option D: Use `newlib`

#### Important distinction
`newlib` is primarily a bare-metal / embedded libc.
It is **not** the normal answer for a Linux hosted userspace goal like `sqlite`.

#### Practical conclusion
`newlib` could be relevant for freestanding or domain-oriented experiments, but not as the main path to serious Linux user-space software.

---

### Option E: Use `klibc`

`klibc` is generally aimed at tiny early-boot / initramfs-style Linux use cases, not full general-purpose user-space.

#### Practical conclusion
Not a good main path for the project goal of running serious hosted Linux software.

---

### Option F: Use `llvm-libc`

#### Current source-tree state
There is Linux and RISC-V support in `llvm-libc`, but no visible Capstone-specific integration under `libc/`.

#### Practical conclusion
Interesting long-term or research direction, but not currently the shortest path to a working hosted Capstone Linux environment.

---

## 12. The core architectural question that must be answered

The project needs to decide what hosted Linux user-space ABI it wants.

### Model 1: RISC-V-compatible hosted mode
A hosted Linux mode whose userspace ABI is intentionally compatible with the existing RISC-V Linux LP64D world.

#### Consequence
- existing kernel and libc stack become much more reusable,
- easiest path toward serious applications,
- capability-heavy/purecap semantics may need to be reduced or explicitly opt-in in this hosted mode.

### Model 2: Native Capstone Linux userspace ABI
A genuinely native Capstone hosted ABI with its own coherent user-space contract.

#### Consequence
This is a much larger project. It would need agreement across:
- compiler ABI surface,
- pointer/data model,
- loader naming,
- crt/startfiles,
- libc,
- syscall ABI,
- kernel user ABI handling,
- TLS/signals/auxv/vDSO details.

### Model 3: Split model
Keep two distinct worlds:
- current domain runtime world (capability/purecap oriented)
- separate conventional hosted Linux world (more RISC-V-like ABI for normal Linux apps)

#### Consequence
This may be the most practical bootstrap strategy if the immediate goal is to reach `sqlite`, `libpng`, etc., without first solving a full native capability-default Linux ABI.

---

## 13. Current best working hypothesis

Based on the current evidence, the most practical short-term view is:

1. The currently working domain runtime path should not be confused with a hosted Linux user-space bring-up.
2. The present Buildroot/Linux/sysroot stack is still fundamentally RISC-V user-space.
3. A future hosted Capstone stack must decide whether it wants:
   - RISC-V-compatible hosted mode, or
   - a genuinely native Capstone Linux ABI.
4. If the latter is desired, a first serious libc port would probably be more realistic with `musl` than with `glibc`.
5. If the immediate objective is fastest progress toward real applications, a split or compatibility-oriented hosted mode is likely the lower-risk path.

---

## 14. Concrete source pointers for the next LLM/session

### Capstone target / ABI surface
- `clang/lib/Basic/Targets/Capstone.h`
- `clang/lib/Basic/Targets/Capstone.cpp`
- `llvm/lib/TargetParser/CapstoneISAInfo.cpp`

### Linux driver hosted assumptions
- `clang/lib/Driver/ToolChains/Linux.cpp`
- `clang/lib/Driver/ToolChains/CommonArgs.cpp`
- `clang/test/Driver/capstone-linux-toolchain.c`

### Current sysroot / toolchain configuration
- `capstone/caplifive-buildroot/build/.config`
- `capstone/caplifive-buildroot/buildroot/toolchain/toolchain-buildroot/Config.in`
- `capstone/caplifive-buildroot/buildroot/package/glibc/glibc.mk`
- `capstone/caplifive-buildroot/buildroot/package/musl/musl.mk`
- `capstone/caplifive-buildroot/buildroot/package/uclibc/uclibc.mk`

### Header blocker
- `capstone/caplifive-buildroot/build/host/riscv64-buildroot-linux-gnu/sysroot/usr/include/bits/wordsize.h`
- `capstone/caplifive-buildroot/build/host/riscv64-buildroot-linux-gnu/sysroot/usr/include/bits/types.h`
- `capstone/caplifive-buildroot/build/host/riscv64-buildroot-linux-gnu/sysroot/usr/include/gnu/stubs.h`

### Current Linux runtime / QEMU
- `capstone/caplifive-buildroot/configs/qemu_capstone_defconfig`
- `capstone/caplifive-buildroot/configs/kernel.config`
- `capstone/caplifive-buildroot/run-qemu.sh`
- `capstone/tests/runtime-qemu/run-domain-smoke.py`

### Current `/dev/capstone` userspace-kernel contract
- `capstone/caplifive-buildroot/package/modcapstone/include/capstone.h`
- `capstone/caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.c`
- `capstone/caplifive-buildroot/package/modcapstone/module/capstone.c`

### Linux syscall layer in llvm-libc
- `libc/src/__support/OSUtil/linux/syscall.h`
- `libc/src/__support/OSUtil/linux/riscv/syscall.h`

---

## 15. Important caution

This note is diagnostic only.
It does **not** recommend a concrete implementation patch yet.
The main purpose is to prevent future sessions from conflating:
- a working domain runtime sample,
with
- a coherent hosted Linux userspace ABI and libc story.

