# Current recommended next step

This file intentionally contains the **current** recommendation only.
It is expected to change over time and should be updated when the project state changes.

## Current recommendation

The next smallest meaningful step toward the real goal (running serious software such as SPEC-like tests, FFmpeg, sqlite, libpng, etc.) is:

> make the smallest possible hosted Capstone source file survive **real sysroot header parsing** against the current Linux Buildroot sysroot

## Why this is the right next step

The project already has a validated native sample-domain flow:
- in-tree `clang` compiles the sample domain,
- in-tree `ld.lld` links it as native `EM_CAPSTONE`,
- the userspace loader accepts `EM_CAPSTONE`,
- the sample domain executes successfully in the Capstone QEMU/Buildroot runtime.

That means the next real blocker is no longer the sample domain path, but the broader hosted toolchain/runtime path.

We now also have a faster runtime regression path for the current domain baseline:
- `capstone/tests/runtime-qemu/run-smoke.sh` boots QEMU once,
- mounts a host directory via `9p`,
- and revalidates the domain runtime path without rebuilding `rootfs.ext2` on each iteration.

So the runtime baseline is covered well enough to move back to the first hosted blocker.

Large software such as FFmpeg/sqlite/libpng/SPEC will depend on:
- a normal hosted entry flow,
- startup files / crt objects,
- libc/sysroot integration,
- ordinary hosted linking assumptions,
- and runtime support outside the special domain-harness path.

The first concrete failure currently observed on that path is not linker emulation anymore. It is earlier:
- compiling a normal hosted source against the current Buildroot glibc sysroot trips
  `bits/wordsize.h: unsupported ABI`
- the current Capstone target macros do not yet satisfy the RISC-V-oriented glibc header expectations in that sysroot

## Concrete form of the next step

1. Reproduce the real hosted header failure with the smallest possible source, e.g. `#include <stdio.h>`.
2. Decide the intended compatibility contract for the current sysroot layer:
   - temporary RISC-V-compatible preprocessor surface,
   - or a different sysroot/header strategy.
3. Fix only that first blocker.
4. Re-test the same tiny hosted source.
5. Only after header parsing works, move on to crt/startfiles and full linking.

## What not to jump to yet

Do **not** jump straight to FFmpeg/sqlite/libpng unless the hosted smoke test already works.

The point of this step is to remove the narrowest real missing piece in the hosted path, not to jump ahead to full applications before the headers even parse.

