# Native sample validation summary

This file is the compact human-written proof that the minimal native Capstone sample flow has been validated.

## What was validated

The sample domain in `capstone/my_first_domain/` was validated with the following flow:

1. Built by the in-tree LLVM `clang` for `capstone64-unknown-elf`.
2. Linked by the in-tree `ld.lld`.
3. Produced an ELF identified by `llvm-readobj` as:
   - `Format: elf64-littlecapstone`
   - `Arch: capstone64`
   - `Machine: 0x103` (`EM_CAPSTONE`)
4. Accepted by the patched Buildroot userspace loader.
5. Executed successfully inside the Capstone QEMU / Buildroot environment.

## Runtime success markers

The validated run reached the following observable state inside the guest:

- `insmod /capstone.ko` succeeded
- `/capstone-test.user /test-domains/my_domain.dom` printed:
  - `Ok, good file.`
  - `Found 2 segments`
  - `Loadable executable segment found.`
  - `Created domain ID = 0`
  - `Called dom (1-th time) retval = 0`

## Why this matters

This proves that the old manual ELF-header rewrite workaround is no longer needed for the default sample-domain flow:
- the sample is linked natively as `EM_CAPSTONE`
- the loader accepts it
- the runtime path executes correctly in QEMU

## Scope of this proof

This validates the **native sample-domain path** only.
It does **not** yet imply that the broader hosted toolchain/runtime path is ready for normal user-space programs such as FFmpeg/sqlite/libpng/SPEC.

