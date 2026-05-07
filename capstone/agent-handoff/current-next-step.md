# Current recommended next step

This file intentionally contains the **current** recommendation only.
It is expected to change over time and should be updated when the project state changes.

## Current recommendation

The next smallest meaningful step toward the real goal (running serious software such as SPEC-like tests, FFmpeg, sqlite, libpng, etc.) is:

> bring up the smallest possible **hosted** Capstone executable flow

## Why this is the right next step

The project already has a validated native sample-domain flow:
- in-tree `clang` compiles the sample domain,
- in-tree `ld.lld` links it as native `EM_CAPSTONE`,
- the userspace loader accepts `EM_CAPSTONE`,
- the sample domain executes successfully in the Capstone QEMU/Buildroot runtime.

That means the next real blocker is no longer the sample domain path, but the broader hosted toolchain/runtime path.

Large software such as FFmpeg/sqlite/libpng/SPEC will depend on:
- a normal hosted entry flow,
- startup files / crt objects,
- libc/sysroot integration,
- ordinary hosted linking assumptions,
- and runtime support outside the special domain-harness path.

## Concrete form of the next step

1. Determine the intended hosted compile/link flow for a normal program under Capstone.
2. Try to build the smallest possible hosted program, for example:
   - `int main() { return 0; }`
   - or a tiny `puts("ok")` program.
3. Identify the **first** real blocker.
4. Fix only that blocker.
5. Rebuild and re-test.

## What not to jump to yet

Do **not** jump straight to FFmpeg/sqlite/libpng unless the hosted smoke test already works.

The point of this step is to find the narrowest real missing piece in the hosted path.

