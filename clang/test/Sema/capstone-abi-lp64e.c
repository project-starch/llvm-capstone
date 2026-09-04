// `-target-abi lp64e` is rejected on capstone64 (since 2026-09-05).  The RISCV
// copy accepted it and installed a plain 64-bit-pointer datalayout, incoherent
// with a target whose pointers are 128-bit capabilities in address space 200;
// the front end would then have compiled against one layout and the backend
// against another.  lp64 is the control: accepted, and the file compiles.
//
// MUTATION: change `lp64e` to `lp64` in the first RUN line -> clang succeeds and
// `not` fails that line (performed 2026-09-05).
//
// RUN: not %clang_cc1 -triple capstone64-unknown-elf -target-abi lp64e -fsyntax-only %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple capstone64-unknown-elf -target-abi lp64 -fsyntax-only %s

// CHECK: error: unknown target ABI 'lp64e'
int f(int *p) { return *p; }
