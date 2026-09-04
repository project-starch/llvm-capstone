// The predefined macros of the Capstone target.  `__CAPSTONE__` and
// `__CAPSTONE_PURECAP__` (since 2026-09-05) are the conventional names code
// tests for; the `__capstone_*` family is the RISCV copy's naming and stays.
// Nothing named `__riscv*` may leak: the target is not RISC-V to the
// preprocessor, and a header that keys on `__riscv` would take the wrong path.
//
// MUTATION: add `Builder.defineMacro("__riscv")` to getTargetDefines -> the
// implicit-check-not below fires (reasoned from the check's construction; the
// defining code is the thing under test, so the mutation is a compiler edit).
//
// RUN: %clang_cc1 -triple capstone64-unknown-elf -E -dM /dev/null | FileCheck %s --implicit-check-not='#define __riscv'
// RUN: %clang_cc1 -triple capstone64-unknown-elf -target-feature +m -target-feature +a -E -dM /dev/null | FileCheck %s --check-prefix=EXT

// CHECK-DAG: #define __CAPSTONE_PURECAP__ 1
// CHECK-DAG: #define __CAPSTONE__ 1
// CHECK-DAG: #define __capstone 1
// CHECK-DAG: #define __capstone_arch_test 1
// CHECK-DAG: #define __capstone_cmodel_medlow 1
// CHECK-DAG: #define __capstone_float_abi_soft 1
// CHECK-DAG: #define __capstone_xlen 64

// EXT-DAG: #define __capstone_atomic 1
// EXT-DAG: #define __capstone_mul 1
// EXT-DAG: #define __capstone_muldiv 1
