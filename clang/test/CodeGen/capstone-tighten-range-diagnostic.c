// An out-of-range TIGHTEN immediate is a FRONT-END error at the call site since
// SemaCapstone landed (2026-09-05).  Before that, no Sema existed for any
// __builtin_capstone_* builtin (SemaChecking.cpp dispatched riscv but not
// capstone), so the value reached instruction selection and the compiler died
// with "fatal error: error in backend: Capstone TIGHTEN immediate must be in
// range 0-31!" and a stack dump -- measured 2026-09-04 and pinned by the
// earlier version of this file.  That backend route is still pinned at the IR
// level by llvm/test/CodeGen/Capstone/fatal-tighten-range.ll, where nothing
// stops a bad immediate; from C it is now unreachable.
//
// capstone-tighten-nonconst.c is the control for the non-constant route, and
// Sema/capstone-builtins-range.c holds the full range table.
//
// MUTATION: change 999 to 7 -> no diagnostic is produced and -verify fails the
// RUN line (performed 2026-09-05).
//
// RUN: %clang_cc1 -triple capstone64-unknown-elf -target-feature +m -ffreestanding -fsyntax-only -verify %s

void *f(void *p) { return __builtin_capstone_cap_tighten(p, 999); } // expected-error {{argument value 999 is outside the valid range [0, 7]}}
