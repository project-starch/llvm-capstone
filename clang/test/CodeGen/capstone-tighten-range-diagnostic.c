// No Sema exists for any __builtin_capstone_* builtin (SemaChecking.cpp
// dispatches riscv but not capstone), so an out-of-range TIGHTEN immediate
// reaches instruction selection and the compiler dies with a backend fatal
// error and a stack dump.  Measured 2026-09-04.  This file pins that behaviour
// so it is visible the day it changes: once Tier 4's SemaCapstone lands, this
// becomes a front-end error at the call site (`-verify`, no backend involved)
// and the RUN line and CHECK move accordingly.
//
// capstone-tighten-nonconst.c is the control: a NON-constant argument is
// already rejected by Sema through the `_Constant` prototype, which is why the
// non-constant route to the selector's fatal error is recorded as unreachable.
//
// MUTATION: change 999 to 7 -> clang succeeds and `not` fails the RUN line
// (performed 2026-09-04).
//
// RUN: not %clang_cc1 -triple capstone64-unknown-elf -target-feature +m -ffreestanding -O0 -S -o /dev/null %s 2>&1 | FileCheck %s

// CHECK: fatal error: error in backend: Capstone TIGHTEN immediate must be in range 0-31!
void *f(void *p) { return __builtin_capstone_cap_tighten(p, 999); }
