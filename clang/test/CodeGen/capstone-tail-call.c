// A sibling call must end the function with a JUMP.  Today the backend selects
// it as a call (`cjalr ra`) and emits no epilogue and no return after it, so
// control falls off the end of the function.  Cause and fix are described in
// llvm/test/CodeGen/Capstone/tail-call.ll; this file pins the same property
// from C, at the two optimisation levels where clang forms sibling calls, and
// keeps the -fno-optimize-sibling-calls shape that the CoreMark and SQLite
// build scripts have relied on since June as the control.
//
// Fixed 2026-09-04 (selectCall selects PseudoTAILIndirect for a TAIL node); the
// target register is any member of GPCRTC, hence the [at] pattern.
//
// MUTATION: the NOSIB arm *is* the mutation -- the same function built with
// sibling calls disabled contains `cjalr ra` followed by a real return, which is
// exactly what the default arm's `CHECK-NOT: cjalr ra` rejects.  The two arms
// cross-control each other.
//
// RUN: %clang_cc1 -triple capstone64-unknown-elf -target-feature +m -ffreestanding -O1 -S -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple capstone64-unknown-elf -target-feature +m -ffreestanding -Os -S -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple capstone64-unknown-elf -target-feature +m -ffreestanding -O1 -fno-optimize-sibling-calls -S -o - %s | FileCheck %s --check-prefix=NOSIB

long f(long);

// CHECK-LABEL: g:
// CHECK-NOT: cjalr ra
// CHECK: cjalr zero, 0({{[at][0-9]}})
//
// NOSIB-LABEL: g:
// NOSIB: cjalr ra, 0(a{{[0-9]+}})
// NOSIB: ldc ra, {{[0-9]+}}(sp)
// NOSIB: cincoffsetimm sp, sp, {{[0-9]+}}
// NOSIB-NEXT: cjalr zero, 0(ra)
long g(long x) { return f(x + 1); }
