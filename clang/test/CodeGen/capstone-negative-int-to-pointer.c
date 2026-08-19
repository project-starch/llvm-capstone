// RUN: %clang_cc1 -triple capstone64-unknown-elf -ffreestanding -emit-llvm -o - %s | FileCheck %s
//
// C-21 (frontend half): a NEGATIVE integer constant cast to a pointer crashed
// clang's constant evaluator.
//
// PointerExprEvaluator::VisitCastExpr read the value back with getZExtValue(),
// which asserts the value fits in 64 bits. It first extended the constant to
// getTypeSize(pointer type) -- 128 here, because a Capstone pointer's
// REPRESENTATION is a capability while the address it carries is 64 bits -- so a
// negative constant came back with every one of the high bits set and the assert
// fired.
//
// This is not a corner case: AT_FDCWD is -100 and musl casts it in every *at()
// wrapper, so the whole family failed to compile. 38 musl sources came back when
// the clamp landed.
//
// The test is a COMPILE, not a pattern match: before the fix this file did not
// produce output at all, it aborted the compiler.

void *p = (void *) -100;
// CHECK: @p ={{.*}} global ptr addrspace(200)

// The same shape inside a function body, which takes a different evaluator path.
void *f(void) {
  return (void *) -100;
}
// CHECK-LABEL: define{{.*}} ptr addrspace(200) @f(

// A negative constant that needs more than 32 bits, so the clamp is exercised
// rather than merely not reached.
void *g(void) {
  return (void *) -4294967296;
}
// CHECK-LABEL: define{{.*}} ptr addrspace(200) @g(
