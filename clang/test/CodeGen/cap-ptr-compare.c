// RUN: %clang_cc1 -triple capstone64-unknown-elf -ffreestanding -emit-llvm -O0 -o - %s | FileCheck %s
// Verify pointer comparisons and pointer-difference comparisons do not cause
// ICmpInst type-mismatch assertions on the Capstone target, where ptrdiff_t
// (i64) is narrower than the pointer integer type (i128).

typedef unsigned long size_t;
typedef long ptrdiff_t;

// Direct pointer comparison: LLVM emits icmp on ptr addrspace(200) directly.
// No ptrtoint needed; operand types match.
// CHECK-LABEL: @cmp_ptr_arith(
// CHECK: icmp uge ptr addrspace(200)
int cmp_ptr_arith(const unsigned char *base, size_t *sz,
                  const unsigned char *cur) {
    const unsigned char *const end = base + *sz;
    return cur >= end;
}

// Pointer-difference comparison: (a - b) < constant.
// The subtraction path uses ptrtoint to i128, then truncates to ptrdiff_t
// (i64) before the ICmp so both operands are the same type.
// CHECK-LABEL: @cmp_ptrdiff(
// CHECK: ptrtoint ptr addrspace(200) {{.*}} to i128
// CHECK: ptrtoint ptr addrspace(200) {{.*}} to i128
// CHECK: trunc i128 {{.*}} to i64
// CHECK: icmp slt i64
int cmp_ptrdiff(const unsigned char *a, const unsigned char *b) {
    return (a - b) < (ptrdiff_t)4;
}
