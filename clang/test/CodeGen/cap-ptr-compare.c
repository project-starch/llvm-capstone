// RUN: %clang_cc1 -triple capstone64-unknown-elf -ffreestanding -emit-llvm -O0 -o - %s | FileCheck %s
// Verify pointer comparisons and pointer-difference comparisons do not cause
// ICmpInst type-mismatch assertions on the Capstone target.
//
// They cannot any more, which is the update: the mismatch existed because the
// pointer integer type was the POINTER width (i128) while ptrdiff_t was the
// ADDRESS width (i64), so a difference had to be truncated before it could be
// compared. Both are the address width now, so there is nothing to truncate --
// the two operands of the icmp are the same type by construction.

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

// Pointer-difference comparison: (a - b) < constant, entirely at address width.
// CHECK-LABEL: @cmp_ptrdiff(
// CHECK: ptrtoint ptr addrspace(200) {{.*}} to i64
// CHECK: ptrtoint ptr addrspace(200) {{.*}} to i64
// CHECK-NOT: trunc
// CHECK: icmp slt i64
int cmp_ptrdiff(const unsigned char *a, const unsigned char *b) {
    return (a - b) < (ptrdiff_t)4;
}
