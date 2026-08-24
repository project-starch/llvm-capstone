// AS200 carries a 128-bit pointer whose ADDRESS is 64 bits, declared as the
// index width in the datalayout. The pointer itself is unchanged: still a full
// capability, still 16 bytes.

// RUN: %clang_cc1 -triple capstone64-unknown-elf -ffreestanding -emit-llvm -o - %s | FileCheck %s

// CHECK: target datalayout = {{.*}}p200:128:128:128:64

_Static_assert(sizeof(void *) == 16, "a pointer is still a full capability");

// ... and the integers that describe it are ADDRESS-sized, which is what the C
// types already said. clang's own intptr_t/size_t/ptrdiff_t used to be built
// from the POINTER width instead, so every pointer difference was formed in
// i128 and narrowed afterwards. This test pinned that as intended behaviour.
//
// It is not. Once i128 stopped being the capability carrier it became an
// ordinary illegal type on RV64, and a difference over a non-power-of-two
// element size came out of the back end as a call to __divti3 -- which a
// freestanding domain has no way to satisfy. Nothing about a 64-bit address
// needs 128-bit arithmetic to describe it.
_Static_assert(sizeof(__PTRDIFF_TYPE__) == 8, "ptrdiff_t is address-sized");
_Static_assert(sizeof(__SIZE_TYPE__) == 8, "size_t is address-sized");
_Static_assert(sizeof(__INTPTR_TYPE__) == 8, "intptr_t is address-sized");

// CHECK-LABEL: define {{.*}}i64 @diff(
// CHECK: ptrtoint ptr addrspace(200) {{.*}} to i64
// CHECK-NOT: i128
long long diff(int *a, int *b) { return a - b; }

// The shape that produced the libcall: a 48-byte element, so the exact division
// is a multiply by a modular inverse rather than a shift.
struct big { char x[48]; };
// CHECK-LABEL: define {{.*}}i64 @bigdiff(
// CHECK: ptrtoint ptr addrspace(200) {{.*}} to i64
// CHECK-NOT: i128
long long bigdiff(struct big *a, struct big *b) { return a - b; }
