// AS200 carries a 128-bit pointer whose ADDRESS is 64 bits, declared as the
// index width in the datalayout. The pointer itself is unchanged: still a full
// capability, still 16 bytes.

// RUN: %clang_cc1 -triple capstone64-unknown-elf -ffreestanding -emit-llvm -o - %s | FileCheck %s

// CHECK: target datalayout = {{.*}}p200:128:128:128:64

_Static_assert(sizeof(void *) == 16, "a pointer is still a full capability");

// The narrower index width is a property of the pointer TYPE, not of the
// integers clang casts pointers to: a difference is still formed in i128 and
// narrowed afterwards. Pinned because it is the thing most easily assumed the
// other way round -- the index width does not move pointer arithmetic out of
// i128 on its own.
// CHECK-LABEL: define {{.*}}i64 @diff(
// CHECK: ptrtoint ptr addrspace(200) {{.*}} to i128
long long diff(int *a, int *b) { return a - b; }
