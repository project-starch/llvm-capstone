// RUN: %clang_cc1 -triple capstone64-unknown-elf -ffreestanding -emit-llvm \
// RUN:   -O0 -o - %s | FileCheck %s

// A negative subscript uses a sign-extended pointer-width GEP index. CodeGen's
// alignment refinement must not assume that constant indexes fit in uint64_t.

// CHECK-LABEL: define{{.*}} i8 @load_before
// CHECK: getelementptr inbounds i8, ptr addrspace(200) %{{.*}}, i128 -1
// CHECK: load i8
char load_before(char *pointer) {
  return pointer[-1];
}
