// RUN: %clang_cc1 -triple capstone64-unknown-elf -emit-llvm -o - %s | FileCheck %s

void test_cap_return(void *cap, unsigned long code) {
  __builtin_capstone_cap_return(cap, code);
}

void test_cap_exit(void *cap, unsigned long code) {
  __builtin_capstone_cap_exit(cap, code);
}

// CHECK-LABEL: define{{.*}} @test_cap_return(
// CHECK: call addrspace(200) void @llvm.capstone.cap.return.p200
// CHECK-NEXT: unreachable

// CHECK-LABEL: define{{.*}} @test_cap_exit(
// CHECK: call addrspace(200) void @llvm.capstone.cap.exit.p200
// CHECK-NEXT: unreachable

