// RUN: %clang_cc1 -triple capstone64-unknown-elf -ffreestanding -emit-llvm -o - %s | FileCheck %s

static int callee(void) {
  return 7;
}

struct bundle {
  int (*fn)(void);
  const void *self;
};

static const struct bundle bundle = {
  callee,
  &bundle,
};

int test(void) {
  return bundle.fn() + (bundle.self == &bundle);
}

// CHECK: @bundle = internal addrspace(200) constant %struct.bundle
// CHECK-LABEL: @test(
// CHECK: call addrspace(200) i32
// CHECK: icmp eq ptr addrspace(200)
// CHECK: add nsw i32


