// Atomic moves and compares of an __intcap keep the capability (C-54): the
// value stays a capability pointer in the atomic instruction, as a pointer's
// does, instead of an i128 that splits it into two integer halves and drops
// the tag. Read-modify-write is refused in Sema (Sema/capstone-intcap.c).
//
// RUN: %clang_cc1 -triple capstone64-unknown-elf -target-feature +a -ffreestanding \
// RUN:   -Wno-atomic-alignment -O1 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s

_Atomic __intcap g;
__intcap h;

// CHECK-LABEL: define {{.*}}@c11_load(
// CHECK-NOT: i128
// CHECK: load atomic ptr addrspace(200), ptr addrspace(200) @g seq_cst, align 16
void *c11_load(void) { return (void *)__c11_atomic_load(&g, __ATOMIC_SEQ_CST); }
// CHECK-LABEL: define {{.*}}@gnu_load(
// CHECK-NOT: i128
// CHECK: load atomic ptr addrspace(200), ptr addrspace(200) @h seq_cst, align 16
void *gnu_load(void) { return (void *)__atomic_load_n(&h, __ATOMIC_SEQ_CST); }
// CHECK-LABEL: define {{.*}}@c11_store(
// CHECK-NOT: i128
// CHECK: store atomic ptr addrspace(200) %{{.*}}, ptr addrspace(200) @g seq_cst, align 16
void c11_store(__intcap v) { __c11_atomic_store(&g, v, __ATOMIC_SEQ_CST); }
// CHECK-LABEL: define {{.*}}@gnu_store(
// CHECK-NOT: i128
// CHECK: store atomic ptr addrspace(200) %{{.*}}, ptr addrspace(200) @h seq_cst, align 16
void gnu_store(__intcap v) { __atomic_store_n(&h, v, __ATOMIC_SEQ_CST); }
// CHECK-LABEL: define {{.*}}@gnu_xchg(
// CHECK-NOT: i128
// CHECK: atomicrmw xchg ptr addrspace(200) @h, ptr addrspace(200) %{{.*}} seq_cst, align 16
__intcap gnu_xchg(__intcap v) { return __atomic_exchange_n(&h, v, __ATOMIC_SEQ_CST); }
// CHECK-LABEL: define {{.*}}@c11_cas(
// CHECK-NOT: i128
// CHECK: cmpxchg ptr addrspace(200) @g, ptr addrspace(200) %{{.*}}, ptr addrspace(200) %{{.*}} seq_cst seq_cst, align 16
_Bool c11_cas(__intcap *e, __intcap v) {
  return __c11_atomic_compare_exchange_strong(&g, e, v, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
}
