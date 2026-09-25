// __intcap / __uintcap_t on capstone64 (intcap plan, Phase B.1): an integer whose
// representation is a capability. A pointer converted to it keeps the whole
// capability and converts back unchanged; its integer value is the address; an
// integer converted to it becomes an untagged capability (inttoptr, which the
// backend bridges into a capability register).
//
// The uintptr_t functions are the control: the same round trip through an
// ordinary integer goes ptrtoint -> inttoptr and loses the capability.
//
// RUN: %clang_cc1 -triple capstone64-unknown-elf -ffreestanding -O1 -disable-llvm-passes \
// RUN:   -emit-llvm -o - %s | FileCheck %s

_Static_assert(sizeof(__intcap) == 16, "stored as a capability");
_Static_assert(_Alignof(__uintcap_t) == 16, "aligned as a capability");
_Static_assert(__SIZEOF_INTCAP__ == 16 && __SIZEOF_UINTCAP__ == 16, "");
_Static_assert(__UINTCAP_MAX__ == 0xffffffffffffffffUL, "ranges over the 64-bit address");
_Static_assert(__INTCAP_MAX__ == 0x7fffffffffffffffL, "");
_Static_assert(_Generic((unsigned __intcap)0, __uintcap_t: 1, default: 0), "spellings agree");
_Static_assert(_Generic((__intcap)0, __intcap_t: 1, default: 0), "");

// CHECK-LABEL: define {{.*}}ptr addrspace(200) @from_ptr(ptr addrspace(200) {{.*}}%p)
// CHECK-NOT: ptrtoint
// CHECK-NOT: inttoptr
// CHECK: ret ptr addrspace(200)
__uintcap_t from_ptr(void *p) { return (__uintcap_t)p; }

// CHECK-LABEL: define {{.*}}ptr addrspace(200) @to_ptr(ptr addrspace(200) {{.*}}%u)
// CHECK-NOT: ptrtoint
// CHECK-NOT: inttoptr
// CHECK: ret ptr addrspace(200)
void *to_ptr(__uintcap_t u) { return (void *)u; }

// CHECK-LABEL: define {{.*}}ptr addrspace(200) @from_ulong(i64
// CHECK: inttoptr i64 %{{.*}} to ptr addrspace(200)
__uintcap_t from_ulong(unsigned long x) { return x; }

// A signed int widens by sign extension first.
// CHECK-LABEL: define {{.*}}ptr addrspace(200) @from_int(i32
// CHECK: sext i32 %{{.*}} to i64
// CHECK: inttoptr i64 %{{.*}} to ptr addrspace(200)
__intcap from_int(int x) { return x; }

// CHECK-LABEL: define {{.*}}i64 @to_ulong(ptr addrspace(200)
// CHECK: ptrtoint ptr addrspace(200) %{{.*}} to i64
unsigned long to_ulong(__uintcap_t u) { return u; }

// CHECK-LABEL: define {{.*}}i32 @to_uint(ptr addrspace(200)
// CHECK: ptrtoint ptr addrspace(200) %{{.*}} to i64
// CHECK: trunc i64 %{{.*}} to i32
unsigned to_uint(__uintcap_t u) { return u; }

// CHECK-LABEL: define {{.*}}i32 @truth(ptr addrspace(200)
// CHECK: ptrtoint ptr addrspace(200) %{{.*}} to i64
// CHECK: icmp ne i64 %{{.*}}, 0
int truth(__uintcap_t u) { return u ? 1 : 2; }

// CHECK-LABEL: define {{.*}}i32 @same(ptr addrspace(200) {{.*}}, ptr addrspace(200)
// CHECK: icmp eq ptr addrspace(200)
int same(__uintcap_t a, __uintcap_t b) { return a == b; }

// Stored as a capability: a 16-byte store of the pointer type.
struct holder { __uintcap_t d; };
// CHECK-LABEL: define {{.*}}void @keep(
// CHECK: store ptr addrspace(200) %{{.*}}, ptr addrspace(200) %{{.*}}, align 16
void keep(struct holder *h, void *p) { h->d = (__uintcap_t)p; }

// CONTROL: an ordinary integer round trip drops the capability.
typedef unsigned long uptr;
// CHECK-LABEL: define {{.*}}ptr addrspace(200) @via_ulong(
// CHECK: ptrtoint ptr addrspace(200)
// CHECK: inttoptr i{{[0-9]+}} %{{.*}} to ptr addrspace(200)
void *via_ulong(void *p) { return (void *)(uptr)p; }
