// C-83: __builtin_return_address gives a C pointer. llvm.returnaddress yields an
// address-space-0 pointer, a 64-bit address on Capstone, while `void *` is a
// capability; clang makes the C value from the address (ptrtoint, inttoptr), an
// untagged pointer that carries the return address and no authority. Before, the
// AS0 value went out as it was: passed as an argument it was addrspacecast and
// the call lowering asserted, and returned it was stored as 8 bytes into the
// 16-byte return slot and reloaded as a capability. __builtin_frame_address
// already asks for the stack's address space and is the control, unchanged, as
// is a target whose pointers are all one type.
//
// RUN: %clang_cc1 -triple capstone64-unknown-elf -O0 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple capstone64-unknown-elf -O1 -emit-llvm -o - %s | FileCheck %s --check-prefix=OPT
// RUN: %clang_cc1 -triple riscv64-unknown-elf -O1 -emit-llvm -o - %s | FileCheck %s --check-prefix=RV
// RUN: %clang_cc1 -triple capstone64-unknown-elf -O0 -S -o /dev/null %s
// RUN: %clang_cc1 -triple capstone64-unknown-elf -O2 -S -o /dev/null %s

void sink(void *);
void *slot;

// CHECK-LABEL: define {{.*}}ptr addrspace(200) @returned(
// CHECK:       [[RA:%.*]] = call addrspace(200) ptr @llvm.returnaddress(i32 0)
// CHECK-NEXT:  [[A:%.*]] = ptrtoint ptr [[RA]] to i64
// CHECK-NEXT:  [[P:%.*]] = inttoptr i64 [[A]] to ptr addrspace(200)
// CHECK-NEXT:  ret ptr addrspace(200) [[P]]
// OPT-LABEL:   @returned(
// OPT-NOT:     store
// OPT:         inttoptr i{{64|128}} {{.*}} to ptr addrspace(200)
// RV-LABEL:    @returned(
// RV:          [[RA:%.*]] = tail call ptr @llvm.returnaddress(i32 0)
// RV-NEXT:     ret ptr [[RA]]
void *returned(void) { return __builtin_return_address(0); }

// CHECK-LABEL: @passed(
// CHECK:       [[RA:%.*]] = call addrspace(200) ptr @llvm.returnaddress(i32 0)
// CHECK-NEXT:  [[A:%.*]] = ptrtoint ptr [[RA]] to i64
// CHECK-NEXT:  [[P:%.*]] = inttoptr i64 [[A]] to ptr addrspace(200)
// CHECK-NEXT:  call addrspace(200) void @sink(ptr addrspace(200) noundef [[P]])
// CHECK-NOT:   addrspacecast
void passed(void) { sink(__builtin_return_address(0)); }

// CHECK-LABEL: @stored(
// CHECK:       [[RA:%.*]] = call addrspace(200) ptr @llvm.returnaddress(i32 1)
// CHECK-NEXT:  [[A:%.*]] = ptrtoint ptr [[RA]] to i64
// CHECK-NEXT:  [[P:%.*]] = inttoptr i64 [[A]] to ptr addrspace(200)
// CHECK-NEXT:  store ptr addrspace(200) [[P]], ptr addrspace(200) @slot, align 16
void stored(void) { slot = __builtin_return_address(1); }

// CHECK-LABEL: @frame(
// CHECK:       [[FA:%.*]] = call addrspace(200) ptr addrspace(200) @llvm.frameaddress.p200(i32 1)
// CHECK-NEXT:  ret ptr addrspace(200) [[FA]]
void *frame(void) { return __builtin_frame_address(1); }
