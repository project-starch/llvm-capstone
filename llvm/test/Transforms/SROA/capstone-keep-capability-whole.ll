; RUN: opt -passes=sroa -S < %s | FileCheck %s
; RUN: opt -mtriple=x86_64-unknown-linux-gnu -passes=sroa -S < %s \
; RUN:   | FileCheck %s --check-prefix=OTHER
;
; Capstone: SROA must not split an alloca so that a capability ends up
; misaligned in a new alloca or cut in two. A capability keeps its tag only
; when it is copied as one aligned 16-byte access; a piece copied as bytes or
; words arrives untagged and faults at its first use.
;
; The shape is mruby's mrb_method_t -- { i32 flags, union { proc pointer,
; function pointer } } -- copied in, its flags updated, copied out: SROA made
; [0,4) a scalar and [4,32) an `alloca [28 x i8], align 4`, the capability 12
; bytes into it, and mruby's method proc pointer arrived untagged.

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"
target triple = "capstone64-unknown-elf"

%struct.method = type { i32, ptr addrspace(200) }
%struct.plain = type { i32, i64, i64 }

declare void @llvm.memcpy.p200.p200.i64(ptr addrspace(200), ptr addrspace(200), i64, i1)

; The alloca stays whole, and the flags update is still there.
; CHECK-LABEL: @method_copy(
; CHECK-NOT:   alloca [28 x i8]
; CHECK-NOT:   alloca [12 x i8], align 4
; CHECK:       %m = alloca %struct.method, align 16
; CHECK:       call {{.*}}@llvm.memcpy{{.*}}(ptr addrspace(200) align 16 %m, ptr addrspace(200) align 16 %src, i64 32
; CHECK:       %f = load i32, ptr addrspace(200) %m, align 16
; CHECK:       %g = or i32 %f, 4
; CHECK:       store i32 %g, ptr addrspace(200) %m, align 16
; CHECK:       call {{.*}}@llvm.memcpy{{.*}}(ptr addrspace(200) align 16 %dst, ptr addrspace(200) align 16 %m, i64 32
; CHECK:       ret void
; Off Capstone the rule does not apply (AMDGPU's fat pointers are wider than
; their index too, and carry no tag), and SROA makes the split described above.
; OTHER-LABEL: @method_copy(
; OTHER:       alloca [28 x i8], align 4
define void @method_copy(ptr addrspace(200) %dst, ptr addrspace(200) %src) addrspace(200) {
entry:
  %m = alloca %struct.method, align 16, addrspace(200)
  call void @llvm.memcpy.p200.p200.i64(ptr addrspace(200) align 16 %m, ptr addrspace(200) align 16 %src, i64 32, i1 false)
  %f = load i32, ptr addrspace(200) %m, align 16
  %g = or i32 %f, 4
  store i32 %g, ptr addrspace(200) %m, align 16
  call void @llvm.memcpy.p200.p200.i64(ptr addrspace(200) align 16 %dst, ptr addrspace(200) align 16 %m, i64 32, i1 false)
  ret void
}

; The capability itself is still copied as a capability: every copy of the
; slot at offset 16 is a 16-byte, 16-aligned transfer or a pointer load/store.
; CHECK-LABEL: @method_copy_keeps_pointer_width(
; CHECK:       %m = alloca %struct.method, align 16
; CHECK:       store i32 0, ptr addrspace(200) %m, align 16
; CHECK:       ret void
define void @method_copy_keeps_pointer_width(ptr addrspace(200) %dst, ptr addrspace(200) %src) addrspace(200) {
entry:
  %m = alloca %struct.method, align 16, addrspace(200)
  call void @llvm.memcpy.p200.p200.i64(ptr addrspace(200) align 16 %m, ptr addrspace(200) align 16 %src, i64 32, i1 false)
  store i32 0, ptr addrspace(200) %m, align 16
  call void @llvm.memcpy.p200.p200.i64(ptr addrspace(200) align 16 %dst, ptr addrspace(200) align 16 %m, i64 32, i1 false)
  ret void
}

; Control: the same shape with no capability in it is split as before.
; CHECK-LABEL: @plain_copy(
; CHECK-NOT:   alloca %struct.plain
; CHECK:       ret void
define void @plain_copy(ptr addrspace(200) %dst, ptr addrspace(200) %src) addrspace(200) {
entry:
  %m = alloca %struct.plain, align 16, addrspace(200)
  call void @llvm.memcpy.p200.p200.i64(ptr addrspace(200) align 16 %m, ptr addrspace(200) align 16 %src, i64 24, i1 false)
  %f = load i32, ptr addrspace(200) %m, align 16
  %g = or i32 %f, 4
  store i32 %g, ptr addrspace(200) %m, align 16
  call void @llvm.memcpy.p200.p200.i64(ptr addrspace(200) align 16 %dst, ptr addrspace(200) align 16 %m, i64 24, i1 false)
  ret void
}
