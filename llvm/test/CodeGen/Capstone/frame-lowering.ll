; RUN: llc -mtriple=capstone64 < %s | FileCheck %s

target triple = "capstone64"

declare i32 @callee(i32)

define i32 @frame_alloca(i32 %x) {
; CHECK-LABEL: frame_alloca:
; CHECK: cincoffsetimm sp, sp, -16
; CHECK: cincoffsetimm a2, sp, 12
; CHECK: sw a0, 0(a2)
; CHECK: cincoffsetimm sp, sp, 16
; CHECK: cjalr zero, 0(ra)
entry:
  %slot = alloca i32, align 4, addrspace(200)
  store i32 %x, ptr addrspace(200) %slot, align 4
  %y = load i32, ptr addrspace(200) %slot, align 4
  %z = add i32 %y, 1
  ret i32 %z
}

define i32 @caller(i32 %x) {
; CHECK-LABEL: caller:
; CHECK: cincoffsetimm sp, sp, -48
; CHECK: stc ra, 32(sp)
; CHECK: cjalr ra, 0(a1)
; CHECK: ldc ra, 32(sp)
; CHECK: cincoffsetimm sp, sp, 48
; CHECK: cjalr zero, 0(ra)
entry:
  %slot = alloca i32, align 4, addrspace(200)
  store i32 %x, ptr addrspace(200) %slot, align 4
  %call = call i32 @callee(i32 %x)
  %y = load i32, ptr addrspace(200) %slot, align 4
  %z = add i32 %call, %y
  ret i32 %z
}

define void @large_frame() {
; CHECK-LABEL: large_frame:
; CHECK: lui [[ALLOC:a[0-9]+]], 1048575
; CHECK-NEXT: addi [[ALLOC]], [[ALLOC]], -16
; CHECK-NEXT: cincoffset sp, sp, [[ALLOC]]
; CHECK: .cfi_def_cfa_offset 4112
; CHECK: lui [[FREE:a[0-9]+]], 1
; CHECK-NEXT: addi [[FREE]], [[FREE]], 16
; CHECK-NEXT: cincoffset sp, sp, [[FREE]]
; CHECK: cjalr zero, 0(ra)
entry:
  %slot = alloca [4096 x i8], align 16, addrspace(200)
  call void asm sideeffect "", ""()
  ret void
}


