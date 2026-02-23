; RUN: llc < %s -mtriple=capstone64  | FileCheck %s

declare void @take_ptr_int(ptr addrspace(200), i64)

; CHECK-LABEL: test_call:
; CHECK: auipc [[REG1:a[0-9]+]], %pcrel_hi(take_ptr_int)
; CHECK: addi [[REG2:a[0-9]+]], [[REG1]], %pcrel_lo
; CHECK: cincoffset [[REG2]], gp, [[REG2]]
; CHECK: cjalr ra, 0([[REG2]])
define void @test_call() {
entry:
  %ptr = inttoptr i128 4660 to ptr addrspace(200)
  call void @take_ptr_int(ptr addrspace(200) %ptr, i64 100)
  ret void
}

; CHECK-LABEL: call_ptr:
; CHECK: cjalr ra, 0(a0)
define void @call_ptr(ptr addrspace(200) %fp) {
entry:
  call void %fp()
  ret void
}

declare void @variadic_func(i32, ...)

; CHECK-LABEL: test_call_vararg:
; CHECK: li a0, 3
; CHECK: movc a1, zero
; CHECK: movc a2, zero
; CHECK: movc a3, zero
; CHECK: cjalr ra, 0(a4)
define void @test_call_vararg() {
entry:
  call void (i32, ...) @variadic_func(i32 3, ptr addrspace(200) null, ptr addrspace(200) null, ptr addrspace(200) null)
  ret void
}