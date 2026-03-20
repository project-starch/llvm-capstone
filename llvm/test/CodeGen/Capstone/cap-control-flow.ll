; RUN: llc -mtriple=capstone64 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=capstone64 -O0 -stop-after=instruction-select -o - < %s | FileCheck %s --check-prefix=MIR

declare ptr addrspace(200) @llvm.capstone.cap.call.p200(ptr addrspace(200))
declare i64 @llvm.capstone.cap.enter.p200(ptr addrspace(200))
declare void @llvm.capstone.cap.return.p200(ptr addrspace(200), i64)
declare void @llvm.capstone.cap.exit.p200(ptr addrspace(200), i64)

define ptr addrspace(200) @test_cap_call(ptr addrspace(200) %cap) {
entry:
  %res = call ptr addrspace(200) @llvm.capstone.cap.call.p200(ptr addrspace(200) %cap)
  ret ptr addrspace(200) %res
}

define i64 @test_cap_enter(ptr addrspace(200) %cap) {
entry:
  %res = call i64 @llvm.capstone.cap.enter.p200(ptr addrspace(200) %cap)
  ret i64 %res
}

define void @test_cap_return(ptr addrspace(200) %cap, i64 %code) {
entry:
  call void @llvm.capstone.cap.return.p200(ptr addrspace(200) %cap, i64 %code)
  unreachable
}

define void @test_cap_exit(ptr addrspace(200) %cap, i64 %code) {
entry:
  call void @llvm.capstone.cap.exit.p200(ptr addrspace(200) %cap, i64 %code)
  unreachable
}

; CHECK-LABEL: test_cap_call:
; CHECK: call	a0, a0
; CHECK-NEXT: cjalr	zero, 0(ra)

; CHECK-LABEL: test_cap_enter:
; CHECK: capenter
; CHECK: cjalr	zero, 0(ra)

; CHECK-LABEL: test_cap_return:
; CHECK: return	a0, a1
; CHECK-NOT: cjalr	zero, 0(ra)

; CHECK-LABEL: test_cap_exit:
; CHECK: capexit	a0, a1
; CHECK-NOT: cjalr	zero, 0(ra)

; MIR-LABEL: name:            test_cap_call
; MIR: renamable $x10 = CAP_CALL killed renamable $x10, csr_ilp32_lp64

; MIR-LABEL: name:            test_cap_enter
; MIR: renamable $x10 = CAPENTER killed renamable $x10, csr_ilp32_lp64
; MIR: renamable $x10 = PseudoTRUNC_CAP killed renamable $x10


