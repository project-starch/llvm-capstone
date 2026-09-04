; The four domain-boundary intrinsics, each pinned to its exact body.  The two
; that never return (`return`, `capexit`) must be the LAST instruction of their
; function with no epilogue and no `cjalr zero, 0(ra)` after them; @plain_ret,
; placed directly after them, is the control that proves the negative check can
; fire -- it is nothing but that return.
;
; NOTE for Tier 4.4 of the validation plan: the operand models of CAPENTER,
; CAP_RETURN and CAPEXIT are being audited against the decoder (CAPEXIT has no
; RTL counterpart; CAP_RETURN's operand roles differ from the spec).  If those
; definitions change, the bodies below change with them -- that is the point of
; pinning them.
;
; MUTATION: in @test_cap_exit replace `unreachable` with `ret void` -> a
; The implicit-check-nots on capenter/return/capexit make "exactly the pinned
; instances" a checked property (same mechanism, same demonstration as in
; intrinsics.ll: delete a pinned CHECK line and the instance trips it).
; `cjalr zero, 0(ra)` appears after `capexit` and the negative check that
; follows it fires.
;
; RUN: llc -mtriple=capstone64 -mattr=+m -verify-machineinstrs < %s | FileCheck %s --implicit-check-not=movc --implicit-check-not='capenter a' --implicit-check-not='return a' --implicit-check-not='capexit a'
; RUN: llc -mtriple=capstone64 -O0 -stop-after=instruction-select -o - < %s | FileCheck %s --check-prefix=MIR
; RUN: %llc_cap -O0 < %s -o /dev/null
; RUN: %llc_cap -O1 < %s -o /dev/null

declare ptr addrspace(200) @llvm.capstone.cap.call.p200(ptr addrspace(200))
declare i64 @llvm.capstone.cap.enter.p200(ptr addrspace(200))
declare void @llvm.capstone.cap.return.p200(ptr addrspace(200), i64)
declare void @llvm.capstone.cap.exit.p200(ptr addrspace(200), i64)

; CHECK-LABEL: test_cap_call:
; CHECK: # %bb.0:
; CHECK-NEXT: call a0, a0
; CHECK-NEXT: cjalr zero, 0(ra)
define ptr addrspace(200) @test_cap_call(ptr addrspace(200) %cap) {
entry:
  %res = call ptr addrspace(200) @llvm.capstone.cap.call.p200(ptr addrspace(200) %cap)
  ret ptr addrspace(200) %res
}

; CAPENTER's result is defined into a callee-saved capability register and the
; integer half moved out; the spill/reload around it is the cost of that
; operand model today and is pinned so a change to the model is visible here.
; (.cfi_* directives interleave the prologue and epilogue, so -NEXT is used only
; where instructions are truly adjacent: the capenter and the two that consume
; its result.)
; CHECK-LABEL: test_cap_enter:
; CHECK: # %bb.0:
; CHECK-NEXT: cincoffsetimm sp, sp, -16
; CHECK: stc s0, 0(sp)
; CHECK: capenter s0, a0
; CHECK-NEXT: mv a0, s0
; CHECK-NEXT: ldc s0, 0(sp)
; CHECK: cincoffsetimm sp, sp, 16
; CHECK: cjalr zero, 0(ra)
define i64 @test_cap_enter(ptr addrspace(200) %cap) {
entry:
  %res = call i64 @llvm.capstone.cap.enter.p200(ptr addrspace(200) %cap)
  ret i64 %res
}

; CHECK-LABEL: test_cap_return:
; CHECK: # %bb.0:
; CHECK-NEXT: return a0, a1
; CHECK-NOT: cjalr
; CHECK-NOT: cincoffsetimm sp
define void @test_cap_return(ptr addrspace(200) %cap, i64 %code) {
entry:
  call void @llvm.capstone.cap.return.p200(ptr addrspace(200) %cap, i64 %code)
  unreachable
}

; CHECK-LABEL: test_cap_exit:
; CHECK: # %bb.0:
; CHECK-NEXT: capexit a0, a1
; CHECK-NOT: cjalr
; CHECK-NOT: cincoffsetimm sp
define void @test_cap_exit(ptr addrspace(200) %cap, i64 %code) {
entry:
  call void @llvm.capstone.cap.exit.p200(ptr addrspace(200) %cap, i64 %code)
  unreachable
}

; CONTROL: an ordinary function IS a return and nothing else.  It sits right
; after the two no-return functions, so their negative checks are bounded by
; this label and are shown to see a real `cjalr zero, 0(ra)` when one exists.
; CHECK-LABEL: plain_ret:
; CHECK: # %bb.0:
; CHECK-NEXT: cjalr zero, 0(ra)
define void @plain_ret() {
entry:
  ret void
}

; MIR-LABEL: name:            test_cap_call
; The capability register form: $c10 IS $x10, seen at capability width.
; MIR: renamable $c10 = CAP_CALL killed renamable $c10, csr_ilp32_lp64

; MIR-LABEL: name:            test_cap_enter
; MIR: renamable $c10 = CAPENTER killed renamable $c10, csr_ilp32_lp64
; Reading the returned address needs no instruction at all: $x10 IS the low half
; of $c10, so the truncate is a subregister reference that coalesces away.
; MIR-NEXT: renamable $x10 = KILL renamable $x10, implicit killed $c10
; MIR-NEXT: PseudoRET implicit killed $x10
