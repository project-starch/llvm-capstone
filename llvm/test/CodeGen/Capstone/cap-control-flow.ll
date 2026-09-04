; The three domain-boundary intrinsics, each pinned to its exact body.  The one
; that never returns (`return`) must be the LAST instruction of its function
; with no epilogue and no `cjalr zero, 0(ra)` after it; @plain_ret, placed
; directly after it, is the control that proves the negative check can fire --
; it is nothing but that return.  There used to be a fourth, `capexit`: the
; instruction existed in neither the spec, the RTL nor QEMU and was removed
; with its intrinsic and builtin on 2026-09-05 (C-36); CAPENTER's encoding was
; corrected to the decoders' in the same change.
;
; NOTE for Tier 4.4 of the validation plan: the operand models of CAPENTER and
; CAP_RETURN are still to be audited against the decoder (CAP_RETURN's operand
; roles differ from the spec).  If those definitions change, the bodies below
; change with them -- that is the point of pinning them.
;
; MUTATION: in @test_cap_return replace `unreachable` with `ret void` -> a
; The implicit-check-nots on capenter/return/capexit make "exactly the pinned
; instances" a checked property (same mechanism, same demonstration as in
; intrinsics.ll: delete a pinned CHECK line and the instance trips it).
; `cjalr zero, 0(ra)` appears after `return` and the negative check that
; follows it fires.
;
; RUN: llc -mtriple=capstone64 -mattr=+m -verify-machineinstrs < %s | FileCheck %s --implicit-check-not=movc --implicit-check-not='capenter a' --implicit-check-not='return a'
; RUN: llc -mtriple=capstone64 -O0 -stop-after=instruction-select -o - < %s | FileCheck %s --check-prefix=MIR
; RUN: %llc_cap -O0 < %s -o /dev/null
; RUN: %llc_cap -O1 < %s -o /dev/null

declare ptr addrspace(200) @llvm.capstone.cap.call.p200(ptr addrspace(200))
declare ptr addrspace(200) @llvm.capstone.cap.enter.p200(ptr addrspace(200), i64)
declare void @llvm.capstone.cap.return.p200(ptr addrspace(200), i64, i64)

; A domain call preserves NOTHING (the hardware swaps PC and seven CSRs and
; leaves every general register to the callee domain), so a function that must
; still honour the C ABI for its own caller saves ra and s0-s11 around it: 13
; spills, `call a0, a0` (rd == rs1, both fixed to a0), 13 reloads.  Until
; 2026-09-05 the call carried an ordinary call's callee-saved mask and none of
; this was emitted -- s0-s11 were silently clobbered on both implementations.
; sp is preserved by convention; gp is NOT saved yet (C-36b, open).
; CHECK-LABEL: test_cap_call:
; CHECK: # %bb.0:
; CHECK: stc ra, {{[0-9]+}}(sp)
; CHECK: stc s11, 0(sp)
; CHECK: call a0, a0
; CHECK: ldc ra, {{[0-9]+}}(sp)
; CHECK: ldc s11, 0(sp)
; CHECK: cjalr zero, 0(ra)
define ptr addrspace(200) @test_cap_call(ptr addrspace(200) %cap) {
entry:
  %res = call ptr addrspace(200) @llvm.capstone.cap.call.p200(ptr addrspace(200) %cap)
  ret ptr addrspace(200) %res
}

; CAPENTER's registers are fixed by both implementations: the capability in a0,
; the integer in rs2, the result in a1 (rd is encoded 0 and ignored by the RTL's
; decoder and by QEMU).  The result is moved into a0 to be returned.
; CHECK-LABEL: test_cap_enter:
; CHECK: # %bb.0:
; CHECK-NEXT: capenter a0, a1
; CHECK-NEXT: movc a0, a1
; CHECK-NEXT: cjalr zero, 0(ra)
define ptr addrspace(200) @test_cap_enter(ptr addrspace(200) %cap, i64 %arg) {
entry:
  %res = call ptr addrspace(200) @llvm.capstone.cap.enter.p200(ptr addrspace(200) %cap, i64 %arg)
  ret ptr addrspace(200) %res
}

; RETURN reads the sealed-return capability from the rd FIELD (a0 here), the
; re-entry PC from rs1 (an integer, a1) and the asynchronous code from rs2 (a2).
; Until 2026-09-05 the table encoded rd = 0, which the RTL rejects on every
; execution (x0 is never a capability): every compiler-emitted return faulted.
; CHECK-LABEL: test_cap_return:
; CHECK: # %bb.0:
; CHECK-NEXT: return a0, a1, a2
; CHECK-NOT: cjalr
; CHECK-NOT: cincoffsetimm sp
define void @test_cap_return(ptr addrspace(200) %cap, i64 %pc, i64 %code) {
entry:
  call void @llvm.capstone.cap.return.p200(ptr addrspace(200) %cap, i64 %pc, i64 %code)
  unreachable
}

; CONTROL: an ordinary function IS a return and nothing else.  It sits right
; after the no-return function, so its negative checks are bounded by
; this label and are shown to see a real `cjalr zero, 0(ra)` when one exists.
; CHECK-LABEL: plain_ret:
; CHECK: # %bb.0:
; CHECK-NEXT: cjalr zero, 0(ra)
define void @plain_ret() {
entry:
  ret void
}

; MIR-LABEL: name:            test_cap_call
; The fixed-a0 pseudo with the no-registers-preserved mask: the register
; allocator must assume the callee domain clobbers everything.
; MIR: PseudoDomCall csr_noregs, implicit-def $c10, implicit-def dead $c1, implicit killed $c10

; MIR-LABEL: name:            test_cap_enter
; MIR: CAPENTER killed $c10, {{.*}}implicit-def dead $c10, implicit-def $c11
