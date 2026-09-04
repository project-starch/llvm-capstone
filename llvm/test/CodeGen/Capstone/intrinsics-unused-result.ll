; Side-effect modelling of every capability intrinsic, tested by the one thing
; that exposes it: calling the intrinsic and DISCARDING the result at -O2.
;
; An intrinsic whose instruction carries hasSideEffects = 1 (delin, mrev, drop,
; revoke), or that reads/writes a CSR (ccsrrw), or that crosses a domain
; boundary (call, capenter), MUST survive machine DCE -- deleting an unused
; `revoke` would silently skip a revocation.  An intrinsic that is a pure value
; transform (get_tag, shrink, tighten, scc, init, seal) MUST be deleted: its
; result is the only effect it has, so its survival would mean a stray
; hasSideEffects flag pessimising every real program.
;
; Both directions are pinned.  Measured 2026-09-04 on the branch toolchain:
; the six pure ones lower to a bare return; the seven others keep their
; instruction.
;
; NOTE on `init` and `seal`: on the RTL, INIT duplicates and SEAL consumes a
; LINEAR source (Tier 4.1 of the validation plan).  If the linearity contract
; makes either of them a consuming op in the compiler's model, it moves to the
; "survives" group and this file changes with it -- deliberately, so the change
; is visible.
;
; The IR-level counterpart (opt deleting the pure ones) cannot be pinned yet:
; no Capstone intrinsic carries IntrWillReturn, so opt deletes none of them.
; That is a Tier 4 item; see cap-mrev-delin-side-effects.ll.
;
; MUTATION: in @unused_seal return the seal result instead of discarding it ->
; `seal a0, a0` appears and the negative check for seal fires (performed
; 2026-09-04 on a scratch copy).
;
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -verify-machineinstrs < %s | FileCheck %s
; RUN: %llc_cap -O0 < %s -o /dev/null
; RUN: %llc_cap -O1 < %s -o /dev/null

declare i64 @llvm.capstone.cap.get.tag.p200(ptr addrspace(200))
declare ptr addrspace(200) @llvm.capstone.cap.shrink.p200(ptr addrspace(200), i64, i64)
declare ptr addrspace(200) @llvm.capstone.cap.tighten.p200(ptr addrspace(200), i64)
declare ptr addrspace(200) @llvm.capstone.cap.scc.p200(ptr addrspace(200), i64)
declare ptr addrspace(200) @llvm.capstone.cap.init.p200(ptr addrspace(200), i64)
declare ptr addrspace(200) @llvm.capstone.cap.seal.p200(ptr addrspace(200))
declare ptr addrspace(200) @llvm.capstone.cap.delin.p200(ptr addrspace(200))
declare ptr addrspace(200) @llvm.capstone.cap.mrev.p200(ptr addrspace(200))
declare ptr addrspace(200) @llvm.capstone.cap.drop.p200(ptr addrspace(200))
declare ptr addrspace(200) @llvm.capstone.cap.revoke.p200(ptr addrspace(200))
declare ptr addrspace(200) @llvm.capstone.cap.ccsrrw.p200(ptr addrspace(200), i64 immarg)
declare ptr addrspace(200) @llvm.capstone.cap.call.p200(ptr addrspace(200))
declare i64 @llvm.capstone.cap.enter.p200(ptr addrspace(200))

;===--- Pure value transforms: an unused result means NO instruction. ---------===;

; CHECK-LABEL: unused_get_tag:
; CHECK: # %bb.0:
; CHECK-NEXT: cjalr zero, 0(ra)
define void @unused_get_tag(ptr addrspace(200) %p) {
  %r = call i64 @llvm.capstone.cap.get.tag.p200(ptr addrspace(200) %p)
  ret void
}

; CHECK-LABEL: unused_shrink:
; CHECK: # %bb.0:
; CHECK-NEXT: cjalr zero, 0(ra)
define void @unused_shrink(ptr addrspace(200) %p, i64 %a, i64 %b) {
  %r = call ptr addrspace(200) @llvm.capstone.cap.shrink.p200(ptr addrspace(200) %p, i64 %a, i64 %b)
  ret void
}

; CHECK-LABEL: unused_tighten:
; CHECK: # %bb.0:
; CHECK-NEXT: cjalr zero, 0(ra)
define void @unused_tighten(ptr addrspace(200) %p) {
  %r = call ptr addrspace(200) @llvm.capstone.cap.tighten.p200(ptr addrspace(200) %p, i64 7)
  ret void
}

; CHECK-LABEL: unused_scc:
; CHECK: # %bb.0:
; CHECK-NEXT: cjalr zero, 0(ra)
define void @unused_scc(ptr addrspace(200) %p, i64 %c) {
  %r = call ptr addrspace(200) @llvm.capstone.cap.scc.p200(ptr addrspace(200) %p, i64 %c)
  ret void
}

; CHECK-LABEL: unused_init:
; CHECK: # %bb.0:
; CHECK-NEXT: cjalr zero, 0(ra)
define void @unused_init(ptr addrspace(200) %p, i64 %c) {
  %r = call ptr addrspace(200) @llvm.capstone.cap.init.p200(ptr addrspace(200) %p, i64 %c)
  ret void
}

; (The negative names the instruction form, not the bare word: the function's
; own label comment `# @unused_seal` contains the word.)
; CHECK-LABEL: unused_seal:
; CHECK-NOT: seal a0
; CHECK: # %bb.0:
; CHECK-NEXT: cjalr zero, 0(ra)
define void @unused_seal(ptr addrspace(200) %p) {
  %r = call ptr addrspace(200) @llvm.capstone.cap.seal.p200(ptr addrspace(200) %p)
  ret void
}

;===--- Side-effecting operations: the instruction MUST survive. -------------===;

; CHECK-LABEL: unused_delin:
; CHECK: # %bb.0:
; CHECK-NEXT: delin a0
; CHECK-NEXT: cjalr zero, 0(ra)
define void @unused_delin(ptr addrspace(200) %p) {
  %r = call ptr addrspace(200) @llvm.capstone.cap.delin.p200(ptr addrspace(200) %p)
  ret void
}

; CHECK-LABEL: unused_mrev:
; CHECK: # %bb.0:
; CHECK-NEXT: mrev a0, a0
; CHECK-NEXT: cjalr zero, 0(ra)
define void @unused_mrev(ptr addrspace(200) %p) {
  %r = call ptr addrspace(200) @llvm.capstone.cap.mrev.p200(ptr addrspace(200) %p)
  ret void
}

; CHECK-LABEL: unused_drop:
; CHECK: # %bb.0:
; CHECK-NEXT: drop a0
; CHECK-NEXT: cjalr zero, 0(ra)
define void @unused_drop(ptr addrspace(200) %p) {
  %r = call ptr addrspace(200) @llvm.capstone.cap.drop.p200(ptr addrspace(200) %p)
  ret void
}

; CHECK-LABEL: unused_revoke:
; CHECK: # %bb.0:
; CHECK-NEXT: revoke a0
; CHECK-NEXT: cjalr zero, 0(ra)
define void @unused_revoke(ptr addrspace(200) %p) {
  %r = call ptr addrspace(200) @llvm.capstone.cap.revoke.p200(ptr addrspace(200) %p)
  ret void
}

; CHECK-LABEL: unused_ccsrrw:
; CHECK: # %bb.0:
; CHECK-NEXT: ccsrrw a0, ssp, a0
; CHECK-NEXT: cjalr zero, 0(ra)
define void @unused_ccsrrw(ptr addrspace(200) %p) {
  %r = call ptr addrspace(200) @llvm.capstone.cap.ccsrrw.p200(ptr addrspace(200) %p, i64 17)
  ret void
}

; The domain-crossing ops keep their frame (their result is defined into a
; callee-saved register); what matters here is that the instruction is emitted.
; CHECK-LABEL: unused_call:
; CHECK: call s0, a0
; CHECK: cjalr zero, 0(ra)
define void @unused_call(ptr addrspace(200) %p) {
  %r = call ptr addrspace(200) @llvm.capstone.cap.call.p200(ptr addrspace(200) %p)
  ret void
}

; CHECK-LABEL: unused_enter:
; CHECK: capenter s0, a0
; CHECK: cjalr zero, 0(ra)
define void @unused_enter(ptr addrspace(200) %p) {
  %r = call i64 @llvm.capstone.cap.enter.p200(ptr addrspace(200) %p)
  ret void
}
