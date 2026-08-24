; RUN: llc -mtriple=capstone64 -verify-machineinstrs < %s | FileCheck %s

; --- Field reads (LCC) ---
declare i64 @llvm.capstone.cap.get.tag.p200(ptr addrspace(200))
declare i64 @llvm.capstone.cap.get.cursor.p200(ptr addrspace(200))
declare i64 @llvm.capstone.cap.get.base.p200(ptr addrspace(200))
declare i64 @llvm.capstone.cap.get.end.p200(ptr addrspace(200))
declare i64 @llvm.capstone.cap.get.perm.p200(ptr addrspace(200))

; --- Non-destructive manipulations (no side effects) ---
declare ptr addrspace(200) @llvm.capstone.cap.shrink.p200(ptr addrspace(200), i64, i64)
declare ptr addrspace(200) @llvm.capstone.cap.tighten.p200(ptr addrspace(200), i64)
declare ptr addrspace(200) @llvm.capstone.cap.scc.p200(ptr addrspace(200), i64)
declare ptr addrspace(200) @llvm.capstone.cap.init.p200(ptr addrspace(200), i64)
declare ptr addrspace(200) @llvm.capstone.cap.seal.p200(ptr addrspace(200))

; --- Revocation-tree mutations (side effects; see cap-mrev-delin-side-effects.ll) ---
declare ptr addrspace(200) @llvm.capstone.cap.delin.p200(ptr addrspace(200))
declare ptr addrspace(200) @llvm.capstone.cap.mrev.p200(ptr addrspace(200))

; --- Destructive manipulations (with side effects) ---
declare ptr addrspace(200) @llvm.capstone.cap.drop.p200(ptr addrspace(200))
declare ptr addrspace(200) @llvm.capstone.cap.revoke.p200(ptr addrspace(200))

; --- Capability CSR operations ---
declare ptr addrspace(200) @llvm.capstone.cap.ccsrrw.p200(ptr addrspace(200), i64 immarg)


; CHECK-LABEL: test_get_tag:
; CHECK: lcc a0, a0, 0
define i64 @test_get_tag(ptr addrspace(200) %p) {
entry:
  %0 = call i64 @llvm.capstone.cap.get.tag.p200(ptr addrspace(200) %p)
  ret i64 %0
}

; CHECK-LABEL: test_get_cursor:
; CHECK: lcc a0, a0, 2
define i64 @test_get_cursor(ptr addrspace(200) %p) {
entry:
  %0 = call i64 @llvm.capstone.cap.get.cursor.p200(ptr addrspace(200) %p)
  ret i64 %0
}

; CHECK-LABEL: test_get_base:
; CHECK: lcc a0, a0, 3
define i64 @test_get_base(ptr addrspace(200) %p) {
entry:
  %0 = call i64 @llvm.capstone.cap.get.base.p200(ptr addrspace(200) %p)
  ret i64 %0
}

; CHECK-LABEL: test_get_end:
; CHECK: lcc a0, a0, 4
define i64 @test_get_end(ptr addrspace(200) %p) {
entry:
  %0 = call i64 @llvm.capstone.cap.get.end.p200(ptr addrspace(200) %p)
  ret i64 %0
}

; CHECK-LABEL: test_get_perm:
; CHECK: lcc a0, a0, 5
define i64 @test_get_perm(ptr addrspace(200) %p) {
entry:
  %0 = call i64 @llvm.capstone.cap.get.perm.p200(ptr addrspace(200) %p)
  ret i64 %0
}

; CHECK-LABEL: test_shrink:
; CHECK: shrink a0, a1, a2
define ptr addrspace(200) @test_shrink(ptr addrspace(200) %p, i64 %base, i64 %end) {
entry:
  %0 = call ptr addrspace(200) @llvm.capstone.cap.shrink.p200(ptr addrspace(200) %p, i64 %base, i64 %end)
  ret ptr addrspace(200) %0
}

; CHECK-LABEL: test_tighten:
; CHECK: tighten a0, a0, 7
define ptr addrspace(200) @test_tighten(ptr addrspace(200) %p) {
entry:
  ; Note: TIGHTEN takes an immediate (uimm5), so we pass a constant
  %0 = call ptr addrspace(200) @llvm.capstone.cap.tighten.p200(ptr addrspace(200) %p, i64 7)
  ret ptr addrspace(200) %0
}

; CHECK-LABEL: test_scc:
; CHECK: scc a0, a0, a1
define ptr addrspace(200) @test_scc(ptr addrspace(200) %p, i64 %cursor) {
entry:
  %0 = call ptr addrspace(200) @llvm.capstone.cap.scc.p200(ptr addrspace(200) %p, i64 %cursor)
  ret ptr addrspace(200) %0
}

; CHECK-LABEL: test_delin:
; CHECK: delin a0
define ptr addrspace(200) @test_delin(ptr addrspace(200) %p) {
entry:
  %0 = call ptr addrspace(200) @llvm.capstone.cap.delin.p200(ptr addrspace(200) %p)
  ret ptr addrspace(200) %0
}

; CHECK-LABEL: test_init:
; CHECK: init a0, a0, a1
define ptr addrspace(200) @test_init(ptr addrspace(200) %p, i64 %val) {
entry:
  %0 = call ptr addrspace(200) @llvm.capstone.cap.init.p200(ptr addrspace(200) %p, i64 %val)
  ret ptr addrspace(200) %0
}

; CHECK-LABEL: test_mrev:
; CHECK: mrev a0, a0
define ptr addrspace(200) @test_mrev(ptr addrspace(200) %p) {
entry:
  %0 = call ptr addrspace(200) @llvm.capstone.cap.mrev.p200(ptr addrspace(200) %p)
  ret ptr addrspace(200) %0
}

; CHECK-LABEL: test_seal:
; CHECK: seal a0, a0
define ptr addrspace(200) @test_seal(ptr addrspace(200) %p) {
entry:
  %0 = call ptr addrspace(200) @llvm.capstone.cap.seal.p200(ptr addrspace(200) %p)
  ret ptr addrspace(200) %0
}

; CHECK-LABEL: test_drop:
; CHECK: drop a0
define ptr addrspace(200) @test_drop(ptr addrspace(200) %p) {
entry:
  %0 = call ptr addrspace(200) @llvm.capstone.cap.drop.p200(ptr addrspace(200) %p)
  ret ptr addrspace(200) %0
}

; CHECK-LABEL: test_revoke:
; CHECK: revoke a0
define ptr addrspace(200) @test_revoke(ptr addrspace(200) %p) {
entry:
  %0 = call ptr addrspace(200) @llvm.capstone.cap.revoke.p200(ptr addrspace(200) %p)
  ret ptr addrspace(200) %0
}

; CHECK-LABEL: test_ccsrrw:
; CHECK: ccsrrw a0, ssp, a0
define ptr addrspace(200) @test_ccsrrw(ptr addrspace(200) %p) {
entry:
  %0 = call ptr addrspace(200) @llvm.capstone.cap.ccsrrw.p200(ptr addrspace(200) %p, i64 17)
  ret ptr addrspace(200) %0
}

