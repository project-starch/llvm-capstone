; Every Capstone capability intrinsic, each pinned to EXACTLY the instruction it
; must select and nothing else: the block label / the instruction / the return.
; A bare one-line check per mnemonic (what this file used to be) proves the
; pattern fires somewhere; the -NEXT chain proves the body is that one
; instruction, so an extra move, a spurious cincoffset, or an lcc smuggled in
; beside it fails.  (Prose in this file must never contain the directive
; spelling with its colon -- FileCheck reads every line.)
;
; --implicit-check-not for `lcc ..., 2` is the C-19 guard: reading a
; capability's ADDRESS is `mv rd, rs` (addi rd, rs, 0), NOT `lcc rd, rs, 2`.
; Same value -- the plain regfile slot holds the cursor (RTL ex_stage.sv:463-479;
; QEMU cap.h union aliases scalar onto bounds.cursor) -- but the plain read is
; TOTAL, whereas lcc selector 2 TRAPS on an untagged operand and a NULL pointer
; is untagged.  `p != 0 || q != 0` folded to `(addr(p)|addr(q)) != 0`, both
; operands got `lcc`, and the first null killed the SQLite domain.
;
; MUTATION: in @test_get_cursor call get.base instead of get.cursor -> the
; Every occurrence of tighten/mrev/drop/revoke/ccsrrw is consumed by an explicit
; CHECK line, so the implicit-check-nots make "exactly the pinned instances" a
; checked property.  MUTATION: delete the pinned `mrev a0, a0` line (relaxing the
; -NEXT after it so only the negative can fail) -> the
; unconsumed mrev trips --implicit-check-not (performed 2026-09-04).
; -NEXT line expecting `mv a0, a0` fails, which shows each body is pinned to
; its own function and not merely present somewhere in the file.
;
; RUN: llc -mtriple=capstone64 -mattr=+m -verify-machineinstrs < %s | FileCheck %s --implicit-check-not='lcc {{.*}}, 2' --implicit-check-not='tighten a' --implicit-check-not='mrev a' --implicit-check-not='drop a' --implicit-check-not='revoke a' --implicit-check-not='ccsrrw a'
; RUN: %llc_cap -O0 < %s -o /dev/null
; RUN: %llc_cap -O1 < %s -o /dev/null

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

; get_tag is lcc selector 0.  Selector 0 is NOT total on either implementation
; (it traps on an untagged operand), so this builtin traps on the very value it
; exists to test; Tier 4 of the validation plan re-lowers it through selector 1.
; When that lands this CHECK changes with it.
; CHECK-LABEL: test_get_tag:
; CHECK: # %bb.0:
; CHECK-NEXT: lcc a0, a0, 0
; CHECK-NEXT: cjalr zero, 0(ra)
define i64 @test_get_tag(ptr addrspace(200) %p) {
entry:
  %0 = call i64 @llvm.capstone.cap.get.tag.p200(ptr addrspace(200) %p)
  ret i64 %0
}

; CHECK-LABEL: test_get_cursor:
; CHECK: # %bb.0:
; CHECK-NEXT: mv a0, a0
; CHECK-NEXT: cjalr zero, 0(ra)
define i64 @test_get_cursor(ptr addrspace(200) %p) {
entry:
  %0 = call i64 @llvm.capstone.cap.get.cursor.p200(ptr addrspace(200) %p)
  ret i64 %0
}

; CHECK-LABEL: test_get_base:
; CHECK: # %bb.0:
; CHECK-NEXT: lcc a0, a0, 3
; CHECK-NEXT: cjalr zero, 0(ra)
define i64 @test_get_base(ptr addrspace(200) %p) {
entry:
  %0 = call i64 @llvm.capstone.cap.get.base.p200(ptr addrspace(200) %p)
  ret i64 %0
}

; CHECK-LABEL: test_get_end:
; CHECK: # %bb.0:
; CHECK-NEXT: lcc a0, a0, 4
; CHECK-NEXT: cjalr zero, 0(ra)
define i64 @test_get_end(ptr addrspace(200) %p) {
entry:
  %0 = call i64 @llvm.capstone.cap.get.end.p200(ptr addrspace(200) %p)
  ret i64 %0
}

; CHECK-LABEL: test_get_perm:
; CHECK: # %bb.0:
; CHECK-NEXT: lcc a0, a0, 5
; CHECK-NEXT: cjalr zero, 0(ra)
define i64 @test_get_perm(ptr addrspace(200) %p) {
entry:
  %0 = call i64 @llvm.capstone.cap.get.perm.p200(ptr addrspace(200) %p)
  ret i64 %0
}

; CHECK-LABEL: test_shrink:
; CHECK: # %bb.0:
; CHECK-NEXT: shrink a0, a1, a2
; CHECK-NEXT: cjalr zero, 0(ra)
define ptr addrspace(200) @test_shrink(ptr addrspace(200) %p, i64 %base, i64 %end) {
entry:
  %0 = call ptr addrspace(200) @llvm.capstone.cap.shrink.p200(ptr addrspace(200) %p, i64 %base, i64 %end)
  ret ptr addrspace(200) %0
}

; TIGHTEN takes a uimm5.  Three points of the range: 0, the largest value the
; silicon accepts (7 -- above it the RTL raises ILLEGAL_OPERAND_VALUE, which is
; why the Sema range in Tier 4 is 0..7 and not 0..31), and the encoding's own
; maximum, 31, which the backend must still encode faithfully.
; CHECK-LABEL: test_tighten:
; CHECK: # %bb.0:
; CHECK-NEXT: tighten a0, a0, 7
; CHECK-NEXT: cjalr zero, 0(ra)
define ptr addrspace(200) @test_tighten(ptr addrspace(200) %p) {
entry:
  %0 = call ptr addrspace(200) @llvm.capstone.cap.tighten.p200(ptr addrspace(200) %p, i64 7)
  ret ptr addrspace(200) %0
}

; CHECK-LABEL: test_tighten_0:
; CHECK: # %bb.0:
; CHECK-NEXT: tighten a0, a0, 0
; CHECK-NEXT: cjalr zero, 0(ra)
define ptr addrspace(200) @test_tighten_0(ptr addrspace(200) %p) {
entry:
  %0 = call ptr addrspace(200) @llvm.capstone.cap.tighten.p200(ptr addrspace(200) %p, i64 0)
  ret ptr addrspace(200) %0
}

; CHECK-LABEL: test_tighten_31:
; CHECK: # %bb.0:
; CHECK-NEXT: tighten a0, a0, 31
; CHECK-NEXT: cjalr zero, 0(ra)
define ptr addrspace(200) @test_tighten_31(ptr addrspace(200) %p) {
entry:
  %0 = call ptr addrspace(200) @llvm.capstone.cap.tighten.p200(ptr addrspace(200) %p, i64 31)
  ret ptr addrspace(200) %0
}

; CHECK-LABEL: test_scc:
; CHECK: # %bb.0:
; CHECK-NEXT: scc a0, a0, a1
; CHECK-NEXT: cjalr zero, 0(ra)
define ptr addrspace(200) @test_scc(ptr addrspace(200) %p, i64 %cursor) {
entry:
  %0 = call ptr addrspace(200) @llvm.capstone.cap.scc.p200(ptr addrspace(200) %p, i64 %cursor)
  ret ptr addrspace(200) %0
}

; CHECK-LABEL: test_delin:
; CHECK: # %bb.0:
; CHECK-NEXT: delin a0
; CHECK-NEXT: cjalr zero, 0(ra)
define ptr addrspace(200) @test_delin(ptr addrspace(200) %p) {
entry:
  %0 = call ptr addrspace(200) @llvm.capstone.cap.delin.p200(ptr addrspace(200) %p)
  ret ptr addrspace(200) %0
}

; CHECK-LABEL: test_init:
; CHECK: # %bb.0:
; CHECK-NEXT: init a0, a0, a1
; CHECK-NEXT: cjalr zero, 0(ra)
define ptr addrspace(200) @test_init(ptr addrspace(200) %p, i64 %val) {
entry:
  %0 = call ptr addrspace(200) @llvm.capstone.cap.init.p200(ptr addrspace(200) %p, i64 %val)
  ret ptr addrspace(200) %0
}

; CHECK-LABEL: test_mrev:
; CHECK: # %bb.0:
; CHECK-NEXT: mrev a0, a0
; CHECK-NEXT: cjalr zero, 0(ra)
define ptr addrspace(200) @test_mrev(ptr addrspace(200) %p) {
entry:
  %0 = call ptr addrspace(200) @llvm.capstone.cap.mrev.p200(ptr addrspace(200) %p)
  ret ptr addrspace(200) %0
}

; CHECK-LABEL: test_seal:
; CHECK: # %bb.0:
; CHECK-NEXT: seal a0, a0
; CHECK-NEXT: cjalr zero, 0(ra)
define ptr addrspace(200) @test_seal(ptr addrspace(200) %p) {
entry:
  %0 = call ptr addrspace(200) @llvm.capstone.cap.seal.p200(ptr addrspace(200) %p)
  ret ptr addrspace(200) %0
}

; CHECK-LABEL: test_drop:
; CHECK: # %bb.0:
; CHECK-NEXT: drop a0
; CHECK-NEXT: cjalr zero, 0(ra)
define ptr addrspace(200) @test_drop(ptr addrspace(200) %p) {
entry:
  %0 = call ptr addrspace(200) @llvm.capstone.cap.drop.p200(ptr addrspace(200) %p)
  ret ptr addrspace(200) %0
}

; CHECK-LABEL: test_revoke:
; CHECK: # %bb.0:
; CHECK-NEXT: revoke a0
; CHECK-NEXT: cjalr zero, 0(ra)
define ptr addrspace(200) @test_revoke(ptr addrspace(200) %p) {
entry:
  %0 = call ptr addrspace(200) @llvm.capstone.cap.revoke.p200(ptr addrspace(200) %p)
  ret ptr addrspace(200) %0
}

; CCSRRW with a NAMED CSR (17 = 0x011 = ssp) and with unnamed ones at both ends
; of the encoding (0 and 0xfff): the printer must name what it can and emit the
; number otherwise.  Which ids the silicon actually accepts is a Sema question
; (Tier 4: {0,1,2,4,16..31}); the backend encodes whatever it is given.
; CHECK-LABEL: test_ccsrrw:
; CHECK: # %bb.0:
; CHECK-NEXT: ccsrrw a0, ssp, a0
; CHECK-NEXT: cjalr zero, 0(ra)
define ptr addrspace(200) @test_ccsrrw(ptr addrspace(200) %p) {
entry:
  %0 = call ptr addrspace(200) @llvm.capstone.cap.ccsrrw.p200(ptr addrspace(200) %p, i64 17)
  ret ptr addrspace(200) %0
}

; CHECK-LABEL: test_ccsrrw_0:
; CHECK: # %bb.0:
; CHECK-NEXT: ccsrrw a0, 0, a0
; CHECK-NEXT: cjalr zero, 0(ra)
define ptr addrspace(200) @test_ccsrrw_0(ptr addrspace(200) %p) {
entry:
  %0 = call ptr addrspace(200) @llvm.capstone.cap.ccsrrw.p200(ptr addrspace(200) %p, i64 0)
  ret ptr addrspace(200) %0
}

; CHECK-LABEL: test_ccsrrw_max:
; CHECK: # %bb.0:
; CHECK-NEXT: ccsrrw a0, 4095, a0
; CHECK-NEXT: cjalr zero, 0(ra)
define ptr addrspace(200) @test_ccsrrw_max(ptr addrspace(200) %p) {
entry:
  %0 = call ptr addrspace(200) @llvm.capstone.cap.ccsrrw.p200(ptr addrspace(200) %p, i64 4095)
  ret ptr addrspace(200) %0
}
