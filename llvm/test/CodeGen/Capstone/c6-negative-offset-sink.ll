; C-6: CodeGenPrepare sinks an address computation into the blocks that use it
; and carries the constant offset as AddrMode.BaseOffs.  It once zero-extended
; that offset into the (then 128-bit) pointer index, so a NEGATIVE offset became
; a huge positive one and the load read the wrong address (ISSUES.md C-6, fixed
; with /*IsSigned=*/true at three sites).  The offset must fold into the load's
; immediate in BOTH sunk copies; nothing may be materialised or added to the
; pointer.  Measured 2026-09-04 on the branch tools.
;
; MUTATION: change -8 to 8 -> both loads become `lw a0, 8(a0)` and the checks
; fail (performed 2026-09-04).
;
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -verify-machineinstrs < %s | FileCheck %s --implicit-check-not=lui --implicit-check-not=cincoffset
; RUN: %llc_cap -O0 < %s -o /dev/null
; RUN: %llc_cap -O1 < %s -o /dev/null

; CHECK-LABEL: c6:
; CHECK: beqz a1, .LBB0_2
; CHECK: lw a0, -8(a0)
; CHECK-NEXT: cjalr zero, 0(ra)
; CHECK: .LBB0_2:
; CHECK-NEXT: lw a0, -8(a0)
; CHECK-NEXT: addiw a0, a0, 1
; CHECK-NEXT: cjalr zero, 0(ra)
define i32 @c6(ptr addrspace(200) %p, i1 %c) {
entry:
  %q = getelementptr i8, ptr addrspace(200) %p, i64 -8
  br i1 %c, label %a, label %b
a:
  %va = load i32, ptr addrspace(200) %q
  ret i32 %va
b:
  %vb = load i32, ptr addrspace(200) %q
  %r = add i32 %vb, 1
  ret i32 %r
}
