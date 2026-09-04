; CCSRRW's CSR number is a 12-bit field: 4096 does not encode, and the selector
; refuses it with a fatal error rather than wrapping it onto CSR 0.  Pins the
; message and that it is a hard stop (`not --crash` today; plain `not llc` once
; the selector emits a proper diagnostic -- the CHECK is unchanged either way).
;
; The in-range positives (0, 17 = ssp, 4095) are pinned in intrinsics.ll; that
; is the control that keeps this file honest.  Which numbers the silicon
; actually implements is a Sema question (Tier 4: {0,1,2,4,16..31}); the
; backend's job is the encoding range.
;
; MUTATION: change 4096 to 4095 -> llc succeeds and `not --crash` fails the
; RUN line (performed 2026-09-04).
;
; RUN: not --crash llc -mtriple=capstone64 -o /dev/null < %s 2>&1 | FileCheck %s

declare ptr addrspace(200) @llvm.capstone.cap.ccsrrw.p200(ptr addrspace(200), i64 immarg)

; CHECK: Capstone CCSRRW immediate must be in range 0-4095!
define ptr addrspace(200) @ccsrrw_4096(ptr addrspace(200) %p) {
  %r = call ptr addrspace(200) @llvm.capstone.cap.ccsrrw.p200(ptr addrspace(200) %p, i64 4096)
  ret ptr addrspace(200) %r
}
