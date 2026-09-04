; TIGHTEN's immediate is a uimm5: 32 does not encode, and the selector refuses
; it with a fatal error rather than truncating it into a different mask.  This
; pins the message and the fact that it is a hard stop (`not --crash`: today it
; is report_fatal_error with a stack dump; when the selector is changed to a
; proper diagnostic, this becomes plain `not llc` and the CHECK stays).
;
; The in-range positives (0, 7, 31) are pinned in intrinsics.ll; that is the
; control that keeps this file honest.
;
; MUTATION: change 32 to 31 -> llc succeeds and `not --crash` fails the RUN
; line (performed 2026-09-04).
;
; RUN: not --crash llc -mtriple=capstone64 -o /dev/null < %s 2>&1 | FileCheck %s

declare ptr addrspace(200) @llvm.capstone.cap.tighten.p200(ptr addrspace(200), i64)

; CHECK: Capstone TIGHTEN immediate must be in range 0-31!
define ptr addrspace(200) @tighten_32(ptr addrspace(200) %p) {
  %r = call ptr addrspace(200) @llvm.capstone.cap.tighten.p200(ptr addrspace(200) %p, i64 32)
  ret ptr addrspace(200) %r
}
