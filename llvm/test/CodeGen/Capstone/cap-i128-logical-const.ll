; RUN: rm -rf %t && split-file %s %t
; RUN: llc -mtriple=capstone64 -mattr=+m -verify-machineinstrs < %t/sext.ll | FileCheck %s --check-prefix=SEXT
; RUN: not --crash llc -mtriple=capstone64 -mattr=+m < %t/zext.ll 2>&1 | FileCheck %s --check-prefix=ZEXT
;
; The i128 logical lowering narrows both operands to XLen and re-extends. It had
; no case for a CONSTANT operand, so it bailed on every pair containing one and
; the node reached isel unlowered. DAGCombiner builds exactly such a pair when it
; expands a negation into the NOT idiom (`xor x, -1`), which is how an ordinary
; pointer difference ended as "Cannot select: i128 = xor ..., Constant<-1>".
;
; Both directions are pinned here, because the fix is one sign flip away from a
; miscompile: the signed reading must lower, and the mixed one must still refuse.

;--- sext.ll
target triple = "capstone64-unknown-elf"

; -1 is the sign extension of its own low half, so it pairs with a sign-extended
; operand and the whole thing is one `not` at XLen.
; SEXT-LABEL: sext_not:
; SEXT:       not a0, a0
; SEXT-NOT:   xor
define i128 @sext_not(i64 %x) addrspace(200) {
  %e = sext i64 %x to i128
  %n = xor i128 %e, -1
  ret i128 %n
}

;--- zext.ll
target triple = "capstone64-unknown-elf"

; A zero-extended operand against -1 is the case that must NOT be re-extended:
; the true high half is all ones, which is not a function of the low-half result,
; so re-extending under either rule would MISCOMPILE. Refusing is correct.
; ZEXT: Cannot select
define i128 @zext_not(i64 %x) addrspace(200) {
  %e = zext i64 %x to i128
  %n = xor i128 %e, -1
  ret i128 %n
}
