; RUN: llc -mtriple=capstone64 -filetype=asm -verify-machineinstrs < %s | FileCheck %s

; tryShrinkShlLogicImm asked a constant for its int64 value before checking that it has one. On
; Capstone the constant can be i128: masking a capability's address with a 64-bit mask and widening
; the result back to capability width leaves 0xFFFFFFFFFFFFFFE0 as a POSITIVE 128-bit value, i.e. 65
; significant bits. getSExtValue() asserts on that with assertions enabled, and returns the low 64
; bits without them, which is a wrong immediate rather than a crash.
;
; Reduced from MicroPython's gc_init. The AND form of the same shape is blocked further on by a
; separate issue and lives in cap-i128-and-capability-mask.ll; OR and XOR reach selection and are
; what pins the guard here.

; The same shape reached through OR and XOR, which share the dispatch.

define ptr addrspace(200) @tag_or(ptr addrspace(200) %p) addrspace(200) {
; CHECK-LABEL: tag_or:
  %i = ptrtoint ptr addrspace(200) %p to i64
  %or = or i64 %i, -4096
  %conv = zext i64 %or to i128
  %r = inttoptr i128 %conv to ptr addrspace(200)
  ret ptr addrspace(200) %r
}

define ptr addrspace(200) @tag_xor(ptr addrspace(200) %p) addrspace(200) {
; CHECK-LABEL: tag_xor:
  %i = ptrtoint ptr addrspace(200) %p to i64
  %xor = xor i64 %i, -4096
  %conv = zext i64 %xor to i128
  %r = inttoptr i128 %conv to ptr addrspace(200)
  ret ptr addrspace(200) %r
}

; A constant that DOES fit an int64 must still take the transform, so the guard cannot be a blanket
; bail. This is tryShrinkShlLogicImm's own output shape: 0x50A0 does not fit a 12-bit immediate but
; 0x50A0 >> 4 does, so (x << 4) & 0x50A0 becomes (x & 0x50A) << 4. A blanket bail loses the ANDI and
; this test goes red.

define i64 @narrow_mask_still_folds(i64 %x) addrspace(200) {
; CHECK-LABEL: narrow_mask_still_folds:
; CHECK:       andi a0, a0, 1290
; CHECK-NEXT:  slli a0, a0, 4
  %s = shl i64 %x, 4
  %a = and i64 %s, 20640
  ret i64 %a
}
