; Verify that a 16-byte struct memcpy is lowered with matched-width memory ops.
;
; A sub-capability-aligned (8-byte) 16-byte copy must be lowered as paired 8-byte
; ld/sd, NOT as an 8-byte `ld` feeding a 16-byte `stc`.  The latter dropped the
; upper 8 bytes of every 16-byte unit and corrupted by-value 16-byte {i64,i64}
; struct copies such as `range = MakeRange(...)` (assignment of a struct return
; into a pre-declared local), which silently broke e.g. the BEEBS wikisort sort.
;
; A 16-byte-aligned copy may use the capability-preserving ldc/stc pair, but it
; must still be a matched 16-byte load+store.
;
; Every "no stc/ldc" negative below is paired with a COUNT of the stores that
; must replace them (measured 2026-09-04: 2 sd, 4 sw, 11 sw).  Without the
; counts, a lowering that emitted nothing at all would have satisfied the
; negatives -- the C-26 class of vacuous check.
;
; MUTATION: change `align 4` to `align 16` on both operands of @copy_i32x4_align4
; -> the copy becomes one ldc/stc pair, the negative on stc fires and the count
; of four sw fails (performed 2026-09-04 on a scratch copy).
;
; RUN: llc -mtriple=capstone64 -mattr=+m -verify-machineinstrs < %s | FileCheck %s
; RUN: %llc_cap -O0 < %s -o /dev/null
; RUN: %llc_cap -O1 < %s -o /dev/null

declare void @llvm.memcpy.p200.p200.i64(ptr addrspace(200), ptr addrspace(200), i64, i1)

; A 16-byte copy aligned to only 8 bytes cannot hold an in-place tagged
; capability, so it is copied as two matched 8-byte integer accesses.
; CHECK-LABEL: copy_i64x2_align8:
; CHECK-NOT:   stc
; CHECK-NOT:   ldc
; CHECK-COUNT-2: sd {{[a-z0-9]+}}, {{[0-9]+}}(
; CHECK-NOT:   stc
; CHECK:       cjalr zero, 0(ra)
define void @copy_i64x2_align8(ptr addrspace(200) %dst, ptr addrspace(200) %src) {
entry:
  call void @llvm.memcpy.p200.p200.i64(ptr addrspace(200) align 8 %dst,
                                       ptr addrspace(200) align 8 %src,
                                       i64 16, i1 false)
  ret void
}

; A 16-byte, 16-byte-aligned copy may use the capability load/store pair, which
; is correctly width-matched (ldc loads all 16 bytes; stc stores all 16 bytes).
; This is the control for the negatives elsewhere in the file: ldc/stc IS what
; the backend emits when the alignment permits it.
; CHECK-LABEL: copy_align16:
; CHECK: # %bb.0:
; CHECK-NEXT:  ldc [[R:[a-z0-9]+]], 0(a1)
; CHECK-NEXT:  stc [[R]], 0(a0)
; CHECK-NEXT:  cjalr zero, 0(ra)
define void @copy_align16(ptr addrspace(200) %dst, ptr addrspace(200) %src) {
entry:
  call void @llvm.memcpy.p200.p200.i64(ptr addrspace(200) align 16 %dst,
                                       ptr addrspace(200) align 16 %src,
                                       i64 16, i1 false)
  ret void
}

; A 16-byte copy aligned to only 4 bytes: the destination might still be
; 16-aligned, so the generic lowering used to pick an i128 (stc) unit whose
; misaligned source load only supplied 8 bytes.  It must decompose to matched
; 4-byte accesses (four sw) instead -- no stc/ldc.
; CHECK-LABEL: copy_i32x4_align4:
; CHECK-NOT:   stc
; CHECK-NOT:   ldc
; CHECK-COUNT-4: sw {{[a-z0-9]+}}, {{[0-9]+}}(
; CHECK-NOT:   stc
; CHECK:       cjalr zero, 0(ra)
define void @copy_i32x4_align4(ptr addrspace(200) %dst, ptr addrspace(200) %src) {
entry:
  call void @llvm.memcpy.p200.p200.i64(ptr addrspace(200) align 4 %dst,
                                       ptr addrspace(200) align 4 %src,
                                       i64 16, i1 false)
  ret void
}

; A 44-byte copy aligned to 4 (`int e[11] = {...}` local-array init): size is a
; multiple of neither 16 nor 8, so it matched none of the special-cased branches
; and fell through to the generic lowering, which picked a leading i128 (stc)
; unit and dropped the upper 8 bytes of it.  It must decompose to matched 4-byte
; accesses (eleven sw) -- no stc/ldc.
; CHECK-LABEL: copy_44_align4:
; CHECK-NOT:   stc
; CHECK-NOT:   ldc
; CHECK-COUNT-11: sw {{[a-z0-9]+}}, {{[0-9]+}}(
; CHECK-NOT:   stc
; CHECK:       cjalr zero, 0(ra)
define void @copy_44_align4(ptr addrspace(200) %dst, ptr addrspace(200) %src) {
entry:
  call void @llvm.memcpy.p200.p200.i64(ptr addrspace(200) align 4 %dst,
                                       ptr addrspace(200) align 4 %src,
                                       i64 44, i1 false)
  ret void
}
