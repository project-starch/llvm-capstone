; Verify that a 16-byte struct memcpy is lowered with matched-width memory ops.
;
; A sub-capability-aligned 16-byte copy must never be lowered as an 8-byte `ld`
; feeding a 16-byte `stc`.  That dropped the upper 8 bytes of every 16-byte unit
; and corrupted by-value 16-byte {i64,i64} struct copies such as
; `range = MakeRange(...)` (assignment of a struct return into a pre-declared
; local), which silently broke e.g. the BEEBS wikisort sort.
;
; A 16-byte-aligned copy may use the capability-preserving ldc/stc pair, but it
; must still be a matched 16-byte load+store.
;
; C-81: a copy of 16 bytes or more whose 16-byte alignment is unknown calls
; `memcpy` instead of being split into scalar units, because its bytes may hold
; a capability and scalar loads and stores strip tags (Perl's pack 'p' copied a
; pointer through `char *` this way).  Those bodies are matched line by line, so
; a scalar copy, an `ld`/`stc` pair or an empty lowering all fail.  A copy from
; constant integer data holds no pointer and still stores its values inline.
;
; MUTATION: with the C-81 block in CapstoneTargetLowering::
; findOptimalMemOpLowering disabled and nothing else changed, the three
; under-aligned copies are 2 sd, 4 sw and 11 sw and fail their memcpy lines; the
; aligned control and the constant-data control pass either way (performed
; 2026-10-16).
;
; RUN: llc -mtriple=capstone64 -mattr=+m -verify-machineinstrs < %s | FileCheck %s
; RUN: %llc_cap -O0 < %s -o /dev/null
; RUN: %llc_cap -O1 < %s -o /dev/null

declare void @llvm.memcpy.p200.p200.i64(ptr addrspace(200), ptr addrspace(200), i64, i1)

; A 16-byte copy aligned to only 8 bytes: both ends may still be 16-aligned at
; run time and hold a capability, so it calls memcpy, whose run-time alignment
; check copies the granule as a capability.
; CHECK-LABEL: copy_i64x2_align8:
; CHECK:       # %bb.0:
; CHECK-NEXT:  cincoffsetimm sp, sp, -16
; CHECK-NEXT:  .cfi_def_cfa_offset 16
; CHECK-NEXT:  stc ra, 0(sp)
; CHECK-NEXT:  .cfi_offset ra, -16
; CHECK-NEXT:  .Lpcrel_hi{{[0-9]+}}:
; CHECK-NEXT:  auipc a2, %pcrel_hi(memcpy)
; CHECK-NEXT:  addi a2, a2, %pcrel_lo(.Lpcrel_hi{{[0-9]+}})
; CHECK-NEXT:  cincoffset a3, gp, a2
; CHECK-NEXT:  delin a3
; CHECK-NEXT:  li a2, 16
; CHECK-NEXT:  cjalr ra, 0(a3)
; CHECK-NEXT:  ldc ra, 0(sp)
; CHECK-NEXT:  .cfi_restore ra
; CHECK-NEXT:  cincoffsetimm sp, sp, 16
; CHECK-NEXT:  .cfi_def_cfa_offset 0
; CHECK-NEXT:  cjalr zero, 0(ra)
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

; A 16-byte copy aligned to only 4 bytes: the generic lowering used to pick an
; i128 (stc) unit whose misaligned source load only supplied 8 bytes, then a
; scalar split stripped tags.  It calls memcpy.
; CHECK-LABEL: copy_i32x4_align4:
; CHECK:       # %bb.0:
; CHECK-NEXT:  cincoffsetimm sp, sp, -16
; CHECK-NEXT:  .cfi_def_cfa_offset 16
; CHECK-NEXT:  stc ra, 0(sp)
; CHECK-NEXT:  .cfi_offset ra, -16
; CHECK-NEXT:  .Lpcrel_hi{{[0-9]+}}:
; CHECK-NEXT:  auipc a2, %pcrel_hi(memcpy)
; CHECK-NEXT:  addi a2, a2, %pcrel_lo(.Lpcrel_hi{{[0-9]+}})
; CHECK-NEXT:  cincoffset a3, gp, a2
; CHECK-NEXT:  delin a3
; CHECK-NEXT:  li a2, 16
; CHECK-NEXT:  cjalr ra, 0(a3)
; CHECK-NEXT:  ldc ra, 0(sp)
; CHECK-NEXT:  .cfi_restore ra
; CHECK-NEXT:  cincoffsetimm sp, sp, 16
; CHECK-NEXT:  .cfi_def_cfa_offset 0
; CHECK-NEXT:  cjalr zero, 0(ra)
define void @copy_i32x4_align4(ptr addrspace(200) %dst, ptr addrspace(200) %src) {
entry:
  call void @llvm.memcpy.p200.p200.i64(ptr addrspace(200) align 4 %dst,
                                       ptr addrspace(200) align 4 %src,
                                       i64 16, i1 false)
  ret void
}

; A 44-byte copy aligned to 4: a multiple of neither 16 nor 8, it once fell
; through to the generic lowering, which picked a leading i128 (stc) unit and
; dropped the upper 8 bytes of it.  It calls memcpy.
; CHECK-LABEL: copy_44_align4:
; CHECK:       # %bb.0:
; CHECK-NEXT:  cincoffsetimm sp, sp, -16
; CHECK-NEXT:  .cfi_def_cfa_offset 16
; CHECK-NEXT:  stc ra, 0(sp)
; CHECK-NEXT:  .cfi_offset ra, -16
; CHECK-NEXT:  .Lpcrel_hi{{[0-9]+}}:
; CHECK-NEXT:  auipc a2, %pcrel_hi(memcpy)
; CHECK-NEXT:  addi a2, a2, %pcrel_lo(.Lpcrel_hi{{[0-9]+}})
; CHECK-NEXT:  cincoffset a3, gp, a2
; CHECK-NEXT:  delin a3
; CHECK-NEXT:  li a2, 44
; CHECK-NEXT:  cjalr ra, 0(a3)
; CHECK-NEXT:  ldc ra, 0(sp)
; CHECK-NEXT:  .cfi_restore ra
; CHECK-NEXT:  cincoffsetimm sp, sp, 16
; CHECK-NEXT:  .cfi_def_cfa_offset 0
; CHECK-NEXT:  cjalr zero, 0(ra)
define void @copy_44_align4(ptr addrspace(200) %dst, ptr addrspace(200) %src) {
entry:
  call void @llvm.memcpy.p200.p200.i64(ptr addrspace(200) align 4 %dst,
                                       ptr addrspace(200) align 4 %src,
                                       i64 44, i1 false)
  ret void
}

; The same 44 bytes from a constant int array (`int e[11] = {...}` local-array
; initialization): constant data holds no pointer, so the values are stored
; inline with matched 4-byte stores, eleven `sw` and no call.
@ints = private unnamed_addr addrspace(200) constant [11 x i32] [i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11], align 4
; CHECK-LABEL: copy_44_const_ints:
; CHECK-NOT:   memcpy
; CHECK-NOT:   stc
; CHECK-COUNT-11: sw {{[a-z0-9]+}}, {{[0-9]+}}(a0)
; CHECK-NOT:   memcpy
; CHECK:       cjalr zero, 0(ra)
define void @copy_44_const_ints(ptr addrspace(200) %dst) {
entry:
  call void @llvm.memcpy.p200.p200.i64(ptr addrspace(200) align 4 %dst,
                                       ptr addrspace(200) align 4 @ints,
                                       i64 44, i1 false)
  ret void
}
