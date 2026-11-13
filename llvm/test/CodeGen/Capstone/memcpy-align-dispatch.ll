; -capstone-memcpy-align-dispatch (C8): a 16-byte llvm.memcpy whose operands
; are known 8-aligned but not both 16-aligned is inlined with a run-time
; alignment test instead of the memcpy call C-81 makes:
;   both cursors 16-aligned -> one ldc/stc pair (the tag is kept);
;   otherwise               -> two ld, then two sd (no whole granule moves).
; That is the runtime memcpy's rule for this size. Left alone: an operand known
; only 1-aligned, 32 bytes (a whole middle granule can exist when both cursors
; are 8 mod 16, so a word copy would strip a tag memcpy keeps), a volatile
; copy, and a copy both of whose ends are 16-aligned (C-78 already inlines it).
; Without the flag every candidate stays a call.
;
; RUN: llc -mtriple=capstone64 -mattr=+m -verify-machineinstrs \
; RUN:   -capstone-gp-free -capstone-image-gp -capstone-memcpy-align-dispatch < %s \
; RUN:   | FileCheck %s --check-prefix=ON
; RUN: llc -mtriple=capstone64 -mattr=+m -verify-machineinstrs \
; RUN:   -capstone-gp-free -capstone-image-gp < %s | FileCheck %s --check-prefix=OFF
; RUN: llc -mtriple=capstone64 -mattr=+m -verify-machineinstrs -O0 \
; RUN:   -capstone-gp-free -capstone-image-gp -capstone-memcpy-align-dispatch < %s \
; RUN:   | FileCheck %s --check-prefix=O0

declare void @llvm.memcpy.p200.p200.i64(ptr addrspace(200), ptr addrspace(200), i64, i1)

; gcc's bitmap_copy shape: both ends 8-aligned.
define void @copy16_align8(ptr addrspace(200) align 8 %d, ptr addrspace(200) align 8 %s) addrspace(200) nounwind {
; ON-LABEL: copy16_align8:
; ON-NOT:     memcpy
; ON:         mv [[D:a[0-9]]], a0
; ON-NEXT:    mv [[S:a[0-9]]], a1
; ON-NEXT:    or [[D]], [[D]], [[S]]
; ON-NEXT:    andi [[D]], [[D]], 15
; ON-NEXT:    beqz [[D]], .LBB0_2
; ON-NEXT:  # %bb.1:
; ON-NEXT:    ld [[L0:a[0-9]]], 0(a1)
; ON-NEXT:    ld [[L1:a[0-9]]], 8(a1)
; ON-NEXT:    sd [[L0]], 0(a0)
; ON-NEXT:    sd [[L1]], 8(a0)
; ON-NEXT:    ret
; ON-NEXT:  .LBB0_2:
; ON-NEXT:    ldc [[C:a[0-9]]], 0(a1)
; ON-NEXT:    stc [[C]], 0(a0)
; ON-NEXT:    ret
; OFF-LABEL: copy16_align8:
; OFF:         %pcrel_hi(memcpy)
; O0-LABEL: copy16_align8:
; O0:          %pcrel_hi(memcpy)
  call void @llvm.memcpy.p200.p200.i64(ptr addrspace(200) align 8 %d, ptr addrspace(200) align 8 %s, i64 16, i1 false)
  ret void
}

; One end known 16-aligned is not enough.
define void @copy16_dst16_src8(ptr addrspace(200) align 16 %d, ptr addrspace(200) align 8 %s) addrspace(200) nounwind {
; ON-LABEL: copy16_dst16_src8:
; ON-NOT:     memcpy
; ON:         andi {{[a-z0-9]+}}, {{[a-z0-9]+}}, 15
; ON:         ld
; ON:         sd
; ON:         ldc
; ON:         stc
; ON:         .size copy16_dst16_src8
; OFF-LABEL: copy16_dst16_src8:
; OFF:         %pcrel_hi(memcpy)
  call void @llvm.memcpy.p200.p200.i64(ptr addrspace(200) align 16 %d, ptr addrspace(200) align 8 %s, i64 16, i1 false)
  ret void
}

; Left alone: alignment 1, so a word copy could be misaligned.
define void @copy16_align1(ptr addrspace(200) %d, ptr addrspace(200) %s) addrspace(200) nounwind {
; ON-LABEL: copy16_align1:
; ON-NOT:     andi
; ON:         %pcrel_hi(memcpy)
; ON:         .size copy16_align1
  call void @llvm.memcpy.p200.p200.i64(ptr addrspace(200) %d, ptr addrspace(200) %s, i64 16, i1 false)
  ret void
}

; Left alone: 32 bytes.
define void @copy32_align8(ptr addrspace(200) align 8 %d, ptr addrspace(200) align 8 %s) addrspace(200) nounwind {
; ON-LABEL: copy32_align8:
; ON-NOT:     andi
; ON:         %pcrel_hi(memcpy)
; ON:         .size copy32_align8
  call void @llvm.memcpy.p200.p200.i64(ptr addrspace(200) align 8 %d, ptr addrspace(200) align 8 %s, i64 32, i1 false)
  ret void
}

; Left alone: volatile.
define void @copy16_volatile(ptr addrspace(200) align 8 %d, ptr addrspace(200) align 8 %s) addrspace(200) nounwind {
; ON-LABEL: copy16_volatile:
; ON-NOT:     andi
; ON:         .size copy16_volatile
  call void @llvm.memcpy.p200.p200.i64(ptr addrspace(200) align 8 %d, ptr addrspace(200) align 8 %s, i64 16, i1 true)
  ret void
}

; Left alone: both ends 16-aligned, C-78's inline capability copy.
define void @copy16_align16(ptr addrspace(200) align 16 %d, ptr addrspace(200) align 16 %s) addrspace(200) nounwind {
; ON-LABEL: copy16_align16:
; ON-NOT:     andi
; ON:         ldc [[C:[a-z0-9]+]], 0(a1)
; ON-NEXT:    stc [[C]], 0(a0)
; ON-NEXT:    ret
  call void @llvm.memcpy.p200.p200.i64(ptr addrspace(200) align 16 %d, ptr addrspace(200) align 16 %s, i64 16, i1 false)
  ret void
}
