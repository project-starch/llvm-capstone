; C-81: a copy of 16 bytes or more whose 16-byte alignment is not known calls
; memcpy/memmove; it is never split into scalar units.
;
; `align 1` says the alignment is unknown, not that the bytes hold no pointer.
; Perl's pack 'p' stores a pointer with `Copy(buf, cur, 16, char)`, a 16-byte
; memcpy between two `char *` that are 16-aligned at run time; it was inlined as
; sixteen `lbu`/`sb` pairs, which strip the tag, and `unpack 'p'` read back an
; untagged pointer. musl's memcpy checks the run-time alignment of both cursors
; and copies whole granules as capabilities, so the call keeps the tag.
;
; Kept inline: a copy shorter than a granule (it cannot hold a whole
; capability) and a copy from constant integer data (it holds no pointer).
;
; The default profile is matched line by line; the two deployed profiles
; (virtual: gp-free + image-gp, physical: gp-captable) check that the call is
; made pc-relatively with both pointer arguments left in their capability
; registers, and that no byte copy remains.
;
; MUTATION: with the C-81 block in CapstoneTargetLowering::
; findOptimalMemOpLowering disabled and nothing else changed, the six
; constant-size copies that check for a call are scalar copies (16 or 17 `sb`,
; or 2 `sd`) and fail; the run-time-size, 15-byte and constant-data controls
; pass either way (performed 2026-10-16).
;
; RUN: llc -mtriple=capstone64 -mattr=+m -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=capstone64 -mattr=+m -verify-machineinstrs \
; RUN:   -capstone-gp-free -capstone-image-gp < %s | FileCheck %s --check-prefix=PCREL
; RUN: llc -mtriple=capstone64 -mattr=+m -verify-machineinstrs \
; RUN:   -capstone-gp-captable < %s | FileCheck %s --check-prefix=PCREL
; RUN: %llc_cap -O0 < %s -o /dev/null
; RUN: %llc_cap -O1 < %s -o /dev/null

declare void @llvm.memcpy.p200.p200.i64(ptr addrspace(200), ptr addrspace(200), i64, i1)
declare void @llvm.memmove.p200.p200.i64(ptr addrspace(200), ptr addrspace(200), i64, i1)

@ints = private unnamed_addr addrspace(200) constant [11 x i32] [i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11], align 4

; The pack 'p' shape.
; CHECK-LABEL: copy16_align1:
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
; PCREL-LABEL: copy16_align1:
; PCREL-NOT:   {{lbu|sb|ld|sd|mv|movc}} {{.*}}a{{0|1}}
; PCREL:       auipc a2, %pcrel_hi(memcpy)
; PCREL-NEXT:  addi a3, a2, %pcrel_lo(.Lpcrel_hi{{[0-9]+}})
; PCREL-NEXT:  li a2, 16
; PCREL-NEXT:  jalr a3
; PCREL-NOT:   {{lbu|sb}}
; PCREL:       ret
define void @copy16_align1(ptr addrspace(200) %dst, ptr addrspace(200) %src) {
entry:
  call void @llvm.memcpy.p200.p200.i64(ptr addrspace(200) align 1 %dst,
                                       ptr addrspace(200) align 1 %src,
                                       i64 16, i1 false)
  ret void
}

; memmove takes the same decision and calls memmove.
; CHECK-LABEL: move16_align1:
; CHECK:       # %bb.0:
; CHECK-NEXT:  cincoffsetimm sp, sp, -16
; CHECK-NEXT:  .cfi_def_cfa_offset 16
; CHECK-NEXT:  stc ra, 0(sp)
; CHECK-NEXT:  .cfi_offset ra, -16
; CHECK-NEXT:  .Lpcrel_hi{{[0-9]+}}:
; CHECK-NEXT:  auipc a2, %pcrel_hi(memmove)
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
; PCREL-LABEL: move16_align1:
; PCREL-NOT:   {{lbu|sb|ld|sd|mv|movc}} {{.*}}a{{0|1}}
; PCREL:       auipc a2, %pcrel_hi(memmove)
; PCREL-NEXT:  addi a3, a2, %pcrel_lo(.Lpcrel_hi{{[0-9]+}})
; PCREL-NEXT:  li a2, 16
; PCREL-NEXT:  jalr a3
; PCREL-NOT:   {{lbu|sb}}
; PCREL:       ret
define void @move16_align1(ptr addrspace(200) %dst, ptr addrspace(200) %src) {
entry:
  call void @llvm.memmove.p200.p200.i64(ptr addrspace(200) align 1 %dst,
                                        ptr addrspace(200) align 1 %src,
                                        i64 16, i1 false)
  ret void
}

; Control: 15 bytes cannot hold a whole capability and stays a byte copy.
; CHECK-LABEL: copy15_align1:
; CHECK-NOT:   memcpy
; CHECK-COUNT-15: sb {{[a-z0-9]+}}, {{[0-9]+}}(a0)
; CHECK-NEXT:  cjalr zero, 0(ra)
; PCREL-LABEL: copy15_align1:
; PCREL-NOT:   memcpy
; PCREL-COUNT-15: sb {{[a-z0-9]+}}, {{[0-9]+}}(a0)
; PCREL-NEXT:  ret
define void @copy15_align1(ptr addrspace(200) %dst, ptr addrspace(200) %src) {
entry:
  call void @llvm.memcpy.p200.p200.i64(ptr addrspace(200) align 1 %dst,
                                       ptr addrspace(200) align 1 %src,
                                       i64 15, i1 false)
  ret void
}

; One granule and a byte.
; CHECK-LABEL: copy17_align1:
; CHECK-NOT:   {{lbu|sb}}
; CHECK:       auipc a2, %pcrel_hi(memcpy)
; CHECK:       li a2, 17
; CHECK-NEXT:  cjalr ra, 0(a3)
; PCREL-LABEL: copy17_align1:
; PCREL-NOT:   {{lbu|sb}}
; PCREL:       auipc a2, %pcrel_hi(memcpy)
; PCREL:       li a2, 17
; PCREL-NEXT:  jalr a3
define void @copy17_align1(ptr addrspace(200) %dst, ptr addrspace(200) %src) {
entry:
  call void @llvm.memcpy.p200.p200.i64(ptr addrspace(200) align 1 %dst,
                                       ptr addrspace(200) align 1 %src,
                                       i64 17, i1 false)
  ret void
}

; One end 16-aligned is not enough, on either side.
; CHECK-LABEL: copy16_dst16_src8:
; CHECK-NOT:   {{ld|sd}} {{.*}}(a{{0|1}})
; CHECK:       auipc a2, %pcrel_hi(memcpy)
; CHECK:       li a2, 16
; CHECK-NEXT:  cjalr ra, 0(a3)
; PCREL-LABEL: copy16_dst16_src8:
; PCREL:       auipc a2, %pcrel_hi(memcpy)
; PCREL:       jalr a3
define void @copy16_dst16_src8(ptr addrspace(200) %dst, ptr addrspace(200) %src) {
entry:
  call void @llvm.memcpy.p200.p200.i64(ptr addrspace(200) align 16 %dst,
                                       ptr addrspace(200) align 8 %src,
                                       i64 16, i1 false)
  ret void
}

; CHECK-LABEL: copy16_dst8_src16:
; CHECK-NOT:   {{ld|sd}} {{.*}}(a{{0|1}})
; CHECK:       auipc a2, %pcrel_hi(memcpy)
; CHECK:       li a2, 16
; CHECK-NEXT:  cjalr ra, 0(a3)
; PCREL-LABEL: copy16_dst8_src16:
; PCREL:       auipc a2, %pcrel_hi(memcpy)
; PCREL:       jalr a3
define void @copy16_dst8_src16(ptr addrspace(200) %dst, ptr addrspace(200) %src) {
entry:
  call void @llvm.memcpy.p200.p200.i64(ptr addrspace(200) align 8 %dst,
                                       ptr addrspace(200) align 16 %src,
                                       i64 16, i1 false)
  ret void
}

; A run-time size was always a call; unchanged.
; CHECK-LABEL: copy_dynamic_align1:
; CHECK:       auipc a3, %pcrel_hi(memcpy)
; CHECK:       cjalr ra, 0(a3)
; PCREL-LABEL: copy_dynamic_align1:
; PCREL:       auipc a3, %pcrel_hi(memcpy)
; PCREL:       jalr a3
define void @copy_dynamic_align1(ptr addrspace(200) %dst, ptr addrspace(200) %src, i64 %n) {
entry:
  call void @llvm.memcpy.p200.p200.i64(ptr addrspace(200) align 1 %dst,
                                       ptr addrspace(200) align 1 %src,
                                       i64 %n, i1 false)
  ret void
}

; Control: constant integer data holds no pointer; its values are stored inline.
; CHECK-LABEL: copy44_const_ints:
; CHECK-NOT:   memcpy
; CHECK-COUNT-11: sw {{[a-z0-9]+}}, {{[0-9]+}}(a0)
; CHECK-NEXT:  cjalr zero, 0(ra)
; PCREL-LABEL: copy44_const_ints:
; PCREL-NOT:   memcpy
; PCREL-COUNT-11: sw {{[a-z0-9]+}}, {{[0-9]+}}(a0)
; PCREL-NEXT:  ret
define void @copy44_const_ints(ptr addrspace(200) %dst) {
entry:
  call void @llvm.memcpy.p200.p200.i64(ptr addrspace(200) align 4 %dst,
                                       ptr addrspace(200) align 4 @ints,
                                       i64 44, i1 false)
  ret void
}
