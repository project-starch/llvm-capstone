; C-78: a capability-aligned copy whose size is not a multiple of 16 keeps the
; tags of the capabilities in its whole granules.
;
; A 16-aligned 24-byte copy holds one capability and 8 bytes of plain data,
; which is what `memcpy(d, s, offsetof(T, tail))` over a struct that starts with
; pointers is. It used to be copied entirely with 8-byte `ld`/`sd`, which strips
; the capability's tag. The whole granules must be copied with `ldc`/`stc` and
; only the tail with scalar units.
;
; Every body is matched line by line, so a missing access, an extra scalar copy
; of a granule, or an empty lowering all fail.
;
; MUTATION: without the C-78 block in CapstoneTargetLowering::
; findOptimalMemOpLowering, @copy24_align16 is `ld/sd` at 16, 8 and 0 and the
; four positive functions fail their `ldc` lines; the short-copy control passes
; either way (performed 2026-10-13 with the unfixed llc of the same source).
;
; RUN: llc -mtriple=capstone64 -mattr=+m -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=capstone64 -mattr=+m -verify-machineinstrs \
; RUN:   -capstone-memcpy-high-half-fixup < %s | FileCheck %s --check-prefix=FIXUP
; RUN: %llc_cap -O0 < %s -o /dev/null
; RUN: %llc_cap -O1 < %s -o /dev/null

declare void @llvm.memcpy.p200.p200.i64(ptr addrspace(200), ptr addrspace(200), i64, i1)
declare void @llvm.memmove.p200.p200.i64(ptr addrspace(200), ptr addrspace(200), i64, i1)

; One granule as a capability, the 8-byte tail as one scalar pair.
; CHECK-LABEL: copy24_align16:
; CHECK:       # %bb.0:
; CHECK-NEXT:  ld [[T:[a-z0-9]+]], 16(a1)
; CHECK-NEXT:  sd [[T]], 16(a0)
; CHECK-NEXT:  ldc [[C:[a-z0-9]+]], 0(a1)
; CHECK-NEXT:  stc [[C]], 0(a0)
; CHECK-NEXT:  cjalr zero, 0(ra)
; With the S-06 fixup on, these sizes keep their previous lowering: that
; experiment's decline and hook agree only on the 16-multiple shape.
; FIXUP-LABEL: copy24_align16:
; FIXUP:       # %bb.0:
; FIXUP-NEXT:  ld [[T:[a-z0-9]+]], 16(a1)
; FIXUP-NEXT:  sd [[T]], 16(a0)
; FIXUP-NEXT:  ld [[T:[a-z0-9]+]], 8(a1)
; FIXUP-NEXT:  sd [[T]], 8(a0)
; FIXUP-NEXT:  ld [[T:[a-z0-9]+]], 0(a1)
; FIXUP-NEXT:  sd [[T]], 0(a0)
; FIXUP-NEXT:  cjalr zero, 0(ra)
define void @copy24_align16(ptr addrspace(200) %dst, ptr addrspace(200) %src) {
entry:
  call void @llvm.memcpy.p200.p200.i64(ptr addrspace(200) align 16 %dst,
                                       ptr addrspace(200) align 16 %src,
                                       i64 24, i1 false)
  ret void
}

; Two granules and an 8-byte tail.
; CHECK-LABEL: copy40_align16:
; CHECK:       # %bb.0:
; CHECK-NEXT:  ld [[T:[a-z0-9]+]], 32(a1)
; CHECK-NEXT:  sd [[T]], 32(a0)
; CHECK-NEXT:  ldc [[C:[a-z0-9]+]], 16(a1)
; CHECK-NEXT:  stc [[C]], 16(a0)
; CHECK-NEXT:  ldc [[C:[a-z0-9]+]], 0(a1)
; CHECK-NEXT:  stc [[C]], 0(a0)
; CHECK-NEXT:  cjalr zero, 0(ra)
define void @copy40_align16(ptr addrspace(200) %dst, ptr addrspace(200) %src) {
entry:
  call void @llvm.memcpy.p200.p200.i64(ptr addrspace(200) align 16 %dst,
                                       ptr addrspace(200) align 16 %src,
                                       i64 40, i1 false)
  ret void
}

; One granule and a 7-byte tail of 4, 2 and 1 bytes.
; CHECK-LABEL: copy23_align16:
; CHECK:       # %bb.0:
; CHECK-NEXT:  lbu [[T:[a-z0-9]+]], 22(a1)
; CHECK-NEXT:  sb [[T]], 22(a0)
; CHECK-NEXT:  lh [[T:[a-z0-9]+]], 20(a1)
; CHECK-NEXT:  sh [[T]], 20(a0)
; CHECK-NEXT:  lw [[T:[a-z0-9]+]], 16(a1)
; CHECK-NEXT:  sw [[T]], 16(a0)
; CHECK-NEXT:  ldc [[C:[a-z0-9]+]], 0(a1)
; CHECK-NEXT:  stc [[C]], 0(a0)
; CHECK-NEXT:  cjalr zero, 0(ra)
define void @copy23_align16(ptr addrspace(200) %dst, ptr addrspace(200) %src) {
entry:
  call void @llvm.memcpy.p200.p200.i64(ptr addrspace(200) align 16 %dst,
                                       ptr addrspace(200) align 16 %src,
                                       i64 23, i1 false)
  ret void
}

; memmove is expanded through the same hook and takes the same chunks, all
; loads before the stores.
; CHECK-LABEL: move24_align16:
; CHECK:       # %bb.0:
; CHECK-NEXT:  ld [[T:[a-z0-9]+]], 16(a1)
; CHECK-NEXT:  ldc [[C:[a-z0-9]+]], 0(a1)
; CHECK-NEXT:  sd [[T]], 16(a0)
; CHECK-NEXT:  stc [[C]], 0(a0)
; CHECK-NEXT:  cjalr zero, 0(ra)
define void @move24_align16(ptr addrspace(200) %dst, ptr addrspace(200) %src) {
entry:
  call void @llvm.memmove.p200.p200.i64(ptr addrspace(200) align 16 %dst,
                                        ptr addrspace(200) align 16 %src,
                                        i64 24, i1 false)
  ret void
}

; Without 16-byte alignment known on BOTH ends the granules cannot be copied
; as capabilities inline, and a scalar copy would strip any tag they hold at
; run time, so these call memcpy (C-81; they were scalar `ld`/`sd` until
; 2026-10-16, see memcpy-unknown-alignment-tags.ll).
; CHECK-LABEL: copy24_align8:
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
; CHECK-NEXT:  li a2, 24
; CHECK-NEXT:  cjalr ra, 0(a3)
; CHECK-NEXT:  ldc ra, 0(sp)
; CHECK-NEXT:  .cfi_restore ra
; CHECK-NEXT:  cincoffsetimm sp, sp, 16
; CHECK-NEXT:  .cfi_def_cfa_offset 0
; CHECK-NEXT:  cjalr zero, 0(ra)
define void @copy24_align8(ptr addrspace(200) %dst, ptr addrspace(200) %src) {
entry:
  call void @llvm.memcpy.p200.p200.i64(ptr addrspace(200) align 8 %dst,
                                       ptr addrspace(200) align 8 %src,
                                       i64 24, i1 false)
  ret void
}

; CHECK-LABEL: copy24_dst16_src8:
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
; CHECK-NEXT:  li a2, 24
; CHECK-NEXT:  cjalr ra, 0(a3)
; CHECK-NEXT:  ldc ra, 0(sp)
; CHECK-NEXT:  .cfi_restore ra
; CHECK-NEXT:  cincoffsetimm sp, sp, 16
; CHECK-NEXT:  .cfi_def_cfa_offset 0
; CHECK-NEXT:  cjalr zero, 0(ra)
define void @copy24_dst16_src8(ptr addrspace(200) %dst, ptr addrspace(200) %src) {
entry:
  call void @llvm.memcpy.p200.p200.i64(ptr addrspace(200) align 16 %dst,
                                       ptr addrspace(200) align 8 %src,
                                       i64 24, i1 false)
  ret void
}

; A copy shorter than a granule cannot hold a capability at all.
; CHECK-LABEL: copy12_align16:
; CHECK:       # %bb.0:
; CHECK-NEXT:  lw [[T:[a-z0-9]+]], 8(a1)
; CHECK-NEXT:  sw [[T]], 8(a0)
; CHECK-NEXT:  ld [[T:[a-z0-9]+]], 0(a1)
; CHECK-NEXT:  sd [[T]], 0(a0)
; CHECK-NEXT:  cjalr zero, 0(ra)
define void @copy12_align16(ptr addrspace(200) %dst, ptr addrspace(200) %src) {
entry:
  call void @llvm.memcpy.p200.p200.i64(ptr addrspace(200) align 16 %dst,
                                       ptr addrspace(200) align 16 %src,
                                       i64 12, i1 false)
  ret void
}

; A copy too large for the 32-granule budget is not inlined at all. The granule
; count is 64-bit: narrowed to 32 bits, 2^36 + 17 bytes counted one granule and
; was inlined as a 17-byte copy; 2^36 + 16 (the 16-multiple branch) as one
; capability. MUTATION (2026-10-14): with the 32-bit count, both functions are
; inlined as one `ldc`/`stc` pair (plus a byte) and call no memcpy.
; CHECK-LABEL: copy_huge_partial_align16:
; CHECK-NOT:   ldc
; CHECK:       memcpy
; CHECK-LABEL: copy_huge_whole_align16:
; CHECK-NOT:   ldc
; CHECK:       memcpy
define void @copy_huge_partial_align16(ptr addrspace(200) %dst, ptr addrspace(200) %src) {
entry:
  call void @llvm.memcpy.p200.p200.i64(ptr addrspace(200) align 16 %dst,
                                       ptr addrspace(200) align 16 %src,
                                       i64 68719476753, i1 false)
  ret void
}

define void @copy_huge_whole_align16(ptr addrspace(200) %dst, ptr addrspace(200) %src) {
entry:
  call void @llvm.memcpy.p200.p200.i64(ptr addrspace(200) align 16 %dst,
                                       ptr addrspace(200) align 16 %src,
                                       i64 68719476752, i1 false)
  ret void
}
