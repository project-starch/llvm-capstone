; The S-06 memcpy workaround family (CapstoneSelectionDAGInfo.cpp) and the
; libcall knob that predates it (-capstone-lower-memops-via-libcall), every knob
; at both values on one 32-byte, 16-byte-aligned copy.  With the fixup off the copy is
; two ldc/stc pairs and nothing else.  With it on, per 16-byte chunk, both halves are
; loaded and plain-stored FIRST, then the ldc/stc is laid on top.  The
; max-bytes knob below the copy size switches the fixup off again; the two
; DIAGNOSTIC knobs drop the trailing stc, or turn the plain loads into `ld zero`
; and drop the plain stores.  Measured 2026-09-04 on the branch tools.
;
; MUTATION: the arms are each other's mutation -- the OFF arm's implicit-check-
; not on `sd a` fires on the ON arm's output, and the ON arm's exact chain fails
; on the OFF output (performed 2026-09-04 by running both).
;
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 < %s | FileCheck %s --check-prefix=OFF --implicit-check-not='ld a' --implicit-check-not='sd a'
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -capstone-memcpy-high-half-fixup=false < %s | FileCheck %s --check-prefix=OFF --implicit-check-not='ld a' --implicit-check-not='sd a'
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -capstone-memcpy-high-half-fixup=true -capstone-memcpy-high-half-fixup-max-bytes=512 -capstone-memcpy-fixup-no-stc=false -capstone-memcpy-fixup-no-plain-stores=false < %s | FileCheck %s --check-prefix=ON
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -capstone-memcpy-high-half-fixup -capstone-memcpy-high-half-fixup-max-bytes=16 < %s | FileCheck %s --check-prefix=OFF --implicit-check-not='ld a' --implicit-check-not='sd a'
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -capstone-memcpy-high-half-fixup -capstone-memcpy-fixup-no-stc=true < %s | FileCheck %s --check-prefix=NOSTC --implicit-check-not='stc a'
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -capstone-memcpy-high-half-fixup -capstone-memcpy-fixup-no-plain-stores=true < %s | FileCheck %s --check-prefix=NOPLAIN --implicit-check-not='sd a'
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -capstone-lower-memops-via-libcall=true < %s | FileCheck %s --check-prefix=LIBCALL --implicit-check-not='ldc a' --implicit-check-not='stc a'
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -capstone-lower-memops-via-libcall=false < %s | FileCheck %s --check-prefix=OFF --implicit-check-not='ld a' --implicit-check-not='sd a'
; RUN: %llc_cap -O0 < %s -o /dev/null
; RUN: %llc_cap -O1 < %s -o /dev/null

; OFF-LABEL: copy32:
; OFF: ldc a2, 16(a1)
; OFF-NEXT: stc a2, 16(a0)
; OFF-NEXT: ldc a1, 0(a1)
; OFF-NEXT: stc a1, 0(a0)
; OFF-NEXT: cjalr zero, 0(ra)

; ON-LABEL: copy32:
; ON: ld a2, 0(a1)
; ON-NEXT: ld a3, 8(a1)
; ON-NEXT: ldc a4, 0(a1)
; ON-NEXT: sd a2, 0(a0)
; ON-NEXT: sd a3, 8(a0)
; ON-NEXT: stc a4, 0(a0)
; ON-NEXT: ld a2, 16(a1)
; ON-NEXT: ld a3, 24(a1)
; ON-NEXT: ldc a1, 16(a1)
; ON-NEXT: sd a2, 16(a0)
; ON-NEXT: sd a3, 24(a0)
; ON-NEXT: stc a1, 16(a0)
; ON-NEXT: cjalr zero, 0(ra)

; NOSTC-LABEL: copy32:
; NOSTC: ld a2, 0(a1)
; NOSTC-NEXT: ld a3, 8(a1)
; NOSTC-NEXT: ldc a4, 0(a1)
; NOSTC-NEXT: sd a2, 0(a0)
; NOSTC-NEXT: sd a3, 8(a0)
; NOSTC-NEXT: ld a2, 16(a1)
; NOSTC-NEXT: ld a3, 24(a1)
; NOSTC-NEXT: ldc a1, 16(a1)
; NOSTC-NEXT: sd a2, 16(a0)
; NOSTC-NEXT: sd a3, 24(a0)
; NOSTC-NEXT: cjalr zero, 0(ra)

; NOPLAIN-LABEL: copy32:
; NOPLAIN: ld zero, 0(a1)
; NOPLAIN-NEXT: ld zero, 8(a1)
; NOPLAIN-NEXT: ldc a2, 0(a1)
; NOPLAIN-NEXT: stc a2, 0(a0)
; NOPLAIN-NEXT: ld zero, 16(a1)
; NOPLAIN-NEXT: ld zero, 24(a1)
; NOPLAIN-NEXT: ldc a1, 16(a1)
; NOPLAIN-NEXT: stc a1, 16(a0)
; NOPLAIN-NEXT: cjalr zero, 0(ra)

define void @copy32(ptr addrspace(200) %d, ptr addrspace(200) %s) {
  call void @llvm.memcpy.p200.p200.i64(ptr addrspace(200) align 16 %d, ptr addrspace(200) align 16 %s, i64 32, i1 false)
  ret void
}
declare void @llvm.memcpy.p200.p200.i64(ptr addrspace(200), ptr addrspace(200), i64, i1)

; With -capstone-lower-memops-via-libcall the copy is a call to memcpy and no
; inline capability load or store remains (the callee address is derived from
; gp, like every code address under the default ABI).
; LIBCALL-LABEL: copy32:
; LIBCALL: %pcrel_hi(memcpy)
; LIBCALL: li a2, 32
; LIBCALL-NEXT: cjalr ra, 0(a3)
