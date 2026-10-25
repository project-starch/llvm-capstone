; C-81, forced inline: `llvm.memcpy.inline` (`__builtin_memcpy_inline`) may not
; become a call, so a copy of a granule or more whose 16-byte alignment is not
; known cannot be lowered without dropping capability tags. It is rejected with
; an error instead of being split into byte loads and stores without a word.
; A 16-aligned forced-inline copy is copied as capabilities however long it is:
; the 32-granule budget applies to copies that may fall back to memcpy, and past
; it a forced-inline copy used to be split into `ld`/`sd` (the C-78 residual).
; Under the experimental S-06 fixup the budget stays (its hook covers only 32
; granules), and the libcall flag no longer declines a forced-inline copy into
; an empty lowering: the error is reported there too.
;
; MUTATION (each with nothing else changed, performed 2026-10-16): with the
; C-81 block disabled, @inline16_align1 is sixteen `lbu`/`sb` pairs and llc
; exits 0, so the ERR run fails; with the 32-granule budget applied to forced
; inline copies too, @inline544 and @inline552 hold no `stc` and fail their
; counts; with the budget lifted under the S-06 fixup as well, the FIXUP run
; fails; with the libcall flag declining forced-inline copies again, its ERR
; run fails.
;
; RUN: split-file %s %t
; RUN: not llc -mtriple=capstone64 -mattr=+m < %t/reject.ll 2>&1 \
; RUN:   | FileCheck %s --check-prefix=ERR
; RUN: llc -mtriple=capstone64 -mattr=+m -verify-machineinstrs < %t/accept.ll \
; RUN:   | FileCheck %s
; RUN: %llc_cap -O0 < %t/accept.ll -o /dev/null
; RUN: not llc -mtriple=capstone64 -mattr=+m -capstone-lower-memops-via-libcall \
; RUN:   < %t/reject.ll 2>&1 | FileCheck %s --check-prefix=ERR
; RUN: llc -mtriple=capstone64 -mattr=+m -verify-machineinstrs \
; RUN:   -capstone-memcpy-high-half-fixup < %t/accept.ll \
; RUN:   | FileCheck %s --check-prefix=FIXUP

; ERR: error: Capstone: cannot inline a copy of 16 bytes whose 16-byte alignment is unknown without dropping capability tags; align both operands to 16 or call memcpy

;--- reject.ll
declare void @llvm.memcpy.inline.p200.p200.i64(ptr addrspace(200), ptr addrspace(200), i64, i1)
define void @inline16_align1(ptr addrspace(200) %dst, ptr addrspace(200) %src) {
entry:
  call void @llvm.memcpy.inline.p200.p200.i64(ptr addrspace(200) align 1 %dst,
                                              ptr addrspace(200) align 1 %src,
                                              i64 16, i1 false)
  ret void
}

;--- accept.ll
declare void @llvm.memcpy.inline.p200.p200.i64(ptr addrspace(200), ptr addrspace(200), i64, i1)

; Shorter than a granule: a byte copy, as before.
; CHECK-LABEL: inline15_align1:
; CHECK-NOT:   memcpy
; CHECK-COUNT-15: sb {{[a-z0-9]+}}, {{[0-9]+}}(a0)
; CHECK-NEXT:  cjalr zero, 0(ra)
define void @inline15_align1(ptr addrspace(200) %dst, ptr addrspace(200) %src) {
entry:
  call void @llvm.memcpy.inline.p200.p200.i64(ptr addrspace(200) align 1 %dst,
                                              ptr addrspace(200) align 1 %src,
                                              i64 15, i1 false)
  ret void
}

; 34 granules, past the 32-granule budget: 34 capability pairs, no call.
; With the S-06 fixup the budget applies and no capability pair is emitted.
; FIXUP-LABEL: inline544_align16:
; FIXUP-NOT:   stc
; FIXUP:       cjalr zero, 0(ra)
; CHECK-LABEL: inline544_align16:
; CHECK-NOT:   {{memcpy|ld |sd }}
; CHECK-COUNT-34: stc {{[a-z0-9]+}}, {{[0-9]+}}(a0)
; CHECK-NOT:   {{memcpy|ld |sd }}
; CHECK:       cjalr zero, 0(ra)
define void @inline544_align16(ptr addrspace(200) %dst, ptr addrspace(200) %src) {
entry:
  call void @llvm.memcpy.inline.p200.p200.i64(ptr addrspace(200) align 16 %dst,
                                              ptr addrspace(200) align 16 %src,
                                              i64 544, i1 false)
  ret void
}

; 34 granules and an 8-byte tail: the granules as capabilities, the tail scalar.
; CHECK-LABEL: inline552_align16:
; CHECK-NOT:   memcpy
; CHECK:       sd {{[a-z0-9]+}}, 544(a0)
; CHECK-COUNT-34: stc {{[a-z0-9]+}}, {{[0-9]+}}(a0)
; CHECK-NOT:   {{memcpy|sd }}
; CHECK:       cjalr zero, 0(ra)
define void @inline552_align16(ptr addrspace(200) %dst, ptr addrspace(200) %src) {
entry:
  call void @llvm.memcpy.inline.p200.p200.i64(ptr addrspace(200) align 16 %dst,
                                              ptr addrspace(200) align 16 %src,
                                              i64 552, i1 false)
  ret void
}
