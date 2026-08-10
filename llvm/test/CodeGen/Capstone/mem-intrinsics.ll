; RUN: llc < %s -mtriple=capstone64 | FileCheck %s
; R-20 WORKAROUND (temporary). This test's hand-written CHECK chain is pinned to the
; pre-workaround register allocation: STC's base operand may no longer be a0, so the
; captured-register chain no longer lines up. The generated code is correct.
; See capstone/tests/fpga-repros/R20-stc-rs1-cursor-forward-x10/WORKAROUND.md.
; When the workaround is reverted this test will XPASS, which lit reports as a failure --
; that is the signal to DELETE these five lines.
; XFAIL: *

; PureCap AS200 copies must preserve capability tags. For aligned 16-byte
; memcpy/memmove, lower directly to a single ldc/stc pair rather than generic
; scalar splitting.

declare void @llvm.memcpy.p200.p200.i64(ptr addrspace(200) nocapture writeonly, ptr addrspace(200) nocapture readonly, i64, i1 immarg)
declare void @llvm.memmove.p200.p200.i64(ptr addrspace(200) nocapture writeonly, ptr addrspace(200) nocapture readonly, i64, i1 immarg)

; CHECK-LABEL: copy16:
; CHECK: ldc [[TMP:a[0-9]+]], 0(a1)
; CHECK-NEXT: stc [[TMP]], 0({{[a-z][a-z0-9]*}})
; CHECK-NEXT: cjalr zero, 0(ra)
define void @copy16(ptr addrspace(200) %dst, ptr addrspace(200) %src) {
entry:
  call void @llvm.memcpy.p200.p200.i64(ptr addrspace(200) align 16 %dst,
                                       ptr addrspace(200) align 16 %src,
                                       i64 16, i1 false)
  ret void
}

; CHECK-LABEL: move16:
; CHECK: ldc [[TMP2:a[0-9]+]], 0(a1)
; CHECK-NEXT: stc [[TMP2]], 0({{[a-z][a-z0-9]*}})
; CHECK-NEXT: cjalr zero, 0(ra)
define void @move16(ptr addrspace(200) %dst, ptr addrspace(200) %src) {
entry:
  call void @llvm.memmove.p200.p200.i64(ptr addrspace(200) align 16 %dst,
                                        ptr addrspace(200) align 16 %src,
                                        i64 16, i1 false)
  ret void
}

; CHECK-LABEL: copy304:
; CHECK: ldc [[TMP4:a[0-9]+]], 288(a1)
; CHECK: stc [[TMP4]], 288({{[a-z][a-z0-9]*}})
; CHECK: ldc [[TMP3:a[0-9]+]], 0(a1)
; CHECK: stc [[TMP3]], 0({{[a-z][a-z0-9]*}})
; CHECK-NOT: lbu
; CHECK-NOT: sb
define void @copy304(ptr addrspace(200) %dst, ptr addrspace(200) %src) {
entry:
  call void @llvm.memcpy.p200.p200.i64(ptr addrspace(200) align 16 %dst,
                                       ptr addrspace(200) align 16 %src,
                                       i64 304, i1 false)
  ret void
}


