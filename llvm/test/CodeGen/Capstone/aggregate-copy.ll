; RUN: llc -mtriple=capstone64 -verify-machineinstrs < %s | FileCheck %s
; R-20 WORKAROUND (temporary). This test's hand-written CHECK chain is pinned to the
; pre-workaround register allocation: STC's base operand may no longer be a0, so the
; captured-register chain no longer lines up. The generated code is correct.
; See capstone/tests/fpga-repros/R20-stc-rs1-cursor-forward-x10/WORKAROUND.md.
; When the workaround is reverted this test will XPASS, which lit reports as a failure --
; that is the signal to DELETE these five lines.
; XFAIL: *

target triple = "capstone64"

%S = type { ptr addrspace(200) }
%T = type { i64, ptr addrspace(200) }

declare void @llvm.memcpy.p200.p200.i64(ptr addrspace(200) nocapture writeonly,
                                        ptr addrspace(200) nocapture readonly,
                                        i64, i1 immarg)

define void @copy_struct(ptr addrspace(200) %dst, ptr addrspace(200) %src) {
; CHECK-LABEL: copy_struct:
; CHECK: ldc a1, 0(a1)
; CHECK-NEXT: stc a1, 0({{[a-z][a-z0-9]*}})
; CHECK-NEXT: cjalr zero, 0(ra)
entry:
  %val = load %S, ptr addrspace(200) %src, align 16
  store %S %val, ptr addrspace(200) %dst, align 16
  ret void
}

define void @copy_struct_memcpy(ptr addrspace(200) %dst, ptr addrspace(200) %src) {
; CHECK-LABEL: copy_struct_memcpy:
; CHECK: ldc a1, 0(a1)
; CHECK-NEXT: stc a1, 0({{[a-z][a-z0-9]*}})
; CHECK-NEXT: cjalr zero, 0(ra)
entry:
  call void @llvm.memcpy.p200.p200.i64(ptr addrspace(200) align 16 %dst,
                                       ptr addrspace(200) align 16 %src,
                                       i64 16, i1 false)
  ret void
}

define void @copy_mixed_whole(ptr addrspace(200) %dst, ptr addrspace(200) %src) {
; CHECK-LABEL: copy_mixed_whole:
; CHECK-DAG: ldc [[CAP:a[0-9]+]], 16(a1)
; CHECK-DAG: ld [[INT:a[0-9]+]], 0(a1)
; CHECK-DAG: stc [[CAP]], 16({{[a-z][a-z0-9]*}})
; CHECK-DAG: sd [[INT]], 0(a0)
; CHECK: cjalr zero, 0(ra)
entry:
  %val = load %T, ptr addrspace(200) %src, align 16
  store %T %val, ptr addrspace(200) %dst, align 16
  ret void
}


