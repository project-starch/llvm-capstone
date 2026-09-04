; The IR forms clang emits for aggregates on this target -- byval and sret of
; a struct with a capability member -- at the backend level: the member is
; stored and copied with stc/ldc, the integer member with sd/ld.  The clang
; half is clang/test/CodeGen/capstone-abi-cap-struct.c.  Measured 2026-09-04
; on the branch tools.
;
; MUTATION: make the stored value in @mk an i64 (store i64 %n) -> `sd` replaces
; the `stc a1, 0(a0)` line and it fails (performed 2026-09-04, as the clang
; half's member-type mutation).
;
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -verify-machineinstrs < %s | FileCheck %s
; RUN: %llc_cap -O0 < %s -o /dev/null
; RUN: %llc_cap -O1 < %s -o /dev/null

%struct.S = type { ptr addrspace(200), i64 }

declare void @llvm.memcpy.p200.p200.i64(ptr addrspace(200), ptr addrspace(200), i64, i1)

; CHECK-LABEL: mk:
; CHECK: stc a1, 0(a0)
; CHECK-NEXT: sd a2, 16(a0)
; CHECK-NEXT: cjalr zero, 0(ra)
define void @mk(ptr addrspace(200) sret(%struct.S) align 16 %r, ptr addrspace(200) %p, i64 %n) {
  store ptr addrspace(200) %p, ptr addrspace(200) %r, align 16
  %np = getelementptr inbounds %struct.S, ptr addrspace(200) %r, i32 0, i32 1
  store i64 %n, ptr addrspace(200) %np, align 16
  ret void
}

; CHECK-LABEL: pass:
; CHECK: ldc a2, 16(a1)
; CHECK-NEXT: ldc a1, 0(a1)
; CHECK-NEXT: stc a2, 16(a0)
; CHECK-NEXT: stc a1, 0(a0)
; CHECK-NEXT: cjalr zero, 0(ra)
define void @pass(ptr addrspace(200) sret(%struct.S) align 16 %r, ptr addrspace(200) byval(%struct.S) align 16 %s) {
  call void @llvm.memcpy.p200.p200.i64(ptr addrspace(200) align 16 %r, ptr addrspace(200) align 16 %s, i64 32, i1 false)
  ret void
}

; CHECK-LABEL: first:
; CHECK: ldc a0, 0(a0)
; CHECK-NEXT: cjalr zero, 0(ra)
define ptr addrspace(200) @first(ptr addrspace(200) byval(%struct.S) align 16 %s) {
  %p = load ptr addrspace(200), ptr addrspace(200) %s, align 16
  ret ptr addrspace(200) %p
}
