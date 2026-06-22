; Verify that the PureCap va_list is handled as a capability, not a scalar.
;
; On Capstone PureCap the va_list holds a capability (an AS200 pointer to the
; variadic save area).  va_start must store it with `stc` and va_arg must reload
; it with `ldc`, advance it by one capability granule (CLEN = 16 bytes) with a
; `cincoffset`, and store it back with `stc`.  The earlier generic lowering used
; the AS0 pointer type (i64): it stored/reloaded the va_list with scalar `sd`/`ld`
; (dropping the capability tag, which faults on the first argument dereference in
; cap_mem mode) and advanced by the raw argument size (8 for an i64) instead of
; the 16-byte slot stride.
;
; RUN: llc -O0 -mtriple=capstone64 -mattr=+m -verify-machineinstrs < %s | FileCheck %s

target datalayout = "e-m:e-p:64:128-p200:128:128:128-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"

; va_start stores the save-area capability via `stc` (no scalar `sd` anywhere up
; to the first va_arg); va_arg then reloads the va_list via `ldc`, advances by 16
; (one capability granule) with `cincoffset`, stores it back via `stc`, and loads
; the argument value through the tagged capability.
; CHECK-LABEL: va_one:
; CHECK-NOT:   sd
; CHECK:       ldc [[P:[a-z0-9]+]], 0([[VL:[a-z0-9]+]])
; CHECK-NEXT:  cincoffsetimm {{[a-z0-9]+}}, [[P]], 16
; CHECK-NEXT:  stc {{[a-z0-9]+}}, 0([[VL]])
; CHECK-NEXT:  ld {{[a-z0-9]+}}, 0([[P]])
define i64 @va_one(i32 %n, ...) addrspace(200) {
entry:
  %ap = alloca ptr addrspace(200), align 16, addrspace(200)
  call addrspace(200) void @llvm.va_start.p200(ptr addrspace(200) %ap)
  %v = va_arg ptr addrspace(200) %ap, i64
  call addrspace(200) void @llvm.va_end.p200(ptr addrspace(200) %ap)
  ret i64 %v
}

; Repeated va_arg reads must keep using one 16-byte capability slot per
; argument.  This covers the stride bug independently from the tag-preservation
; checks above.
; CHECK-LABEL: va_two:
; CHECK:       ldc [[P0:[a-z0-9]+]], 0([[VL0:[a-z0-9]+]])
; CHECK-NEXT:  cincoffsetimm {{[a-z0-9]+}}, [[P0]], 16
; CHECK-NEXT:  stc {{[a-z0-9]+}}, 0([[VL0]])
; CHECK-NEXT:  ld {{[a-z0-9]+}}, 0([[P0]])
; CHECK:       ldc [[P1:[a-z0-9]+]], 0([[VL1:[a-z0-9]+]])
; CHECK-NEXT:  cincoffsetimm {{[a-z0-9]+}}, [[P1]], 16
; CHECK-NEXT:  stc {{[a-z0-9]+}}, 0([[VL1]])
; CHECK-NEXT:  ld {{[a-z0-9]+}}, 0([[P1]])
define i64 @va_two(i32 %n, ...) addrspace(200) {
entry:
  %ap = alloca ptr addrspace(200), align 16, addrspace(200)
  call addrspace(200) void @llvm.va_start.p200(ptr addrspace(200) %ap)
  %a = va_arg ptr addrspace(200) %ap, i64
  %b = va_arg ptr addrspace(200) %ap, i64
  %sum = add i64 %a, %b
  call addrspace(200) void @llvm.va_end.p200(ptr addrspace(200) %ap)
  ret i64 %sum
}

; va_copy copies the va_list capability with `ldc`/`stc` so the tag survives.
; CHECK-LABEL: va_copy_fn:
; CHECK:       ldc [[R:[a-z0-9]+]], 0(
; CHECK:       stc [[R]], 0(
define void @va_copy_fn(ptr addrspace(200) %dst, ptr addrspace(200) %src) addrspace(200) {
entry:
  call addrspace(200) void @llvm.va_copy.p200(ptr addrspace(200) %dst, ptr addrspace(200) %src)
  ret void
}

declare void @llvm.va_start.p200(ptr addrspace(200))
declare void @llvm.va_end.p200(ptr addrspace(200))
declare void @llvm.va_copy.p200(ptr addrspace(200), ptr addrspace(200))
