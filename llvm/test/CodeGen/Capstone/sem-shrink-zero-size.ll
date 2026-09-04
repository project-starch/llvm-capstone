; C-34 (Tier 4.2): a SHRINK whose new base equals its new end is ILLEGAL on the
; spec, QEMU and the RTL alike (cap-man-insn.adoc:227-229; op_helper.c:1045;
; capstone_flu_unit.anvil:192), so a zero-size object must not produce one.
; Measured 2026-09-05: it does not.  The frame lowering gives a zero-size
; alloca one byte, and the stack SHRINK covers that byte (the size operand is
; `li 1`, the same sequence a one-byte alloca gets); a zero-size global gets no
; SHRINK at all under -capstone-shrink-globals.  Closed without a fix; this
; file keeps it closed.  The object's base is read with the plain integer
; write (PseudoTRUNC_CAP), never `lcc rd, rs, 2` (Tier 4.2).
;
; MUTATION: change `alloca [0 x i8]` in @zero_array to `alloca [2 x i8]` -> the
; size operand becomes `li 2` and the `li {{.*}}, 1` check in that function
; fails (performed 2026-09-05).
;
; RUN: llc -mtriple=capstone64 -mattr=+m -O2 -verify-machineinstrs < %s | FileCheck %s --implicit-check-not='lcc {{.*}}, 2'
; RUN: llc -mtriple=capstone64 -mattr=+m -O0 -verify-machineinstrs < %s | FileCheck %s --implicit-check-not='lcc {{.*}}, 2'
; RUN: %llc_cap -O1 < %s -o /dev/null

declare void @use(ptr addrspace(200))

; CHECK-LABEL: zero_array:
; CHECK: li {{a[0-9]+}}, 1
; CHECK: shrink
; CHECK: cjalr ra
define void @zero_array() {
  %p = alloca [0 x i8], addrspace(200)
  call void @use(ptr addrspace(200) %p)
  ret void
}

; CHECK-LABEL: empty_struct:
; CHECK: li {{a[0-9]+}}, 1
; CHECK: shrink
define void @empty_struct() {
  %p = alloca {}, addrspace(200)
  call void @use(ptr addrspace(200) %p)
  ret void
}

; CONTROL: a one-byte object gets the same one-byte SHRINK.
; CHECK-LABEL: one_byte:
; CHECK: li {{a[0-9]+}}, 1
; CHECK: shrink
define void @one_byte() {
  %p = alloca i8, addrspace(200)
  call void @use(ptr addrspace(200) %p)
  ret void
}

; A zero-size global: materialised, delin'd, and NOT shrunk.
@g0 = addrspace(200) global [0 x i8] zeroinitializer
; CHECK-LABEL: zero_global:
; CHECK: delin
; CHECK-NOT: shrink
; CHECK: cjalr zero, 0(ra)
define ptr addrspace(200) @zero_global() {
  ret ptr addrspace(200) @g0
}
